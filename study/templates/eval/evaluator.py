from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import torch
from pipeline.shared import SQUARE_CLASS_NAMES
from study.templates.inference.embedder import get_embedder
from study.templates.inference.template_match import TemplateMatchResult, classify_embedding
from study.templates.proposals.common import ProposalFrame, SquareCropProposal
from study.templates.proposals.cuboid import propose_cuboid
from study.templates.proposals.sam3_source import propose_sam3
from study.templates.shared import PROJECT_ROOT, load_base_head_data_module

_PROPOSAL_SOURCES = {
    "cuboid": propose_cuboid,
    "sam3": propose_sam3,
}
_PIECE_TYPE_TO_LABEL = {
    piece_type: label_index
    for label_index, piece_type in enumerate(SQUARE_CLASS_NAMES)
    if label_index > 0
}


@dataclass
class CategoryAccumulator:
    count: int = 0
    strict_exact: int = 0
    placed_exact: int = 0
    square_correct: int = 0
    square_total: int = 0
    tp_by_piece: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fp_by_piece: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fn_by_piece: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failure_examples: list[str] = field(default_factory=list)

    def add(
        self,
        *,
        predicted_labels: tuple[int, ...],
        gt_labels: tuple[int, ...],
        predicted_pieces: tuple[tuple[str, str | None], ...],
        gt_pieces: tuple[tuple[str, str | None], ...],
    ) -> None:
        self.count += 1
        if predicted_pieces == gt_pieces:
            self.strict_exact += 1
        if predicted_labels == gt_labels:
            self.placed_exact += 1
        self.square_total += len(gt_labels)
        self.square_correct += sum(
            int(predicted == gt) for predicted, gt in zip(predicted_labels, gt_labels)
        )

        predicted_set = {item for item in predicted_pieces if item[1] is not None}
        gt_set = {item for item in gt_pieces if item[1] is not None}
        for piece_name in SQUARE_CLASS_NAMES[1:]:
            predicted_piece_set = {item for item in predicted_set if item[0] == piece_name}
            gt_piece_set = {item for item in gt_set if item[0] == piece_name}
            self.tp_by_piece[piece_name] += len(predicted_piece_set & gt_piece_set)
            self.fp_by_piece[piece_name] += len(predicted_piece_set - gt_piece_set)
            self.fn_by_piece[piece_name] += len(gt_piece_set - predicted_piece_set)

    def finalize(self) -> dict[str, object]:
        per_piece_f1: dict[str, float] = {}
        for piece_name in SQUARE_CLASS_NAMES[1:]:
            tp = self.tp_by_piece[piece_name]
            fp = self.fp_by_piece[piece_name]
            fn = self.fn_by_piece[piece_name]
            denom = (2 * tp) + fp + fn
            per_piece_f1[piece_name] = (2 * tp / denom) if denom else 0.0
        return {
            "count": self.count,
            "strict_piece_exact_match": self.strict_exact / max(self.count, 1),
            "placed_board_exact_match": self.placed_exact / max(self.count, 1),
            "per_square_accuracy": self.square_correct / max(self.square_total, 1),
            "piece_f1_macro": mean(per_piece_f1.values()) if per_piece_f1 else 0.0,
            "per_piece_f1": per_piece_f1,
            "failure_examples": list(self.failure_examples),
        }


def evaluate_template_matching(
    *,
    template_bank: dict[str, Any],
    proposal_source: str,
    output_dir: str | Path,
    eval_labels: str | Path = PROJECT_ROOT / "study" / "eval" / "labels.jsonl",
    device: str = "cpu",
    max_frames: int | None = None,
    source_video_id: str | None = None,
    match_threshold: float = 0.75,
    failures_per_category: int = 4,
) -> dict[str, dict[str, object]]:
    base_head_data = load_base_head_data_module()
    eval_records = base_head_data.load_eval_records(eval_labels)
    if source_video_id is not None:
        eval_records = [
            record for record in eval_records if record.source_video_id == source_video_id
        ]
    if max_frames is not None:
        eval_records = eval_records[:max_frames]
    if not eval_records:
        raise ValueError("No eval records selected")

    proposal_fn = _proposal_function(proposal_source)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    failure_root = output_root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)

    accumulators: dict[str, CategoryAccumulator] = defaultdict(CategoryAccumulator)
    accumulators["overall"] = CategoryAccumulator()
    per_frame: list[dict[str, object]] = []

    for record in eval_records:
        image_bgr = cv2.imread(
            str(base_head_data.resolve_project_path(record.image_path)),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise ValueError(f"Failed to read eval image: {record.image_path}")
        frame = ProposalFrame(image_bgr=image_bgr, corners=record.corners, frame_id=record.frame_id)
        proposals = proposal_fn(frame)
        matches = _classify_proposals(proposals, template_bank=template_bank, device=device)
        predicted_labels = _predicted_board_labels(
            proposals,
            matches,
            match_threshold=match_threshold,
        )
        predicted_pieces = base_head_data.labels_to_piece_tuples(predicted_labels)
        gt_pieces = base_head_data.placed_piece_tuples(record.pieces)

        accumulators[record.category].add(
            predicted_labels=predicted_labels,
            gt_labels=record.placed_labels,
            predicted_pieces=predicted_pieces,
            gt_pieces=gt_pieces,
        )
        accumulators["overall"].add(
            predicted_labels=predicted_labels,
            gt_labels=record.placed_labels,
            predicted_pieces=predicted_pieces,
            gt_pieces=gt_pieces,
        )

        strict_exact = predicted_pieces == gt_pieces
        placed_exact = predicted_labels == record.placed_labels
        failure_example = None
        if (not strict_exact or not placed_exact) and len(
            accumulators[record.category].failure_examples
        ) < failures_per_category:
            failure_example = _save_failure_example(
                output_root=failure_root,
                record=record,
                image_bgr=image_bgr,
                predicted_pieces=predicted_pieces,
                gt_pieces=gt_pieces,
                proposal_count=len(proposals),
                strict_exact=strict_exact,
                placed_exact=placed_exact,
            )
            accumulators[record.category].failure_examples.append(failure_example)
            accumulators["overall"].failure_examples.append(failure_example)

        per_square_accuracy = _frame_per_square_accuracy(predicted_labels, record.placed_labels)
        piece_f1_macro = _frame_piece_f1_macro(predicted_pieces, gt_pieces)
        per_frame.append(
            {
                "frame_id": record.frame_id,
                "category": record.category,
                "strict_piece_exact": strict_exact,
                "placed_board_exact": placed_exact,
                "per_square_accuracy": per_square_accuracy,
                "piece_f1_macro": piece_f1_macro,
                "predicted_labels": list(predicted_labels),
                "gt_labels": list(record.placed_labels),
                "predicted_pieces": list(predicted_pieces),
                "gt_pieces": list(gt_pieces),
                "proposal_count": len(proposals),
                "failure_example": failure_example,
            }
        )

    category_metrics = {
        category: accumulator.finalize() for category, accumulator in accumulators.items()
    }
    macro_categories = [category for category in category_metrics if category != "overall"]
    category_metrics["macro"] = {
        "strict_piece_exact_match": mean(
            float(category_metrics[category]["strict_piece_exact_match"])
            for category in macro_categories
        ),
        "placed_board_exact_match": mean(
            float(category_metrics[category]["placed_board_exact_match"])
            for category in macro_categories
        ),
        "per_square_accuracy": mean(
            float(category_metrics[category]["per_square_accuracy"])
            for category in macro_categories
        ),
        "piece_f1_macro": mean(
            float(category_metrics[category]["piece_f1_macro"])
            for category in macro_categories
        ),
    }

    (output_root / "metrics.json").write_text(
        json.dumps(category_metrics, indent=2, sort_keys=True)
    )
    (output_root / "per_frame.json").write_text(json.dumps(per_frame, indent=2))
    (output_root / "run_config.json").write_text(
        json.dumps(
            {
                "proposal_source": proposal_source,
                "device": device,
                "max_frames": max_frames,
                "source_video_id": source_video_id,
                "match_threshold": match_threshold,
                "template_bank_path": template_bank.get("_loaded_from"),
                "frame_count": len(eval_records),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return category_metrics


def _proposal_function(proposal_source: str):
    proposal_fn = _PROPOSAL_SOURCES.get(proposal_source)
    if proposal_fn is None:
        raise ValueError(
            f"proposal_source must be one of {sorted(_PROPOSAL_SOURCES)}, got {proposal_source!r}"
        )
    return proposal_fn


def _classify_proposals(
    proposals: list[SquareCropProposal],
    *,
    template_bank: dict[str, Any],
    device: str,
) -> list[TemplateMatchResult]:
    if not proposals:
        return []
    encoder_config = dict(template_bank.get("encoder_config", {}))
    embedder = get_embedder(
        encoder_type=str(encoder_config.get("encoder_type", "dinov3")),
        model_name=(
            None
            if encoder_config.get("model_name") is None
            else str(encoder_config.get("model_name"))
        ),
        input_size=int(encoder_config.get("input_size", 224)),
        device=device,
    )
    embeddings = embedder.embed_many([proposal.crop_bgr for proposal in proposals])
    return [classify_embedding(embedding, template_bank) for embedding in embeddings]


def _predicted_board_labels(
    proposals: list[SquareCropProposal],
    matches: list[TemplateMatchResult],
    *,
    match_threshold: float,
) -> tuple[int, ...]:
    labels = [0] * 64
    for proposal, match in zip(proposals, matches, strict=False):
        if match.confidence < match_threshold:
            continue
        square_index = _square_name_to_index(proposal.square)
        labels[square_index] = _PIECE_TYPE_TO_LABEL[match.piece_type]
    return tuple(labels)


def _square_name_to_index(square_name: str) -> int:
    file_index = ord(square_name[0]) - ord("a")
    rank = int(square_name[1])
    row_index = 8 - rank
    return row_index * 8 + file_index


def _save_failure_example(
    *,
    output_root: Path,
    record: Any,
    image_bgr,
    predicted_pieces: tuple[tuple[str, str | None], ...],
    gt_pieces: tuple[tuple[str, str | None], ...],
    proposal_count: int,
    strict_exact: bool,
    placed_exact: bool,
) -> str:
    category_dir = output_root / record.category
    category_dir.mkdir(parents=True, exist_ok=True)
    rendered = image_bgr.copy()
    corners = torch.tensor(record.corners, dtype=torch.float32).numpy().astype("int32")
    cv2.polylines(rendered, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    overlay_lines = [
        f"frame={record.frame_id} category={record.category}",
        (
            f"strict_exact={int(strict_exact)} "
            f"placed_exact={int(placed_exact)} proposals={proposal_count}"
        ),
        f"pred={_piece_summary(predicted_pieces)}",
        f"gt={_piece_summary(gt_pieces)}",
    ]
    y_offset = 28
    for line in overlay_lines:
        cv2.putText(
            rendered,
            line,
            (12, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 28
    output_path = category_dir / f"{record.frame_id}.png"
    cv2.imwrite(str(output_path), rendered)
    return str(output_path.relative_to(PROJECT_ROOT))


def _piece_summary(pieces: tuple[tuple[str, str | None], ...]) -> str:
    if not pieces:
        return "-"
    return ", ".join(f"{piece}@{square or 'no_square'}" for piece, square in pieces)


def _frame_per_square_accuracy(
    predicted_labels: tuple[int, ...],
    gt_labels: tuple[int, ...],
) -> float:
    return sum(int(predicted == gt) for predicted, gt in zip(predicted_labels, gt_labels)) / max(
        len(gt_labels),
        1,
    )


def _frame_piece_f1_macro(
    predicted_pieces: tuple[tuple[str, str | None], ...],
    gt_pieces: tuple[tuple[str, str | None], ...],
) -> float:
    predicted_set = {item for item in predicted_pieces if item[1] is not None}
    gt_set = {item for item in gt_pieces if item[1] is not None}
    f1_scores: list[float] = []
    for piece_name in SQUARE_CLASS_NAMES[1:]:
        predicted_piece_set = {item for item in predicted_set if item[0] == piece_name}
        gt_piece_set = {item for item in gt_set if item[0] == piece_name}
        tp = len(predicted_piece_set & gt_piece_set)
        fp = len(predicted_piece_set - gt_piece_set)
        fn = len(gt_piece_set - predicted_piece_set)
        denom = (2 * tp) + fp + fn
        f1_scores.append((2 * tp / denom) if denom else 0.0)
    return mean(f1_scores) if f1_scores else 0.0


__all__ = [
    "CategoryAccumulator",
    "evaluate_template_matching",
]
