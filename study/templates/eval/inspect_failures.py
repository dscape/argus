#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from study.templates.builder.template_bank import rebuild_template_crop
from study.templates.inference.embedder import get_embedder
from study.templates.inference.template_match import (
    TemplateInstanceMatch,
    classify_embedding,
    load_template_bank,
    top_template_matches_for_embedding,
)
from study.templates.proposals.common import ProposalFrame, SquareCropProposal
from study.templates.proposals.cuboid import propose_cuboid
from study.templates.proposals.sam3_source import propose_sam3
from study.templates.shared import load_base_head_data_module

_PROPOSAL_SOURCES = {
    "cuboid": propose_cuboid,
    "sam3": propose_sam3,
}


@dataclass(frozen=True)
class InspectionRow:
    square: str
    gt_label: str
    predicted_label: str
    status: str
    crop_bgr: np.ndarray
    top_matches: list[TemplateInstanceMatch]


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    run_config = json.loads((run_dir / "run_config.json").read_text())
    template_bank = load_template_bank(args.template_bank or run_config["template_bank_path"])
    per_frame = json.loads((run_dir / "per_frame.json").read_text())
    eval_labels = Path(args.eval_labels or (_PROJECT_ROOT / "study" / "eval" / "labels.jsonl"))

    inspect_failures(
        run_dir=run_dir,
        template_bank=template_bank,
        proposal_source=str(run_config["proposal_source"]),
        eval_labels=eval_labels,
        device=str(run_config.get("device", "cpu")),
        match_threshold=float(run_config.get("match_threshold", 0.75)),
        max_frames=args.max_frames,
        max_rows_per_frame=args.max_rows_per_frame,
        per_frame_payloads=per_frame,
    )


def inspect_failures(
    *,
    run_dir: Path,
    template_bank: dict[str, Any],
    proposal_source: str,
    eval_labels: Path,
    device: str,
    match_threshold: float,
    max_frames: int,
    max_rows_per_frame: int,
    per_frame_payloads: list[dict[str, Any]] | None = None,
) -> None:
    base_head_data = load_base_head_data_module()
    eval_records = {
        record.frame_id: record for record in base_head_data.load_eval_records(eval_labels)
    }
    ranked_frames = sorted(
        per_frame_payloads or json.loads((run_dir / "per_frame.json").read_text()),
        key=lambda payload: (
            float(payload.get("piece_f1_macro", 0.0)),
            float(payload.get("per_square_accuracy", 0.0)),
            payload.get("frame_id", ""),
        ),
    )
    selected_frames = ranked_frames[:max_frames]
    output_root = run_dir / "failures" / "inspection"
    output_root.mkdir(parents=True, exist_ok=True)

    proposal_fn = _PROPOSAL_SOURCES[proposal_source]
    embedder = get_embedder(
        encoder_type=str(template_bank.get("encoder_config", {}).get("encoder_type", "dinov3")),
        model_name=template_bank.get("encoder_config", {}).get("model_name"),
        input_size=int(template_bank.get("encoder_config", {}).get("input_size", 224)),
        device=device,
    )

    for rank, payload in enumerate(selected_frames, start=1):
        frame_id = str(payload["frame_id"])
        record = eval_records[frame_id]
        image_bgr = cv2.imread(
            str(base_head_data.resolve_project_path(record.image_path)),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise ValueError(f"Failed to read eval image: {record.image_path}")
        proposals = proposal_fn(
            ProposalFrame(image_bgr=image_bgr, corners=record.corners, frame_id=frame_id)
        )
        proposal_by_square = {proposal.square: proposal for proposal in proposals}
        proposal_embeddings = embedder.embed_many([proposal.crop_bgr for proposal in proposals])
        match_by_square = {}
        for proposal, embedding in zip(proposals, proposal_embeddings, strict=False):
            match_by_square[proposal.square] = (
                classify_embedding(embedding, template_bank),
                top_template_matches_for_embedding(embedding, template_bank, top_k=3),
                embedding,
            )

        rows = _collect_inspection_rows(
            base_head_data=base_head_data,
            record=record,
            image_bgr=image_bgr,
            proposal_by_square=proposal_by_square,
            match_by_square=match_by_square,
            template_bank=template_bank,
            embedder=embedder,
            match_threshold=match_threshold,
            max_rows=max_rows_per_frame,
        )
        rendered = _render_frame_inspection(
            frame_id=frame_id,
            category=record.category,
            proposal_source=proposal_source,
            rank=rank,
            total=len(selected_frames),
            rows=rows,
            frame_summary=payload,
            template_bank=template_bank,
        )
        output_path = output_root / f"{rank:02d}_{frame_id}.png"
        cv2.imwrite(str(output_path), rendered)


def _collect_inspection_rows(
    *,
    base_head_data: Any,
    record: Any,
    image_bgr: np.ndarray,
    proposal_by_square: dict[str, SquareCropProposal],
    match_by_square: dict[str, tuple[Any, list[TemplateInstanceMatch], Any]],
    template_bank: dict[str, Any],
    embedder: Any,
    match_threshold: float,
    max_rows: int,
) -> list[InspectionRow]:
    rows: list[tuple[int, InspectionRow]] = []
    for square_index, gt_label_index in enumerate(record.placed_labels):
        square_name = base_head_data.index_to_square_name(square_index)
        gt_label = base_head_data.SQUARE_CLASS_NAMES[int(gt_label_index)]
        proposal = proposal_by_square.get(square_name)
        match_bundle = match_by_square.get(square_name)
        predicted_label = "empty"
        status = ""
        crop_bgr: np.ndarray | None = None
        top_matches: list[TemplateInstanceMatch] = []

        if proposal is not None and match_bundle is not None:
            match_result, top_matches, _embedding = match_bundle
            crop_bgr = proposal.crop_bgr
            if match_result.confidence >= match_threshold:
                predicted_label = match_result.piece_type
            if predicted_label != gt_label:
                status = "wrong-piece" if gt_label != "empty" else "false-positive"
        elif gt_label != "empty":
            crop_bgr = _cuboid_crop_for_square(
                base_head_data,
                image_bgr,
                record.corners,
                square_index,
            )
            embedding = embedder.embed(crop_bgr)
            top_matches = top_template_matches_for_embedding(embedding, template_bank, top_k=3)
            predicted_label = "empty"
            status = "missing-proposal"

        if not status or crop_bgr is None:
            continue

        priority = 0 if gt_label != "empty" else 1
        rows.append(
            (
                priority,
                InspectionRow(
                    square=square_name,
                    gt_label=gt_label,
                    predicted_label=predicted_label,
                    status=status,
                    crop_bgr=crop_bgr,
                    top_matches=top_matches,
                ),
            )
        )

    rows.sort(key=lambda item: (item[0], item[1].square))
    return [row for _priority, row in rows[:max_rows]]


def _cuboid_crop_for_square(
    base_head_data: Any,
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    square_index: int,
) -> np.ndarray:
    return base_head_data.extract_study_piece_crop(
        image_bgr,
        corners,
        row=square_index // 8,
        col=square_index % 8,
        output_size=224,
        piece_height=2.0,
        flip_left_half=True,
    )


def _render_frame_inspection(
    *,
    frame_id: str,
    category: str,
    proposal_source: str,
    rank: int,
    total: int,
    rows: list[InspectionRow],
    frame_summary: dict[str, Any],
    template_bank: dict[str, Any],
) -> np.ndarray:
    cell_size = 160
    header_height = 84
    row_header_width = 280
    image_columns = 4
    row_height = cell_size + 26
    width = row_header_width + (image_columns * cell_size)
    height = header_height + max(1, len(rows)) * row_height
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)

    summary_line_1 = f"#{rank}/{total} | {frame_id} | {category} | source={proposal_source}"
    summary_line_2 = (
        f"piece_f1={frame_summary.get('piece_f1_macro', 0.0):.4f} | "
        f"square_acc={frame_summary.get('per_square_accuracy', 0.0):.4f} | "
        f"strict={int(frame_summary.get('strict_piece_exact', False))} | "
        f"placed={int(frame_summary.get('placed_board_exact', False))}"
    )
    cv2.putText(
        canvas,
        summary_line_1,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        summary_line_2,
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "columns: input crop | top1 | top2 | top3 template matches",
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if not rows:
        cv2.putText(
            canvas,
            "no mismatched proposal crops available",
            (12, header_height + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas

    for row_index, row in enumerate(rows):
        top = header_height + row_index * row_height
        header_lines = [
            f"{row.square} | gt={row.gt_label} pred={row.predicted_label}",
            f"status={row.status}",
        ]
        for line_index, line in enumerate(header_lines):
            cv2.putText(
                canvas,
                line,
                (12, top + 28 + line_index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        panels = [row.crop_bgr] + [
            rebuild_template_crop(template_bank, match.metadata)
            for match in row.top_matches[:3]
        ]
        labels = ["input"] + [
            f"{match.piece_type} {match.similarity:.3f}" for match in row.top_matches[:3]
        ]
        while len(panels) < image_columns:
            panels.append(np.zeros_like(row.crop_bgr))
            labels.append("-")

        for panel_index, (panel, label) in enumerate(zip(panels, labels, strict=False)):
            resized = cv2.resize(panel, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
            left = row_header_width + panel_index * cell_size
            canvas[top : top + cell_size, left : left + cell_size] = resized
            cv2.putText(
                canvas,
                label,
                (left + 6, top + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the worst template-matching failures.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--template-bank", type=Path, default=None)
    parser.add_argument("--eval-labels", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-rows-per-frame", type=int, default=6)
    return parser


if __name__ == "__main__":
    main()
