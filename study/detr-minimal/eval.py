#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import cv2
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from data import (
    EvalFrameRecord,
    load_eval_records,
    placed_piece_tuples,
    preprocess_board_neighborhood_image,
    resolve_project_path,
)
from model import decode_predictions, load_checkpoint
from pipeline.shared import SQUARE_CLASS_NAMES


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
        self.square_correct += sum(int(p == g) for p, g in zip(predicted_labels, gt_labels))
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


def main() -> None:
    args = build_parser().parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(payload.get("image_size", 224))
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device=device)
    eval_records = load_eval_records(args.eval_labels)

    output_dir = (args.output_dir or (_THIS_DIR / "runs" / args.run_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_root = output_dir / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)

    accumulators: dict[str, CategoryAccumulator] = defaultdict(CategoryAccumulator)
    accumulators["overall"] = CategoryAccumulator()
    per_frame: list[dict[str, object]] = []

    for record in eval_records:
        image = cv2.imread(str(resolve_project_path(record.image_path)), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read eval image: {record.image_path}")
        tensor, _scaled_corners = preprocess_board_neighborhood_image(
            image,
            record.corners,
            size=image_size,
        )
        batch = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(batch)
        decoded = decode_predictions(outputs, presence_threshold=args.presence_threshold)[0]
        predicted_labels = tuple(int(value) for value in decoded["board_labels"])
        predicted_pieces = tuple(sorted(decoded["pieces"]))
        gt_pieces = placed_piece_tuples(record.pieces)

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
        example_path = None
        if (not strict_exact or not placed_exact) and len(
            accumulators[record.category].failure_examples
        ) < args.failures_per_category:
            example_path = save_failure_example(
                output_root=failure_root,
                record=record,
                image=image,
                predicted_pieces=predicted_pieces,
                strict_exact=strict_exact,
                placed_exact=placed_exact,
            )
            accumulators[record.category].failure_examples.append(example_path)
            accumulators["overall"].failure_examples.append(example_path)

        per_frame.append(
            {
                "frame_id": record.frame_id,
                "category": record.category,
                "strict_piece_exact": strict_exact,
                "placed_board_exact": placed_exact,
                "predicted_pieces": list(predicted_pieces),
                "gt_pieces": list(gt_pieces),
                "failure_example": example_path,
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
            float(category_metrics[category]["piece_f1_macro"]) for category in macro_categories
        ),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(category_metrics, indent=2, sort_keys=True))
    (output_dir / "per_frame.json").write_text(json.dumps(per_frame, indent=2))
    results_path = _THIS_DIR / "RESULTS.md"
    results_path.write_text(render_results_markdown(category_metrics))

    print(json.dumps(category_metrics["overall"], indent=2, sort_keys=True))
    print(f"wrote {metrics_path}")
    print(f"updated {results_path}")


def save_failure_example(
    *,
    output_root: Path,
    record: EvalFrameRecord,
    image,
    predicted_pieces: tuple[tuple[str, str | None], ...],
    strict_exact: bool,
    placed_exact: bool,
) -> str:
    category_dir = output_root / record.category
    category_dir.mkdir(parents=True, exist_ok=True)
    rendered = image.copy()
    corners = torch.tensor(record.corners, dtype=torch.float32).numpy().astype("int32")
    cv2.polylines(rendered, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    overlay_lines = [
        f"frame={record.frame_id} category={record.category}",
        f"strict_exact={int(strict_exact)} placed_exact={int(placed_exact)}",
        f"pred={piece_summary(predicted_pieces)}",
        f"gt={piece_summary(placed_piece_tuples(record.pieces))}",
    ]
    y = 28
    for line in overlay_lines:
        cv2.putText(
            rendered,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    output_path = category_dir / f"{record.frame_id}.png"
    cv2.imwrite(str(output_path), rendered)
    return str(output_path.relative_to(_PROJECT_ROOT))


def piece_summary(pieces: tuple[tuple[str, str | None], ...]) -> str:
    if not pieces:
        return "-"
    return ", ".join(f"{piece}@{square or 'no_square'}" for piece, square in pieces)


def render_results_markdown(metrics: dict[str, dict[str, object]]) -> str:
    rows = []
    ordered_categories = [
        "a-file-rook",
        "lateral-occlusion",
        "low-camera-angle",
        "dense-middlegame",
        "mid-move",
        "easy-stationary",
        "macro",
        "overall",
    ]
    for category in ordered_categories:
        if category not in metrics:
            continue
        row = metrics[category]
        count = row.get("count", "-")
        rows.append(
            "| "
            f"{category} | {count} | "
            f"{float(row['strict_piece_exact_match']):.4f} | "
            f"{float(row['placed_board_exact_match']):.4f} | "
            f"{float(row['per_square_accuracy']):.4f} | "
            f"{float(row['piece_f1_macro']):.4f} |"
        )
    lines = [
        "# Minimal DETR results",
        "",
        (
            "| category | count | strict piece exact | placed board exact | "
            "per-square acc | piece F1 macro |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
    ]
    overall_failures = metrics.get("overall", {}).get("failure_examples", [])
    if overall_failures:
        lines.append("## Failure examples")
        lines.append("")
        for path in overall_failures[:12]:
            lines.append(f"- `{path}`")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the minimal DETR study on study/eval.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eval-labels", type=Path, default=_PROJECT_ROOT / "study" / "eval" / "labels.jsonl"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--failures-per-category", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="latest")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    main()
