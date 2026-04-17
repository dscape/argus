#!/usr/bin/env python3
"""End-to-end eval of the two-stage per-square board reader on a physical split.

Loads the occupancy and piece classifier checkpoints, iterates the annotations,
runs the combined reader on each board's native frame, and reports:

- per-square accuracy
- non-empty accuracy (restricted to squares where GT is a piece)
- empty accuracy (restricted to squares where GT is empty)
- board_exact_match (fraction of boards where all 64 squares are correct)
- piece-only exact match (all non-empty squares correct; ignores empty squares)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.oblique_square_context import load_annotated_oblique_rows
from pipeline.physical.square_classifier_data import NativeFrameLoader
from pipeline.physical.square_classifiers import (
    SquareClassifier,
    SquareClassifierConfig,
)
from pipeline.physical.two_stage_board_reader import read_board
from pipeline.shared import SQUARE_CLASS_NAMES

from argus.model.vision_encoder import VisionEncoder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalMetrics:
    board_count: int
    square_count: int
    per_square_accuracy: float
    empty_accuracy: float
    non_empty_accuracy: float
    occupancy_accuracy: float
    piece_accuracy_on_correctly_occupied: float
    board_exact_match: float
    piece_only_exact_match: float
    mean_squares_correct_per_board: float
    per_class_recall: dict[str, float]
    confusion: dict[str, dict[str, int]]


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)

    occupancy_model = _load_classifier(args.occupancy_checkpoint, device=device)
    piece_model = _load_classifier(args.piece_checkpoint, device=device)

    rows = load_annotated_oblique_rows(args.annotation_root)
    rows = [r for r in rows if r.native_corners and r.native_image_bbox and r.source_video_id]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"evaluating on {len(rows)} annotations from {args.annotation_root}")

    frame_loader = NativeFrameLoader(capacity=args.frame_cache_capacity)

    board_count = 0
    boards_exact = 0
    piece_only_exact = 0
    all_correct: list[int] = []
    per_square_total = 0
    per_square_correct = 0
    empty_total = empty_correct = 0
    non_empty_total = non_empty_correct = 0
    occupancy_correct = 0  # predicted-empty vs actual-empty match
    piece_on_correct_occupied_total = piece_on_correct_occupied_correct = 0
    confusion: dict[str, Counter] = {name: Counter() for name in SQUARE_CLASS_NAMES}
    per_class_total: Counter = Counter()

    t_start = time.time()
    try:
        for row in rows:
            frame = frame_loader.load(
                source_video_id=str(row.source_video_id),
                source_frame_index=int(row.source_frame_index),
            )
            x_off, y_off, _, _ = row.native_image_bbox
            corners = tuple((float(c[0] + x_off), float(c[1] + y_off)) for c in row.native_corners)
            result = read_board(
                frame,
                corners,
                occupancy_model=occupancy_model,
                piece_model=piece_model,
                device=device,
            )
            predicted = result.class_ids
            gt = tuple(int(v) for v in row.labels)

            board_correct = sum(int(p == g) for p, g in zip(predicted, gt))
            all_correct.append(board_correct)
            per_square_total += 64
            per_square_correct += board_correct
            if board_correct == 64:
                boards_exact += 1
            piece_squares = [i for i, g in enumerate(gt) if g != 0]
            if piece_squares and all(predicted[i] == gt[i] for i in piece_squares):
                piece_only_exact += 1
            for p, g in zip(predicted, gt):
                g_name = SQUARE_CLASS_NAMES[g]
                p_name = SQUARE_CLASS_NAMES[p]
                confusion[g_name][p_name] += 1
                per_class_total[g_name] += 1
                if g == 0:
                    empty_total += 1
                    if p == 0:
                        empty_correct += 1
                        occupancy_correct += 1
                else:
                    non_empty_total += 1
                    if p == g:
                        non_empty_correct += 1
                    if p != 0:
                        occupancy_correct += 1
                        piece_on_correct_occupied_total += 1
                        if p == g:
                            piece_on_correct_occupied_correct += 1
            board_count += 1
            if board_count % 50 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  board {board_count}/{len(rows)} - "
                    f"exact={boards_exact / board_count:.3f} "
                    f"per-sq={per_square_correct / per_square_total:.3f} "
                    f"({elapsed:.0f}s elapsed)"
                )
    finally:
        frame_loader.close()

    per_class_recall = {
        name: (confusion[name][name] / per_class_total[name] if per_class_total[name] else 0.0)
        for name in SQUARE_CLASS_NAMES
    }

    metrics = EvalMetrics(
        board_count=board_count,
        square_count=per_square_total,
        per_square_accuracy=per_square_correct / max(per_square_total, 1),
        empty_accuracy=empty_correct / max(empty_total, 1),
        non_empty_accuracy=non_empty_correct / max(non_empty_total, 1),
        occupancy_accuracy=occupancy_correct / max(per_square_total, 1),
        piece_accuracy_on_correctly_occupied=(
            piece_on_correct_occupied_correct / max(piece_on_correct_occupied_total, 1)
        ),
        board_exact_match=boards_exact / max(board_count, 1),
        piece_only_exact_match=piece_only_exact / max(board_count, 1),
        mean_squares_correct_per_board=sum(all_correct) / max(board_count, 1),
        per_class_recall=per_class_recall,
        confusion={name: dict(counter) for name, counter in confusion.items()},
    )

    output_path = args.output or (
        _PROJECT_ROOT / "weights" / "physical" / "square_classifier" / "two_stage_eval.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2, sort_keys=True))
    print("\n===== EVAL METRICS =====")
    print(f"boards:                        {metrics.board_count}")
    print(f"per-square accuracy:           {metrics.per_square_accuracy:.4f}")
    print(f"empty accuracy:                {metrics.empty_accuracy:.4f}")
    print(f"non-empty accuracy:            {metrics.non_empty_accuracy:.4f}")
    print(f"occupancy accuracy:            {metrics.occupancy_accuracy:.4f}")
    print(f"piece accuracy on occupied:    {metrics.piece_accuracy_on_correctly_occupied:.4f}")
    print(f"board exact match:             {metrics.board_exact_match:.4f}")
    print(f"piece-only exact match:        {metrics.piece_only_exact_match:.4f}")
    print(f"mean squares correct / board:  {metrics.mean_squares_correct_per_board:.1f} / 64")
    print(f"\nwrote: {output_path.relative_to(_PROJECT_ROOT)}")


def _load_classifier(checkpoint_path: Path, *, device: torch.device) -> SquareClassifier:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_cfg = payload["encoder_config"]
    classifier_cfg = payload["classifier_config"]
    encoder = VisionEncoder(
        encoder_type=encoder_cfg.get("encoder_type", "dinov2"),
        model_name=encoder_cfg.get("model_name", "facebook/dinov2-base"),
        frozen=encoder_cfg.get("frozen", True),
    )
    model = SquareClassifier(
        vision_encoder=encoder,
        config=SquareClassifierConfig(
            num_classes=classifier_cfg["num_classes"],
            dropout=classifier_cfg.get("dropout", 0.1),
        ),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval two-stage board reader.")
    parser.add_argument(
        "--annotation-root", type=Path, default=_PROJECT_ROOT / "data" / "physical" / "val"
    )
    parser.add_argument(
        "--occupancy-checkpoint",
        type=Path,
        default=_PROJECT_ROOT
        / "weights"
        / "physical"
        / "square_classifier"
        / "occupancy"
        / "occupancy_classifier.pt",
    )
    parser.add_argument(
        "--piece-checkpoint",
        type=Path,
        default=_PROJECT_ROOT
        / "weights"
        / "physical"
        / "square_classifier"
        / "piece_corrected"
        / "piece_classifier.pt",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--frame-cache-capacity", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


if __name__ == "__main__":
    main()
