#!/usr/bin/env python3
"""Evaluate the committed physical runtime reader on held-out rectified boards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.board_data import PhysicalEvalBoardDataset
from pipeline.physical.square_classifier import (
    PhysicalBoardSequenceReader,
    read_board_observation_from_frame,
)
from pipeline.physical.square_probe import evaluate_probe
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "physical_runtime_eval.json"
_CLASS_NAME_TO_INDEX = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate physical runtime reader")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--temporal-mode",
        type=str,
        choices=["off", "fixed", "metadata"],
        default="off",
    )
    parser.add_argument("--temporal-ema-alpha", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    dataset = PhysicalEvalBoardDataset()
    rows = sorted(
        dataset.rows,
        key=lambda row: (
            row.clip_path or row.annotation_id,
            -1 if row.frame_index is None else row.frame_index,
            row.annotation_id,
        ),
    )
    sequence_reader = None
    current_clip_key = None
    if args.temporal_mode == "fixed":
        if args.temporal_ema_alpha <= 0.0:
            raise ValueError("--temporal-ema-alpha must be > 0 when --temporal-mode=fixed")
        sequence_reader = PhysicalBoardSequenceReader(
            device=args.device,
            ema_alpha=args.temporal_ema_alpha,
        )
    elif args.temporal_mode == "metadata":
        sequence_reader = PhysicalBoardSequenceReader(device=args.device)

    board_annotation_ids: list[str] = []
    predicted_labels: list[int] = []
    target_labels: list[int] = []
    missing_predictions = 0

    for row in rows:
        image = cv2.imread(str(_PROJECT_ROOT / row.board_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load board image: {row.board_path}")
        if sequence_reader is not None:
            clip_key = row.clip_path or row.annotation_id
            if clip_key != current_clip_key:
                sequence_reader.reset()
                current_clip_key = clip_key
            observation = sequence_reader.read_board_observation_from_frame(image)
        else:
            observation = read_board_observation_from_frame(image, device=args.device)
        if observation is None:
            missing_predictions += 1
            continue
        predicted_labels.extend(fen_to_class_ids(observation.fen))
        target_labels.extend(row.labels)
        board_annotation_ids.extend([row.annotation_id] * 64)

    import torch
    import torch.nn as nn

    class IdentityProbe(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    num_classes = len(SQUARE_CLASS_NAMES)
    logits = torch.zeros((len(predicted_labels), num_classes), dtype=torch.float32)
    for index, class_id in enumerate(predicted_labels):
        logits[index, class_id] = 1.0
    metrics = evaluate_probe(
        IdentityProbe(),
        logits,
        torch.tensor(target_labels, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )

    report = {
        "missing_predictions": missing_predictions,
        "evaluated_boards": len(predicted_labels) // 64,
        "temporal_mode": args.temporal_mode,
        "temporal_ema_alpha": args.temporal_ema_alpha,
        "metrics": metrics.to_dict(),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))


def fen_to_class_ids(fen: str) -> list[int]:
    placement = fen.split(" ", 1)[0]
    class_ids: list[int] = []
    for rank in placement.split("/"):
        for char in rank:
            if char.isdigit():
                class_ids.extend([0] * int(char))
            else:
                class_ids.append(_CLASS_NAME_TO_INDEX[char])
    if len(class_ids) != 64:
        raise ValueError(f"Expected 64 squares from FEN, got {len(class_ids)}: {fen}")
    return class_ids


if __name__ == "__main__":
    main()
