#!/usr/bin/env python3
"""Export the actual crop images used by a two-stage training run.

This is for inspection. It saves the deterministic dataset images that the
training code would consume when augmentation is off.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    OccupancySquareDataset,
    PieceSquareDataset,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = load_annotated_oblique_rows(args.physical_train_root)
    val_rows = load_annotated_oblique_rows(args.physical_val_root)

    export_summary = {
        "train": export_split(
            split_name="train",
            rows=train_rows,
            output_root=output_root,
            include_occupancy=not args.piece_only,
            include_piece=not args.occupancy_only,
        ),
        "val": export_split(
            split_name="val",
            rows=val_rows,
            output_root=output_root,
            include_occupancy=not args.piece_only,
            include_piece=not args.occupancy_only,
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(export_summary, indent=2, sort_keys=True))
    print(json.dumps(export_summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export crop images used by the two-stage training datasets."
    )
    parser.add_argument(
        "--physical-train-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "train",
    )
    parser.add_argument(
        "--physical-val-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "val",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--occupancy-only", action="store_true", default=False)
    parser.add_argument("--piece-only", action="store_true", default=False)
    return parser


def export_split(
    *,
    split_name: str,
    rows,
    output_root: Path,
    include_occupancy: bool,
    include_piece: bool,
) -> dict[str, object]:
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "board_count": len(rows),
    }

    if include_occupancy:
        occupancy_dataset = OccupancySquareDataset(
            rows=rows,
            input_size=DEFAULT_OCCUPANCY_CROP_SIZE,
        )
        try:
            summary["occupancy"] = export_dataset(
                dataset=occupancy_dataset,
                split_root=split_root,
                task_name="occupancy",
                class_names=("empty", "occupied"),
            )
        finally:
            occupancy_dataset.close()

    if include_piece:
        piece_dataset = PieceSquareDataset(rows=rows, input_size=DEFAULT_PIECE_CROP_SIZE)
        try:
            summary["piece"] = export_dataset(
                dataset=piece_dataset,
                split_root=split_root,
                task_name="piece",
                class_names=SQUARE_CLASS_NAMES[1:],
            )
        finally:
            piece_dataset.close()

    return summary


def export_dataset(
    *,
    dataset,
    split_root: Path,
    task_name: str,
    class_names,
) -> dict[str, object]:
    task_root = split_root / task_name
    images_root = task_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    manifest_path = task_root / "manifest.jsonl"

    counts = Counter()
    manifest_lines: list[str] = []

    for index, sample in enumerate(dataset.indices, start=1):
        row = dataset.rows[sample.row_index]
        image_tensor, label_tensor = dataset[index - 1]
        image_rgb = tensor_to_rgb_uint8(image_tensor)
        square_name = square_index_to_name(sample.square_index)
        label_index = int(label_tensor.item())
        label_name = str(class_names[label_index])
        counts[label_name] += 1

        board_dir = images_root / row.annotation_id
        board_dir.mkdir(parents=True, exist_ok=True)
        image_path = board_dir / f"{square_name}_{label_name}.jpg"
        ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError(f"Failed to write image {image_path}")

        manifest_lines.append(
            json.dumps(
                {
                    "annotation_id": row.annotation_id,
                    "task": task_name,
                    "square_index": sample.square_index,
                    "square_name": square_name,
                    "label_index": label_index,
                    "label_name": label_name,
                    "source_video_id": row.source_video_id,
                    "image_path": str(image_path.relative_to(_PROJECT_ROOT)),
                },
                sort_keys=True,
            )
        )

        if index % 1000 == 0:
            print(f"[{task_name}] exported {index}/{len(dataset.indices)}")

    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    return {
        "sample_count": len(dataset.indices),
        "class_counts": dict(sorted(counts.items())),
        "manifest_path": str(manifest_path.relative_to(_PROJECT_ROOT)),
        "images_root": str(images_root.relative_to(_PROJECT_ROOT)),
    }


def tensor_to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    rgb = (image_tensor.detach().cpu() * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0.0, 1.0)
    return (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def square_index_to_name(square_index: int) -> str:
    row, col = divmod(square_index, 8)
    return f"{chr(ord('a') + col)}{8 - row}"


if __name__ == "__main__":
    main()
