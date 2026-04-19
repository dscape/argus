"""Dump argus-projected crops to chesscog ImageFolder layout (Study 1).

Mirrors `dataset_export.py` but uses argus's `extract_projected_*_crop` instead
of chesscog warp + crop. Output is consumed by chesscog's training loop to
train ResNet18 / InceptionV3 on argus's 3D-box geometry.

Layout:
    <output_root>/occupancy/{train|val}/{empty|occupied}/{id}_{square}.png
    <output_root>/pieces/{train|val}/{chesscog_folder}/{id}_{square}.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image

from pipeline.physical.chesscog_baseline import ARGUS_CLASS_TO_CHESSCOG_FOLDER
from pipeline.physical.chesscog_baseline.dataset_export import (
    _resolve_full_frame_and_corners,
    _square_from_argus_index,
)
from pipeline.physical.piece_projection import (
    extract_projected_occupancy_crop,
    extract_projected_piece_crop,
)
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader


def export_argus_geom_split(
    annotation_root: Path,
    output_root: Path,
    split_name: str,
    *,
    occupancy_size: int = 100,
    piece_size: int = 200,
    pad_ratio: float = 0.3,
    limit: int | None = None,
) -> dict[str, int]:
    rows = [
        r
        for r in load_annotated_oblique_rows(annotation_root)
        if r.native_corners
        and r.native_image_bbox
        and r.source_video_id
        and r.source_frame_index is not None
    ]
    if limit is not None:
        rows = rows[:limit]
    print(f"[{split_name}] argus-geom export on {len(rows)} boards → {output_root}")

    occ_root = output_root / "occupancy" / split_name
    piece_root = output_root / "pieces" / split_name
    for class_name in ("empty", "occupied"):
        (occ_root / class_name).mkdir(parents=True, exist_ok=True)
    for folder in ARGUS_CLASS_TO_CHESSCOG_FOLDER.values():
        (piece_root / folder).mkdir(parents=True, exist_ok=True)

    native_loader = NativeFrameLoader()
    stats = {"boards": 0, "occupancy_crops": 0, "piece_crops": 0, "skipped_rows": 0}
    t0 = time.time()
    try:
        for row_idx, row in enumerate(rows):
            try:
                frame_bgr, corners = _resolve_full_frame_and_corners(
                    row, native_loader=native_loader, clip_cache={}
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {row.annotation_id}: {exc}")
                stats["skipped_rows"] += 1
                continue

            for i, class_id in enumerate(row.labels):
                square = _square_from_argus_index(i)
                square_name = chess.square_name(square)
                board_row, col = i // 8, i % 8

                occ_bgr = extract_projected_occupancy_crop(
                    frame_bgr,
                    corners,
                    row=board_row,
                    col=col,
                    output_size=occupancy_size,
                    pad_ratio=pad_ratio,
                )
                occ_rgb = cv2.cvtColor(occ_bgr, cv2.COLOR_BGR2RGB)
                occ_folder = "occupied" if class_id != 0 else "empty"
                Image.fromarray(occ_rgb, "RGB").save(
                    occ_root / occ_folder / f"{row.annotation_id}_{square_name}.png"
                )
                stats["occupancy_crops"] += 1

                if class_id != 0:
                    piece_bgr = extract_projected_piece_crop(
                        frame_bgr,
                        corners,
                        row=board_row,
                        col=col,
                        output_size=piece_size,
                        flip_left_half=True,
                    )
                    piece_rgb = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2RGB)
                    folder = ARGUS_CLASS_TO_CHESSCOG_FOLDER[int(class_id)]
                    Image.fromarray(piece_rgb, "RGB").save(
                        piece_root / folder / f"{row.annotation_id}_{square_name}.png"
                    )
                    stats["piece_crops"] += 1

            stats["boards"] += 1
            if (row_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{split_name}] {row_idx + 1}/{len(rows)} boards, "
                    f"{stats['occupancy_crops']} occ / {stats['piece_crops']} piece, "
                    f"elapsed {elapsed:.0f}s"
                )
    finally:
        native_loader.close()

    print(
        f"[{split_name}] done — {stats['boards']} boards, "
        f"{stats['occupancy_crops']} occupancy crops, "
        f"{stats['piece_crops']} piece crops, "
        f"{stats['skipped_rows']} skipped"
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/physical"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/study1_argus_geom"),
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--occupancy-size", type=int, default=100)
    ap.add_argument("--piece-size", type=int, default=200)
    ap.add_argument("--pad-ratio", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for split in args.splits:
        export_argus_geom_split(
            annotation_root=args.data_root / split,
            output_root=args.output_root,
            split_name=split,
            occupancy_size=args.occupancy_size,
            piece_size=args.piece_size,
            pad_ratio=args.pad_ratio,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
