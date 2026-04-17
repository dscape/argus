#!/usr/bin/env python3
"""Generate contact sheets and class stats for the two-stage per-square dataset.

Primary purpose: verify the chesscog-style crops visually before training. For
a sample of annotated frames, render the 64 occupancy and piece crops side by
side with their ground-truth labels so a reviewer can confirm piece tops are
now inside their tiles.

The datasets themselves (``OccupancySquareDataset`` / ``PieceSquareDataset``)
read directly from ``board_annotations.jsonl``, so no manifest is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.oblique_square_context import (
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.square_crop import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    extract_all_occupancy_crops,
    extract_all_piece_crops,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "square_classifier_review"


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir or _DEFAULT_OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    split_outputs: dict[str, dict[str, object]] = {}
    for split_name, annotation_root in (
        ("train", args.physical_train_root),
        ("val", args.physical_val_root),
    ):
        rows = load_annotated_oblique_rows(annotation_root)
        class_counts = _count_classes(rows)
        sampled_rows = _evenly_spaced_sample(rows, args.contact_sheet_count)
        contact_paths: list[str] = []
        for row in sampled_rows:
            try:
                frame_bgr = _load_clip_frame_bgr(row, clip_cache={})
            except (FileNotFoundError, ValueError) as error:
                print(f"[{split_name}] skip {row.annotation_id}: {error}")
                continue
            occupancy_crops = extract_all_occupancy_crops(
                frame_bgr, row.corners, output_size=DEFAULT_OCCUPANCY_CROP_SIZE
            )
            piece_crops = extract_all_piece_crops(
                frame_bgr, row.corners, output_size=DEFAULT_PIECE_CROP_SIZE
            )
            sheet_path = output_dir / f"{split_name}_{row.annotation_id}.jpg"
            _render_contact_sheet(
                sheet_path,
                occupancy_crops=occupancy_crops,
                piece_crops=piece_crops,
                labels=row.labels,
                annotation_id=row.annotation_id,
            )
            contact_paths.append(str(sheet_path.relative_to(_PROJECT_ROOT)))

        split_outputs[split_name] = {
            "annotation_count": len(rows),
            "class_counts": {name: int(class_counts.get(name, 0)) for name in SQUARE_CLASS_NAMES},
            "contact_sheets": contact_paths,
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(split_outputs, indent=2, sort_keys=True))
    print(json.dumps(split_outputs, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Contact-sheet review for the two-stage square classifier."
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
    parser.add_argument(
        "--contact-sheet-count",
        type=int,
        default=6,
        help="Contact sheets per split (evenly spaced through the annotation list).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _count_classes(rows) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        for label in row.labels:
            counter[SQUARE_CLASS_NAMES[int(label)]] += 1
    return counter


def _evenly_spaced_sample(rows, count: int):
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    indices = sorted(
        {
            min(len(rows) - 1, round(index * (len(rows) - 1) / max(count - 1, 1)))
            for index in range(count)
        }
    )
    return [rows[index] for index in indices]


def _render_contact_sheet(
    sheet_path: Path,
    *,
    occupancy_crops: list[np.ndarray],
    piece_crops: list[np.ndarray],
    labels: tuple[int, ...],
    annotation_id: str,
) -> None:
    tile_size = 96
    caption_height = 20
    cols = 8
    header_height = 40
    column_gap = 24
    column_width = cols * tile_size
    sheet_width = column_width * 2 + column_gap
    sheet_height = header_height + 8 * (tile_size + caption_height)

    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(14)
    label_font = _load_font(11)
    draw.text(
        (12, 10), f"{annotation_id} - occupancy (symmetric 0.5SS pad)", font=font, fill=(40, 40, 40)
    )
    draw.text(
        (column_width + column_gap + 12, 10),
        f"{annotation_id} - piece (asymmetric + flip)",
        font=font,
        fill=(40, 40, 40),
    )

    for square_index in range(64):
        row = square_index // 8
        col = square_index % 8
        label_name = SQUARE_CLASS_NAMES[int(labels[square_index])]
        square_name = _square_name(square_index)
        caption = f"{square_name}:{label_name}"

        occ_tile = _resize_to_tile(occupancy_crops[square_index], tile_size)
        piece_tile = _resize_to_tile(piece_crops[square_index], tile_size)

        x_left = col * tile_size
        y = header_height + row * (tile_size + caption_height)
        sheet.paste(occ_tile, (x_left, y))
        draw.text((x_left + 4, y + tile_size + 2), caption, font=label_font, fill=(30, 30, 30))

        x_right = column_width + column_gap + col * tile_size
        sheet.paste(piece_tile, (x_right, y))
        draw.text((x_right + 4, y + tile_size + 2), caption, font=label_font, fill=(30, 30, 30))

    sheet.save(sheet_path, quality=92)


def _resize_to_tile(image_bgr: np.ndarray, tile_size: int) -> Image.Image:
    from cv2 import COLOR_BGR2RGB, INTER_AREA, INTER_LINEAR, cvtColor, resize

    rgb = cvtColor(image_bgr, COLOR_BGR2RGB)
    interpolation = INTER_AREA if min(rgb.shape[:2]) >= tile_size else INTER_LINEAR
    resized = resize(rgb, (tile_size, tile_size), interpolation=interpolation)
    return Image.fromarray(resized)


def _square_name(square_index: int) -> str:
    row, col = divmod(square_index, 8)
    return f"{chr(ord('a') + col)}{8 - row}"


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("Inter-Regular.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
