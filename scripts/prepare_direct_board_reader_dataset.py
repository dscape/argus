#!/usr/bin/env python3
"""Prepare a consolidated full-image board-state dataset manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.direct_board_reader_data import (
    DirectBoardImageLoader,
    DirectBoardRecord,
    chessred_image_path,
    chessred_labels_by_image_id,
    resolve_source_video_path,
    write_direct_board_records,
)

from argus.chess.board_state import fen_to_square_targets

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "direct_board_reader_dataset" / "fullphoto"


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chessred_payload = json.loads(args.chessred_annotations.read_text())
    chessred_labels = chessred_labels_by_image_id(chessred_payload)
    chessred_corner_lookup = build_chessred_corner_lookup(chessred_payload)

    train_rows: list[DirectBoardRecord] = []
    train_rows.extend(
        build_chessred_records(
            chessred_payload,
            chessred_labels_by_id=chessred_labels,
            corner_lookup=chessred_corner_lookup,
            images_root=args.chessred_images_dir,
            split_name="train",
            sample_weight=args.chessred_train_weight,
        )
    )
    train_rows.extend(
        build_physical_records(
            annotation_root=args.physical_train_root,
            split_name="train",
            sample_weight=args.physical_train_weight,
        )
    )
    train_rows.extend(
        build_synthetic_records_from_clips(
            clips_dir=args.synthetic_clips_dir,
            split_name="train",
            sample_weight=args.synthetic_train_weight,
            max_frames=args.synthetic_max_frames,
        )
    )

    physical_val_rows = build_physical_records(
        annotation_root=args.physical_val_root,
        split_name="physical_val",
        sample_weight=1.0,
    )
    chessred_val_rows = build_chessred_records(
        chessred_payload,
        chessred_labels_by_id=chessred_labels,
        corner_lookup=chessred_corner_lookup,
        images_root=args.chessred_images_dir,
        split_name="val",
        sample_weight=1.0,
    )

    train_manifest = output_dir / "train_manifest.jsonl"
    physical_val_manifest = output_dir / "physical_val_manifest.jsonl"
    chessred_val_manifest = output_dir / "chessred_val_manifest.jsonl"
    write_direct_board_records(train_manifest, train_rows)
    write_direct_board_records(physical_val_manifest, physical_val_rows)
    write_direct_board_records(chessred_val_manifest, chessred_val_rows)

    review_summary = write_review_bundle(
        output_dir=output_dir,
        train_rows=train_rows,
        physical_val_rows=physical_val_rows,
        chessred_val_rows=chessred_val_rows,
        samples_per_group=args.review_samples_per_group,
    )

    summary = {
        "output_dir": str(output_dir.relative_to(_PROJECT_ROOT)),
        "train_manifest": str(train_manifest.relative_to(_PROJECT_ROOT)),
        "physical_val_manifest": str(physical_val_manifest.relative_to(_PROJECT_ROOT)),
        "chessred_val_manifest": str(chessred_val_manifest.relative_to(_PROJECT_ROOT)),
        "review_dir": str((output_dir / "review").relative_to(_PROJECT_ROOT)),
        "data_roots": {
            "chessred_images_dir": relative_to_project(args.chessred_images_dir),
            "chessred_annotations": relative_to_project(args.chessred_annotations),
            "physical_train_root": relative_to_project(args.physical_train_root),
            "physical_val_root": relative_to_project(args.physical_val_root),
            "synthetic_clips_dir": relative_to_project(args.synthetic_clips_dir),
        },
        "train_counts_by_domain": dict(Counter(row.domain for row in train_rows)),
        "train_counts_by_split": dict(Counter(row.split for row in train_rows)),
        "physical_val_count": len(physical_val_rows),
        "chessred_val_count": len(chessred_val_rows),
        "physical_train_weight": args.physical_train_weight,
        "chessred_train_weight": args.chessred_train_weight,
        "synthetic_train_weight": args.synthetic_train_weight,
        "synthetic_max_frames": args.synthetic_max_frames,
        "review_summary": review_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a consolidated direct-board dataset")
    parser.add_argument(
        "--chessred-images-dir",
        type=Path,
        default=Path("data/chessred/images"),
    )
    parser.add_argument(
        "--chessred-annotations",
        type=Path,
        default=Path("data/chessred/annotations.json"),
    )
    parser.add_argument("--physical-train-root", type=Path, default=Path("data/physical/train"))
    parser.add_argument("--physical-val-root", type=Path, default=Path("data/physical/val"))
    parser.add_argument("--synthetic-clips-dir", type=Path, default=Path("data/argus/train"))
    parser.add_argument(
        "--synthetic-max-frames",
        type=int,
        default=0,
        help="0 uses all available Blender-rendered frames with FEN labels",
    )
    parser.add_argument("--physical-train-weight", type=float, default=8.0)
    parser.add_argument("--chessred-train-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-train-weight", type=float, default=1.0)
    parser.add_argument("--review-samples-per-group", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return _DEFAULT_OUTPUT_ROOT.resolve()


def build_chessred_corner_lookup(
    payload: dict[str, Any],
) -> dict[int, tuple[tuple[float, float], ...]]:
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        return {}
    corners = annotations.get("corners")
    if not isinstance(corners, list):
        return {}
    lookup: dict[int, tuple[tuple[float, float], ...]] = {}
    for record in corners:
        raw_corners = record.get("corners")
        if not isinstance(raw_corners, dict):
            continue
        try:
            lookup[int(record["image_id"])] = (
                tuple(float(value) for value in raw_corners["top_left"]),
                tuple(float(value) for value in raw_corners["top_right"]),
                tuple(float(value) for value in raw_corners["bottom_right"]),
                tuple(float(value) for value in raw_corners["bottom_left"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return lookup


def build_chessred_records(
    payload: dict[str, Any],
    *,
    chessred_labels_by_id: dict[int, tuple[int, ...]],
    corner_lookup: dict[int, tuple[tuple[float, float], ...]],
    images_root: Path,
    split_name: str,
    sample_weight: float,
) -> list[DirectBoardRecord]:
    images = payload.get("images")
    splits = payload.get("splits")
    if not isinstance(images, list) or not isinstance(splits, dict):
        raise ValueError("Unexpected ChessReD payload structure")
    split_payload = splits.get(split_name)
    if not isinstance(split_payload, dict):
        raise ValueError(f"ChessReD split {split_name!r} is missing")
    image_ids = set(int(value) for value in split_payload.get("image_ids", []))

    rows: list[DirectBoardRecord] = []
    for image_record in images:
        image_id = int(image_record["id"])
        if image_id not in image_ids:
            continue
        labels = chessred_labels_by_id[image_id]
        image_path = chessred_image_path(image_record, images_root=images_root)
        rows.append(
            DirectBoardRecord(
                example_id=f"chessred:{image_id}",
                domain="chessred",
                split=split_name,
                image_path=str(image_path.resolve().relative_to(_PROJECT_ROOT)),
                labels=labels,
                width=int(image_record["width"]),
                height=int(image_record["height"]),
                sample_weight=sample_weight,
                corners=corner_lookup.get(image_id),
            )
        )
    rows.sort(key=lambda row: row.example_id)
    return rows


def build_physical_records(
    *,
    annotation_root: Path,
    split_name: str,
    sample_weight: float,
) -> list[DirectBoardRecord]:
    rows = load_fully_labeled_physical_annotations(annotation_root)
    exported: list[DirectBoardRecord] = []
    for row in rows:
        x, y, w, h = [int(value) for value in row["native_image_bbox"]]
        exported.append(
            DirectBoardRecord(
                example_id=f"physical:{row['annotation_id']}",
                domain="physical",
                split=split_name,
                labels=tuple(int(value) for value in row["labels"]),
                width=w,
                height=h,
                sample_weight=sample_weight,
                annotation_id=str(row["annotation_id"]),
                clip_path=str(row["clip_path"]),
                frame_index=int(row["frame_index"]),
                source_video_id=str(row["source_video_id"]),
                source_frame_index=int(row["source_frame_index"]),
                native_image_bbox=(x, y, w, h),
                corners=tuple((float(px), float(py)) for px, py in row["native_corners"]),
            )
        )
    exported.sort(key=lambda record: record.example_id)
    return exported


def build_synthetic_records_from_clips(
    *,
    clips_dir: Path,
    split_name: str,
    sample_weight: float,
    max_frames: int,
) -> list[DirectBoardRecord]:
    clip_paths = sorted(clips_dir.glob("*.pt"))
    rows: list[DirectBoardRecord] = []
    for clip_path in clip_paths:
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip, dict):
            continue
        frames = clip.get("frames")
        fens = clip.get("fens")
        if not isinstance(frames, torch.Tensor) or not isinstance(fens, list) or not fens:
            continue
        board_flipped = bool(clip.get("board_flipped", False))
        frame_count = min(int(frames.shape[0]), len(fens))
        for frame_index in range(frame_count):
            fen = fens[frame_index]
            if not isinstance(fen, str) or not fen:
                continue
            frame = frames[frame_index]
            if frame.ndim != 3:
                continue
            if frame.shape[0] == 3:
                height = int(frame.shape[1])
                width = int(frame.shape[2])
            elif frame.shape[-1] == 3:
                height = int(frame.shape[0])
                width = int(frame.shape[1])
            else:
                continue
            labels = fen_to_square_targets(fen, board_flipped=board_flipped)
            rows.append(
                DirectBoardRecord(
                    example_id=f"synthetic:{clip_path.stem}:{frame_index:04d}",
                    domain="synthetic",
                    split=split_name,
                    labels=tuple(int(value) for value in labels.tolist()),
                    width=width,
                    height=height,
                    sample_weight=sample_weight,
                    clip_path=str(clip_path.resolve().relative_to(_PROJECT_ROOT.resolve())),
                    frame_index=frame_index,
                )
            )
            if max_frames > 0 and len(rows) >= max_frames:
                return rows
    return rows


def write_review_bundle(
    *,
    output_dir: Path,
    train_rows: list[DirectBoardRecord],
    physical_val_rows: list[DirectBoardRecord],
    chessred_val_rows: list[DirectBoardRecord],
    samples_per_group: int,
) -> dict[str, Any]:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[DirectBoardRecord]] = {
        "train_chessred": [row for row in train_rows if row.domain == "chessred"],
        "train_physical": [row for row in train_rows if row.domain == "physical"],
        "train_synthetic": [row for row in train_rows if row.domain == "synthetic"],
        "val_physical": physical_val_rows,
        "val_chessred": chessred_val_rows,
    }

    loader = DirectBoardImageLoader()
    manifest: dict[str, Any] = {"groups": {}}
    try:
        for group_name, rows in groups.items():
            sampled_rows = evenly_spaced_sample(rows, samples_per_group)
            if not sampled_rows:
                continue
            sheet_path = review_dir / f"{group_name}.jpg"
            create_contact_sheet(sampled_rows, loader=loader, output_path=sheet_path)
            manifest["groups"][group_name] = {
                "count": len(rows),
                "sample_count": len(sampled_rows),
                "contact_sheet": str(sheet_path.relative_to(_PROJECT_ROOT)),
                "examples": [review_manifest_entry(row) for row in sampled_rows],
            }
    finally:
        loader.close()

    manifest_path = review_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    summary_lines = [
        "# Direct board reader data review",
        "",
        f"- review manifest: `{manifest_path.relative_to(_PROJECT_ROOT)}`",
    ]
    for group_name, payload in manifest["groups"].items():
        summary_lines.append(
            f"- {group_name}: `{payload['count']}` rows, sample sheet `{payload['contact_sheet']}`"
        )
    (review_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")
    return {
        "manifest": str(manifest_path.relative_to(_PROJECT_ROOT)),
        "groups": {
            key: {
                "count": value["count"],
                "sample_count": value["sample_count"],
                "contact_sheet": value["contact_sheet"],
            }
            for key, value in manifest["groups"].items()
        },
    }


def evenly_spaced_sample(rows: list[DirectBoardRecord], count: int) -> list[DirectBoardRecord]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows
    indices = sorted(
        {
            min(len(rows) - 1, round(index * (len(rows) - 1) / max(count - 1, 1)))
            for index in range(count)
        }
    )
    return [rows[index] for index in indices]


def create_contact_sheet(
    rows: list[DirectBoardRecord],
    *,
    loader: DirectBoardImageLoader,
    output_path: Path,
    columns: int = 4,
    tile_size: int = 256,
    caption_height: int = 44,
) -> None:
    columns = max(columns, 1)
    cell_height = tile_size + caption_height
    sheet_width = columns * tile_size
    sheet_rows = math.ceil(len(rows) / columns)
    sheet_height = sheet_rows * cell_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        image_bgr = loader.load_bgr(row)
        tile = render_review_tile(
            image_bgr=image_bgr,
            caption_lines=[row.domain, short_source_label(row)],
            tile_size=tile_size,
            caption_height=caption_height,
        )
        x_offset = (index % columns) * tile_size
        y_offset = (index // columns) * cell_height
        sheet.paste(tile, (x_offset, y_offset))
        draw.rectangle(
            [x_offset, y_offset, x_offset + tile_size - 1, y_offset + cell_height - 1],
            outline=(180, 180, 180),
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def render_review_tile(
    *,
    image_bgr,
    caption_lines: list[str],
    tile_size: int,
    caption_height: int,
) -> Image.Image:
    rgb = Image.fromarray(image_bgr[:, :, ::-1])
    image_area_height = tile_size
    image_area = Image.new("RGB", (tile_size, image_area_height), color=(127, 127, 127))
    scale = min(tile_size / max(rgb.width, 1), image_area_height / max(rgb.height, 1))
    resized_width = max(1, int(round(rgb.width * scale)))
    resized_height = max(1, int(round(rgb.height * scale)))
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    x_offset = (tile_size - resized_width) // 2
    y_offset = (image_area_height - resized_height) // 2
    image_area.paste(resized, (x_offset, y_offset))

    tile = Image.new("RGB", (tile_size, image_area_height + caption_height), color=(255, 255, 255))
    tile.paste(image_area, (0, 0))
    draw = ImageDraw.Draw(tile)
    for line_index, line in enumerate(caption_lines[:2]):
        draw.text((6, image_area_height + 4 + line_index * 16), line[:40], fill=(0, 0, 0))
    return tile


def short_source_label(row: DirectBoardRecord) -> str:
    if row.image_path is not None:
        return Path(row.image_path).name
    if row.source_video_id is not None and row.source_frame_index is not None:
        return f"{row.source_video_id}:f{row.source_frame_index}"
    if row.clip_path is not None and row.frame_index is not None:
        return f"{Path(row.clip_path).name}:f{row.frame_index}"
    return row.example_id


def review_manifest_entry(row: DirectBoardRecord) -> dict[str, Any]:
    entry = {
        "example_id": row.example_id,
        "domain": row.domain,
        "split": row.split,
        "width": row.width,
        "height": row.height,
        "image_path": row.image_path,
        "clip_path": row.clip_path,
        "frame_index": row.frame_index,
        "source_video_id": row.source_video_id,
        "source_frame_index": row.source_frame_index,
        "native_image_bbox": row.native_image_bbox,
        "annotation_id": row.annotation_id,
    }
    return entry


def load_fully_labeled_physical_annotations(annotation_root: Path) -> list[dict[str, Any]]:
    annotations_path = annotation_root / "board_annotations.jsonl"
    rows: list[dict[str, Any]] = []
    for line in annotations_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        labels = payload.get("labels")
        source_video_id = payload.get("source_video_id")
        if (
            not isinstance(labels, list)
            or len(labels) != 64
            or any(value is None for value in labels)
            or not isinstance(payload.get("native_corners"), list)
            or not isinstance(payload.get("native_image_bbox"), list)
            or payload.get("source_frame_index") is None
            or not isinstance(source_video_id, str)
        ):
            continue
        resolve_source_video_path(source_video_id)
        rows.append(payload)
    return rows


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
