#!/usr/bin/env python3
"""Migrate physical board annotations from 224x224 frame space to native resolution.

Existing annotations have corners positioned on 224x224 downscaled clip frames.
This script re-rectifies them using the native-resolution source video frames:

1. Loads each annotation's clip to determine the stored frame dimensions.
2. Opens the source video and crops the camera region at native resolution.
3. Scales corner coordinates from stored-frame space to native-frame space.
4. Re-rectifies the board and re-extracts square crops at 512x512.
5. Updates the JSONL manifest with new corner coordinates.

Usage:
    python scripts/migrate_annotations_to_native_res.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.physical.annotation_dataset import (  # noqa: E402
    DEFAULT_BOARD_SIZE,
    extract_square_crops,
    rectify_board_image,
)

logger = logging.getLogger(__name__)

_REAL_CLIP_RE = re.compile(r"^clip_overlay_(?P<video_id>.+?)_clip(?P<clip_id>\d+)_\d+\.pt$")

ANNOTATION_DIRS = [
    _PROJECT_ROOT / "data" / "physical" / "train",
    _PROJECT_ROOT / "data" / "physical" / "val",
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_rgb_jpeg(path: Path, image_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _get_clip_db_info(clip_id: int) -> dict | None:
    try:
        from pipeline.db.connection import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT video_id, camera_bbox, ref_resolution
                       FROM video_clips WHERE id = %s""",
                    (clip_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "video_id": row[0],
                    "camera_bbox": tuple(int(v) for v in row[1]),
                    "ref_resolution": tuple(int(v) for v in row[2]),
                }
    except Exception as e:
        logger.warning("DB lookup failed for clip %d: %s", clip_id, e)
        return None


def _get_video_path(video_id: str) -> str | None:
    try:
        from pipeline.download.video_downloader import get_video_path

        return get_video_path(video_id)
    except Exception:
        return None


def _get_native_frame(
    video_path: str,
    camera_bbox: tuple[int, int, int, int],
    ref_resolution: tuple[int, int],
    source_frame_index: int,
) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_w, ref_h = ref_resolution
        sx = width / ref_w if ref_w > 0 else 1.0
        sy = height / ref_h if ref_h > 0 else 1.0
        x, y, w, h = camera_bbox
        bx = max(0, min(int(round(x * sx)), width - 1))
        by = max(0, min(int(round(y * sy)), height - 1))
        bw = max(1, min(int(round(w * sx)), width - bx))
        bh = max(1, min(int(round(h * sy)), height - by))

        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        crop = frame[by : by + bh, bx : bx + bw]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def migrate_annotations(dry_run: bool = False) -> None:
    clip_cache: dict[str, dict] = {}
    total = 0
    migrated = 0
    skipped = 0

    for ann_dir in ANNOTATION_DIRS:
        board_ann_path = ann_dir / "board_annotations.jsonl"
        square_manifest_path = ann_dir / "square_manifest.jsonl"
        boards_dir = ann_dir / "boards"
        squares_dir = ann_dir / "squares"

        if not board_ann_path.exists():
            continue

        board_records = _load_jsonl(board_ann_path)
        square_records = _load_jsonl(square_manifest_path)
        updated_board_records = []
        updated_square_records = [
            r
            for r in square_records
            if r.get("annotation_id") not in {b.get("annotation_id") for b in board_records}
        ]

        for record in board_records:
            total += 1
            clip_path = record.get("clip_path", "")
            frame_index = int(record.get("frame_index", 0))
            corners = record.get("corners", [])
            labels = record.get("labels", [])
            annotation_id = record.get("annotation_id", "")

            # Check if corners are in 224x224 space
            max_corner = max(max(c[0], c[1]) for c in corners) if corners else 0
            if max_corner > 224:
                # Already in native resolution space
                updated_board_records.append(record)
                skipped += 1
                continue

            # Parse clip reference
            clip_filename = Path(clip_path).name
            match = _REAL_CLIP_RE.match(clip_filename)
            if not match:
                logger.warning("Cannot parse clip filename: %s", clip_filename)
                updated_board_records.append(record)
                skipped += 1
                continue

            video_id = match.group("video_id")
            clip_id = int(match.group("clip_id"))

            # Get DB info
            db_info = _get_clip_db_info(clip_id)
            if db_info is None or db_info["video_id"] != video_id:
                logger.warning("No DB info for clip %s (id=%d)", clip_filename, clip_id)
                updated_board_records.append(record)
                skipped += 1
                continue

            camera_bbox = db_info["camera_bbox"]
            if camera_bbox == (0, 0, 100, 100):
                logger.warning("Placeholder bbox for clip %s — skipping", clip_filename)
                updated_board_records.append(record)
                skipped += 1
                continue

            # Get stored frame dimensions
            stored_clip_path = _PROJECT_ROOT / clip_path
            if clip_path not in clip_cache:
                if stored_clip_path.exists():
                    clip_data = torch.load(stored_clip_path, map_location="cpu", weights_only=False)
                    clip_cache[clip_path] = clip_data
                else:
                    clip_cache[clip_path] = {}
            clip_data = clip_cache[clip_path]
            frames = clip_data.get("frames")
            if not isinstance(frames, torch.Tensor):
                logger.warning("No frames tensor in %s", clip_path)
                updated_board_records.append(record)
                skipped += 1
                continue

            stored_h, stored_w = frames.shape[-2], frames.shape[-1]

            # Get source frame index
            frame_indices = clip_data.get("frame_indices")
            if isinstance(frame_indices, torch.Tensor) and frame_index < frame_indices.shape[0]:
                source_frame_idx = int(frame_indices[frame_index].item())
            else:
                source_frame_idx = frame_index

            # Get native-resolution frame
            video_path = _get_video_path(video_id)
            if video_path is None:
                logger.warning("Source video not found for %s", video_id)
                updated_board_records.append(record)
                skipped += 1
                continue

            native_frame = _get_native_frame(
                video_path, camera_bbox, db_info["ref_resolution"], source_frame_idx
            )
            if native_frame is None:
                logger.warning(
                    "Failed to read native frame for %s frame %d", clip_filename, frame_index
                )
                updated_board_records.append(record)
                skipped += 1
                continue

            native_h, native_w = native_frame.shape[:2]

            # Scale corners from stored-frame space to native-frame space
            scale_x = native_w / stored_w
            scale_y = native_h / stored_h
            new_corners = [[c[0] * scale_x, c[1] * scale_y] for c in corners]

            if dry_run:
                logger.info(
                    "[DRY RUN] %s: corners scaled %dx%d → %dx%d (scale %.2fx%.2f)",
                    annotation_id,
                    stored_w,
                    stored_h,
                    native_w,
                    native_h,
                    scale_x,
                    scale_y,
                )
                record["corners"] = new_corners
                updated_board_records.append(record)
                migrated += 1
                continue

            # Re-rectify from native-resolution frame
            output_size = int(record.get("rectified_size", DEFAULT_BOARD_SIZE))
            rectified = rectify_board_image(native_frame, new_corners, output_size=output_size)

            # Save updated rectified board
            board_path = boards_dir / f"{annotation_id}.jpg"
            _write_rgb_jpeg(board_path, rectified)

            # Re-extract square crops
            square_crops = extract_square_crops(rectified)
            for sq_idx, label in enumerate(labels):
                if label is None:
                    continue
                square_name = f"{chr(97 + sq_idx % 8)}{8 - sq_idx // 8}"
                crop_path = squares_dir / f"{annotation_id}_{square_name}.jpg"
                _write_rgb_jpeg(crop_path, square_crops[sq_idx])

                from pipeline.shared import SQUARE_CLASS_NAMES

                updated_square_records.append(
                    {
                        "annotation_id": annotation_id,
                        "clip_path": clip_path,
                        "frame_index": frame_index,
                        "source_video_id": record.get("source_video_id"),
                        "square_index": sq_idx,
                        "square_name": square_name,
                        "label_index": label,
                        "label_name": SQUARE_CLASS_NAMES[label],
                        "crop_path": str(crop_path.relative_to(_PROJECT_ROOT)),
                        "split": ann_dir.name,
                    }
                )

            # Update record with new corners
            record["corners"] = new_corners
            updated_board_records.append(record)
            migrated += 1

        if not dry_run:
            _write_jsonl(board_ann_path, updated_board_records)
            _write_jsonl(square_manifest_path, updated_square_records)

        logger.info(
            "%s: %d annotations, %d migrated, %d skipped",
            ann_dir.name,
            len(board_records),
            migrated,
            skipped,
        )

    logger.info("Total: %d annotations, %d migrated, %d skipped", total, migrated, skipped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log changes without writing")
    args = parser.parse_args()
    migrate_annotations(dry_run=args.dry_run)
