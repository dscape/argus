#!/usr/bin/env python3
"""Backfill native-resolution metadata for physical board annotations.

Existing annotations store `corners` in clip-frame space because current oblique and
move-model training still read frames from `clip_path`.  This script keeps that
canonical clip-space geometry intact while adding native-resolution metadata and
rebuilding rectified board assets from the source video camera crop.

For each annotated frame the script:
1. Loads the stored clip to recover clip-frame size and source-frame index.
2. Opens the source video camera crop at native resolution.
3. Scales clip-space corners into that native crop.
4. Stores:
   - `corner_space = "clip_frame"`
   - `clip_frame_size`
   - `native_corners`
   - `native_image_bbox`
   - `source_frame_index`
5. Re-rectifies the board and regenerates square crops from the native crop.

Usage:
    python scripts/migrate_annotations_to_native_res.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.physical.shared.annotation_dataset import (  # noqa: E402
    DEFAULT_BOARD_SIZE,
    extract_square_crops,
    rectify_board_image,
)
from pipeline.shared import SQUARE_CLASS_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

_REAL_CLIP_RE = re.compile(r"^clip_overlay_(?P<video_id>.+?)_clip(?P<clip_id>\d+)_\d+\.pt$")
ANNOTATION_DIRS = [
    _PROJECT_ROOT / "data" / "physical" / "train",
    _PROJECT_ROOT / "data" / "physical" / "val",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_rgb_jpeg(path: Path, image_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _get_clip_db_info(clip_id: int) -> dict[str, Any] | None:
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
    except Exception as e:  # pragma: no cover - exercised in integration use
        logger.warning("DB lookup failed for clip %d: %s", clip_id, e)
        return None


def _get_video_path(video_id: str) -> str | None:
    try:
        from pipeline.download.video_downloader import get_video_path

        return get_video_path(video_id)
    except Exception:  # pragma: no cover - exercised in integration use
        return None


def _load_clip_payload(
    clip_cache: dict[str, dict[str, Any]],
    clip_path: str,
) -> dict[str, Any] | None:
    cached = clip_cache.get(clip_path)
    if cached is not None:
        return cached

    absolute_path = _PROJECT_ROOT / clip_path
    if not absolute_path.exists():
        return None

    payload = torch.load(absolute_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    clip_cache[clip_path] = payload
    return payload


def _clip_frame_size(clip_payload: dict[str, Any]) -> tuple[int, int] | None:
    frames = clip_payload.get("frames")
    if not isinstance(frames, torch.Tensor) or frames.ndim < 3:
        return None
    sample = frames[0]
    if sample.ndim == 3:
        if sample.shape[0] == 3:
            return int(sample.shape[2]), int(sample.shape[1])
        if sample.shape[-1] == 3:
            return int(sample.shape[1]), int(sample.shape[0])
    return int(frames.shape[-1]), int(frames.shape[-2])


def _source_frame_index(clip_payload: dict[str, Any], frame_index: int) -> int:
    frame_indices = clip_payload.get("frame_indices")
    if isinstance(frame_indices, torch.Tensor) and 0 <= frame_index < frame_indices.shape[0]:
        return int(frame_indices[frame_index].item())
    return int(frame_index)


def _get_native_frame_and_bbox(
    video_path: str,
    camera_bbox: tuple[int, int, int, int],
    ref_resolution: tuple[int, int],
    source_frame_index: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_width, ref_height = ref_resolution
        scale_x = frame_width / ref_width if ref_width > 0 else 1.0
        scale_y = frame_height / ref_height if ref_height > 0 else 1.0
        x, y, width, height = camera_bbox
        bbox_x = max(0, min(int(round(x * scale_x)), frame_width - 1))
        bbox_y = max(0, min(int(round(y * scale_y)), frame_height - 1))
        bbox_w = max(1, min(int(round(width * scale_x)), frame_width - bbox_x))
        bbox_h = max(1, min(int(round(height * scale_y)), frame_height - bbox_y))

        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        crop = frame[bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (bbox_x, bbox_y, bbox_w, bbox_h)
    finally:
        cap.release()


def _coerce_corners(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    corners: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            corners.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return corners


def _coerce_size(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])
    except (TypeError, ValueError):
        return None


def _has_native_metadata(record: dict[str, Any]) -> bool:
    return (
        record.get("corner_space") == "clip_frame"
        and _coerce_size(record.get("clip_frame_size")) is not None
        and _coerce_corners(record.get("native_corners")) is not None
        and _coerce_bbox(record.get("native_image_bbox")) is not None
        and isinstance(record.get("source_frame_index"), int)
    )


def _scale_corners_to_native_crop(
    corners: list[list[float]],
    *,
    clip_frame_size: tuple[int, int],
    native_crop_size: tuple[int, int],
) -> list[list[float]]:
    clip_width, clip_height = clip_frame_size
    native_width, native_height = native_crop_size
    scale_x = float(native_width) / max(float(clip_width), 1.0)
    scale_y = float(native_height) / max(float(clip_height), 1.0)
    return [[point[0] * scale_x, point[1] * scale_y] for point in corners]


def _square_rows_for_annotation(
    *,
    annotation_id: str,
    clip_path: str,
    frame_index: int,
    source_video_id: str | None,
    labels: list[Any],
    square_crops: list[np.ndarray],
    squares_dir: Path,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for square_index, raw_label in enumerate(labels):
        if raw_label is None:
            continue
        label = int(raw_label)
        square_name = f"{chr(97 + square_index % 8)}{8 - square_index // 8}"
        crop_path = squares_dir / f"{annotation_id}_{square_name}.jpg"
        _write_rgb_jpeg(crop_path, square_crops[square_index])
        rows.append(
            {
                "annotation_id": annotation_id,
                "clip_path": clip_path,
                "frame_index": frame_index,
                "source_video_id": source_video_id,
                "square_index": square_index,
                "square_name": square_name,
                "label_index": label,
                "label_name": SQUARE_CLASS_NAMES[label],
                "crop_path": str(crop_path.relative_to(_PROJECT_ROOT)),
                "split": split,
            }
        )
    return rows


def migrate_annotations(*, dry_run: bool = False) -> None:
    clip_cache: dict[str, dict[str, Any]] = {}
    total_annotations = 0
    total_migrated = 0
    total_skipped = 0

    for annotation_dir in ANNOTATION_DIRS:
        board_annotations_path = annotation_dir / "board_annotations.jsonl"
        square_manifest_path = annotation_dir / "square_manifest.jsonl"
        boards_dir = annotation_dir / "boards"
        squares_dir = annotation_dir / "squares"

        if not board_annotations_path.exists():
            continue

        board_records = _load_jsonl(board_annotations_path)
        square_records = _load_jsonl(square_manifest_path)
        square_records_by_annotation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for square_record in square_records:
            annotation_id = square_record.get("annotation_id")
            if isinstance(annotation_id, str):
                square_records_by_annotation[annotation_id].append(square_record)

        updated_board_records: list[dict[str, Any]] = []
        updated_square_records: list[dict[str, Any]] = []
        split_total = 0
        split_migrated = 0
        split_skipped = 0

        for record in board_records:
            split_total += 1
            total_annotations += 1
            annotation_id = str(record.get("annotation_id", ""))
            clip_path = str(record.get("clip_path", ""))
            frame_index = int(record.get("frame_index", 0))
            raw_corners = _coerce_corners(record.get("corners"))
            labels = list(record.get("labels", []))

            if raw_corners is None:
                logger.warning("%s: invalid corners — skipping", annotation_id)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            if _has_native_metadata(record):
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            clip_filename = Path(clip_path).name
            match = _REAL_CLIP_RE.match(clip_filename)
            if match is None:
                logger.warning("%s: cannot parse clip filename %s", annotation_id, clip_filename)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            clip_id = int(match.group("clip_id"))
            video_id = match.group("video_id")
            clip_payload = _load_clip_payload(clip_cache, clip_path)
            if clip_payload is None:
                logger.warning("%s: missing clip payload %s", annotation_id, clip_path)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            clip_frame_size = _clip_frame_size(clip_payload)
            if clip_frame_size is None:
                logger.warning("%s: clip has no usable frames tensor", annotation_id)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            db_info = _get_clip_db_info(clip_id)
            if db_info is None or db_info["video_id"] != video_id:
                logger.warning("%s: DB lookup failed for %s", annotation_id, clip_filename)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            camera_bbox = db_info["camera_bbox"]
            if camera_bbox == (0, 0, 100, 100):
                logger.warning("%s: placeholder camera bbox — skipping", annotation_id)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            source_frame_index = _source_frame_index(clip_payload, frame_index)
            video_path = _get_video_path(video_id)
            if video_path is None:
                logger.warning("%s: source video not found for %s", annotation_id, video_id)
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            native_payload = _get_native_frame_and_bbox(
                video_path,
                camera_bbox,
                db_info["ref_resolution"],
                source_frame_index,
            )
            if native_payload is None:
                logger.warning(
                    "%s: failed to read source frame %d from %s",
                    annotation_id,
                    source_frame_index,
                    video_id,
                )
                updated_board_records.append(record)
                updated_square_records.extend(square_records_by_annotation.get(annotation_id, []))
                split_skipped += 1
                total_skipped += 1
                continue

            native_frame, native_image_bbox = native_payload
            native_height, native_width = native_frame.shape[:2]
            native_corners = _scale_corners_to_native_crop(
                raw_corners,
                clip_frame_size=clip_frame_size,
                native_crop_size=(native_width, native_height),
            )

            updated_record = dict(record)
            updated_record["corner_space"] = "clip_frame"
            updated_record["clip_frame_size"] = [clip_frame_size[0], clip_frame_size[1]]
            updated_record["native_corners"] = native_corners
            updated_record["native_image_bbox"] = list(native_image_bbox)
            updated_record["source_frame_index"] = source_frame_index
            updated_board_records.append(updated_record)

            if dry_run:
                logger.info(
                    "[DRY RUN] %s: clip %dx%d -> native %dx%d (scale %.2fx%.2f)",
                    annotation_id,
                    clip_frame_size[0],
                    clip_frame_size[1],
                    native_width,
                    native_height,
                    native_width / max(float(clip_frame_size[0]), 1.0),
                    native_height / max(float(clip_frame_size[1]), 1.0),
                )
            else:
                output_size = int(record.get("rectified_size", DEFAULT_BOARD_SIZE))
                rectified = rectify_board_image(
                    native_frame,
                    native_corners,
                    output_size=output_size,
                )
                board_path = boards_dir / f"{annotation_id}.jpg"
                _write_rgb_jpeg(board_path, rectified)
                square_crops = extract_square_crops(rectified)
                updated_square_records.extend(
                    _square_rows_for_annotation(
                        annotation_id=annotation_id,
                        clip_path=clip_path,
                        frame_index=frame_index,
                        source_video_id=(
                            None
                            if record.get("source_video_id") is None
                            else str(record.get("source_video_id"))
                        ),
                        labels=labels,
                        square_crops=square_crops,
                        squares_dir=squares_dir,
                        split=annotation_dir.name,
                    )
                )

            split_migrated += 1
            total_migrated += 1

        if not dry_run:
            _write_jsonl(board_annotations_path, updated_board_records)
            _write_jsonl(square_manifest_path, updated_square_records)

        logger.info(
            "%s: %d annotations, %d migrated, %d skipped",
            annotation_dir.name,
            split_total,
            split_migrated,
            split_skipped,
        )

    logger.info(
        "Total: %d annotations, %d migrated, %d skipped",
        total_annotations,
        total_migrated,
        total_skipped,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log changes without writing")
    args = parser.parse_args()
    migrate_annotations(dry_run=args.dry_run)
