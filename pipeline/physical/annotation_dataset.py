"""Shared helpers for physical board-annotation datasets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.shared import SQUARE_CLASS_NAMES

DEFAULT_BOARD_SIZE = 512


@dataclass(frozen=True)
class SavedBoardAnnotation:
    annotation_id: str
    clip_path: str
    frame_index: int
    source_video_id: str | None
    corners: list[list[float]]
    labels: list[int | None]
    labeled_square_count: int
    rectified_board_path: str
    rectified_size: int
    created_at: str


@dataclass(frozen=True)
class SavedSquareCrop:
    annotation_id: str
    clip_path: str
    frame_index: int
    source_video_id: str | None
    square_index: int
    square_name: str
    label_index: int
    label_name: str
    crop_path: str
    split: str


def rectify_board_image(
    image_rgb: np.ndarray,
    corners: list[list[float]] | tuple[tuple[float, float], ...],
    *,
    output_size: int = DEFAULT_BOARD_SIZE,
) -> np.ndarray:
    """Warp a board image to a square top-down view using four corner points."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {points.shape}")

    destination = np.array(
        [
            [0.0, 0.0],
            [output_size - 1.0, 0.0],
            [output_size - 1.0, output_size - 1.0],
            [0.0, output_size - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image_rgb, transform, (output_size, output_size))


def extract_square_crops(board_rgb: np.ndarray) -> list[np.ndarray]:
    """Split a rectified board image into 64 square crops in row-major order."""
    size = board_rgb.shape[0]
    if board_rgb.shape[0] != board_rgb.shape[1]:
        raise ValueError("board image must be square")
    square_size = size // 8
    crops: list[np.ndarray] = []
    for row in range(8):
        for col in range(8):
            y1 = row * square_size
            x1 = col * square_size
            crops.append(board_rgb[y1 : y1 + square_size, x1 : x1 + square_size].copy())
    return crops


def load_board_annotation(
    board_annotations_path: Path,
    *,
    clip_path: str,
    frame_index: int,
) -> dict[str, Any] | None:
    """Return the saved board annotation for one clip frame, if present."""
    for record in _load_jsonl(board_annotations_path):
        record_clip_path = record.get("clip_path")
        record_frame_index = int(record.get("frame_index", -1))
        if record_clip_path == clip_path and record_frame_index == frame_index:
            return record
    return None


def list_board_annotations(board_annotations_path: Path, *, clip_path: str) -> list[dict[str, Any]]:
    """Return all saved board annotations for one clip, sorted by frame."""
    records = [
        record
        for record in _load_jsonl(board_annotations_path)
        if record.get("clip_path") == clip_path
    ]
    return sorted(records, key=lambda record: int(record.get("frame_index", -1)))


def delete_board_annotation(
    project_root: Path,
    *,
    boards_dir: Path,
    squares_dir: Path,
    board_annotations_path: Path,
    square_manifest_path: Path,
    clip_path: str,
    frame_index: int,
) -> bool:
    """Delete a saved board annotation and its associated files."""
    annotation_id = f"{Path(clip_path).stem}_frame{frame_index:04d}"

    board_records = _load_jsonl(board_annotations_path)
    filtered = [record for record in board_records if record.get("annotation_id") != annotation_id]
    if len(filtered) == len(board_records):
        return False

    _upsert_jsonl(board_annotations_path, filtered)
    _upsert_jsonl(
        square_manifest_path,
        [
            record
            for record in _load_jsonl(square_manifest_path)
            if record.get("annotation_id") != annotation_id
        ],
    )

    board_path = boards_dir / f"{annotation_id}.jpg"
    board_path.unlink(missing_ok=True)
    for crop_path in squares_dir.glob(f"{annotation_id}_*.jpg"):
        crop_path.unlink(missing_ok=True)

    return True


def save_board_annotation(
    project_root: Path,
    *,
    dataset_root: Path,
    boards_dir: Path,
    squares_dir: Path,
    board_annotations_path: Path,
    square_manifest_path: Path,
    split: str,
    image_rgb: np.ndarray,
    clip_path: str,
    frame_index: int,
    source_video_id: str | None,
    corners: list[list[float]],
    labels: list[int | None],
    output_size: int = DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    """Persist one manually labeled physical-board frame and its labeled square crops."""
    if len(labels) != 64:
        raise ValueError(f"labels must contain 64 entries, got {len(labels)}")

    dataset_root.mkdir(parents=True, exist_ok=True)
    boards_dir.mkdir(parents=True, exist_ok=True)
    squares_dir.mkdir(parents=True, exist_ok=True)

    annotation_id = f"{Path(clip_path).stem}_frame{frame_index:04d}"
    rectified_board = rectify_board_image(image_rgb, corners, output_size=output_size)
    rectified_board_path = boards_dir / f"{annotation_id}.jpg"
    _write_rgb_jpeg(rectified_board_path, rectified_board)

    for stale_crop_path in squares_dir.glob(f"{annotation_id}_*.jpg"):
        stale_crop_path.unlink(missing_ok=True)

    square_crops = extract_square_crops(rectified_board)
    square_records: list[dict[str, Any]] = []
    labeled_square_count = 0
    for square_index, label in enumerate(labels):
        if label is None:
            continue
        if label < 0 or label >= len(SQUARE_CLASS_NAMES):
            raise ValueError(f"invalid label for square {square_index}: {label}")

        square_name = _square_index_to_name(square_index)
        crop_path = squares_dir / f"{annotation_id}_{square_name}.jpg"
        _write_rgb_jpeg(crop_path, square_crops[square_index])

        labeled_square_count += 1
        square_records.append(
            asdict(
                SavedSquareCrop(
                    annotation_id=annotation_id,
                    clip_path=clip_path,
                    frame_index=frame_index,
                    source_video_id=source_video_id,
                    square_index=square_index,
                    square_name=square_name,
                    label_index=label,
                    label_name=SQUARE_CLASS_NAMES[label],
                    crop_path=str(crop_path.relative_to(project_root)),
                    split=split,
                )
            )
        )

    annotation_record = asdict(
        SavedBoardAnnotation(
            annotation_id=annotation_id,
            clip_path=clip_path,
            frame_index=frame_index,
            source_video_id=source_video_id,
            corners=[[float(x), float(y)] for x, y in corners],
            labels=list(labels),
            labeled_square_count=labeled_square_count,
            rectified_board_path=str(rectified_board_path.relative_to(project_root)),
            rectified_size=output_size,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )

    board_rows = [
        record
        for record in _load_jsonl(board_annotations_path)
        if record.get("annotation_id") != annotation_id
    ]
    board_rows.append(annotation_record)
    _upsert_jsonl(board_annotations_path, board_rows)

    square_rows = [
        record
        for record in _load_jsonl(square_manifest_path)
        if record.get("annotation_id") != annotation_id
    ]
    square_rows.extend(square_records)
    _upsert_jsonl(square_manifest_path, square_rows)

    return annotation_record


def get_saved_frame_counts_by_clip(board_annotations_path: Path) -> dict[str, int]:
    """Return the number of uniquely annotated frames for each clip."""
    frame_indices_by_clip: dict[str, set[int]] = {}
    for record in _load_jsonl(board_annotations_path):
        clip_path = record.get("clip_path")
        if not isinstance(clip_path, str):
            continue
        try:
            frame_index = int(record.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        frame_indices_by_clip.setdefault(clip_path, set()).add(frame_index)

    return {
        clip_path: len(frame_indices)
        for clip_path, frame_indices in frame_indices_by_clip.items()
    }


def get_source_video_ids(board_annotations_path: Path) -> list[str]:
    """Return sorted source video ids represented in one annotation root."""
    source_video_ids = {
        str(record["source_video_id"])
        for record in _load_jsonl(board_annotations_path)
        if record.get("source_video_id")
    }
    return sorted(source_video_ids)


def get_annotation_summary(
    project_root: Path,
    *,
    dataset_root: Path,
    board_annotations_path: Path,
    square_manifest_path: Path,
) -> dict[str, Any]:
    """Return aggregate counts for one physical annotation root."""
    board_records = _load_jsonl(board_annotations_path)
    square_records = _load_jsonl(square_manifest_path)
    class_counts = Counter(
        record["label_name"] for record in square_records if "label_name" in record
    )
    source_video_ids = get_source_video_ids(board_annotations_path)

    recent_annotations = sorted(
        board_records,
        key=lambda record: str(record.get("created_at", "")),
        reverse=True,
    )[:20]

    return {
        "dataset_root": str(dataset_root.relative_to(project_root)),
        "board_annotation_count": len(board_records),
        "square_crop_count": len(square_records),
        "source_video_count": len(source_video_ids),
        "source_video_ids": source_video_ids,
        "class_counts": {name: int(class_counts.get(name, 0)) for name in SQUARE_CLASS_NAMES},
        "recent_annotations": recent_annotations,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _upsert_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(f"{content}\n" if content else "")


def _write_rgb_jpeg(path: Path, image_rgb: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not encoded:
        raise ValueError(f"failed to encode image for {path}")
    path.write_bytes(buffer.tobytes())


def _square_index_to_name(square_index: int) -> str:
    row, col = divmod(square_index, 8)
    return f"{chr(ord('a') + col)}{8 - row}"
