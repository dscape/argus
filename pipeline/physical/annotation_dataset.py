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
    corner_space: str = "clip_frame"
    clip_frame_size: list[int] | None = None
    native_corners: list[list[float]] | None = None
    native_image_bbox: list[int] | None = None
    source_frame_index: int | None = None


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


@dataclass(frozen=True)
class SavedTransientMoveAnnotation:
    move_index: int
    uci: str
    san: str | None
    move_frame_index: int
    side_to_move: str | None
    fen_before: str | None
    fen_after: str | None
    start_frame_index: int | None
    end_frame_index: int | None
    is_capture: bool | None


@dataclass(frozen=True)
class SavedHandOcclusionSpan:
    start_frame_index: int
    end_frame_index: int


@dataclass(frozen=True)
class SavedTransientAnnotation:
    annotation_id: str
    clip_path: str
    source_video_id: str | None
    total_moves: int
    move_annotations: list[SavedTransientMoveAnnotation]
    hand_occlusion_spans: list[SavedHandOcclusionSpan]
    created_at: str
    updated_at: str


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
    image_corners: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
    corner_space: str = "clip_frame",
    clip_frame_size: list[int] | tuple[int, int] | None = None,
    native_corners: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
    native_image_bbox: list[int] | tuple[int, int, int, int] | None = None,
    source_frame_index: int | None = None,
) -> dict[str, Any]:
    """Persist one manually labeled physical-board frame and its labeled square crops."""
    if len(labels) != 64:
        raise ValueError(f"labels must contain 64 entries, got {len(labels)}")

    dataset_root.mkdir(parents=True, exist_ok=True)
    boards_dir.mkdir(parents=True, exist_ok=True)
    squares_dir.mkdir(parents=True, exist_ok=True)

    annotation_id = f"{Path(clip_path).stem}_frame{frame_index:04d}"
    rectified_board = rectify_board_image(
        image_rgb,
        corners if image_corners is None else image_corners,
        output_size=output_size,
    )
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
            corner_space=str(corner_space),
            clip_frame_size=(
                None
                if clip_frame_size is None
                else [int(clip_frame_size[0]), int(clip_frame_size[1])]
            ),
            native_corners=(
                None
                if native_corners is None
                else [[float(x), float(y)] for x, y in native_corners]
            ),
            native_image_bbox=(
                None if native_image_bbox is None else [int(value) for value in native_image_bbox]
            ),
            source_frame_index=(None if source_frame_index is None else int(source_frame_index)),
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


def load_transient_annotation(
    transient_annotations_path: Path,
    *,
    clip_path: str,
) -> dict[str, Any] | None:
    """Return the saved transient annotation for one clip, if present."""
    for record in _load_jsonl(transient_annotations_path):
        if record.get("clip_path") == clip_path:
            return record
    return None


def delete_transient_annotation(
    transient_annotations_path: Path,
    *,
    clip_path: str,
) -> bool:
    """Delete one saved transient annotation."""
    records = _load_jsonl(transient_annotations_path)
    filtered = [record for record in records if record.get("clip_path") != clip_path]
    if len(filtered) == len(records):
        return False
    _upsert_jsonl(transient_annotations_path, filtered)
    return True


def save_transient_annotation(
    transient_annotations_path: Path,
    *,
    clip_path: str,
    source_video_id: str | None,
    move_annotations: list[dict[str, Any]],
    hand_occlusion_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist clip-level transient move timing and occlusion labels."""
    annotation_id = f"{Path(clip_path).stem}_transient"
    existing_record = load_transient_annotation(transient_annotations_path, clip_path=clip_path)
    move_records = _normalize_transient_move_annotations(move_annotations)
    span_records = _normalize_hand_occlusion_spans(hand_occlusion_spans)
    now = datetime.now(timezone.utc).isoformat()

    annotation_record = asdict(
        SavedTransientAnnotation(
            annotation_id=annotation_id,
            clip_path=clip_path,
            source_video_id=source_video_id,
            total_moves=len(move_records),
            move_annotations=move_records,
            hand_occlusion_spans=span_records,
            created_at=(existing_record or {}).get("created_at", now),
            updated_at=now,
        )
    )

    rows = [
        record
        for record in _load_jsonl(transient_annotations_path)
        if record.get("clip_path") != clip_path
    ]
    rows.append(annotation_record)
    rows.sort(key=lambda record: str(record.get("clip_path", "")))
    _upsert_jsonl(transient_annotations_path, rows)
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
        clip_path: len(frame_indices) for clip_path, frame_indices in frame_indices_by_clip.items()
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


def _normalize_transient_move_annotations(
    move_annotations: list[dict[str, Any]],
) -> list[SavedTransientMoveAnnotation]:
    if not isinstance(move_annotations, list):
        raise ValueError("move_annotations must be a list")

    normalized: list[SavedTransientMoveAnnotation] = []
    seen_move_indices: set[int] = set()
    for raw_record in sorted(
        move_annotations,
        key=lambda record: _required_nonnegative_int(
            record.get("move_index") if isinstance(record, dict) else None,
            name="move_index",
        ),
    ):
        if not isinstance(raw_record, dict):
            raise ValueError("move_annotations entries must be objects")

        move_index = _required_nonnegative_int(raw_record.get("move_index"), name="move_index")
        if move_index in seen_move_indices:
            raise ValueError(f"duplicate move_index in move_annotations: {move_index}")
        seen_move_indices.add(move_index)

        move_frame_index = _required_nonnegative_int(
            raw_record.get("move_frame_index"),
            name="move_frame_index",
        )
        start_frame_index = _optional_nonnegative_int(
            raw_record.get("start_frame_index"),
            name="start_frame_index",
        )
        end_frame_index = _optional_nonnegative_int(
            raw_record.get("end_frame_index"),
            name="end_frame_index",
        )
        if (
            start_frame_index is not None
            and end_frame_index is not None
            and end_frame_index < start_frame_index
        ):
            raise ValueError(
                "end_frame_index must be greater than or equal to start_frame_index"
            )

        uci = raw_record.get("uci")
        if not isinstance(uci, str) or not uci.strip():
            raise ValueError("move_annotations entries must include a non-empty uci")

        san = _optional_string(raw_record.get("san"), name="san")
        fen_before = _optional_string(raw_record.get("fen_before"), name="fen_before")
        fen_after = _optional_string(raw_record.get("fen_after"), name="fen_after")
        side_to_move = _optional_string(raw_record.get("side_to_move"), name="side_to_move")
        if side_to_move not in {None, "white", "black"}:
            raise ValueError("side_to_move must be 'white', 'black', or null")

        is_capture_raw = raw_record.get("is_capture")
        if is_capture_raw is not None and not isinstance(is_capture_raw, bool):
            raise ValueError("is_capture must be true, false, or null")

        normalized.append(
            SavedTransientMoveAnnotation(
                move_index=move_index,
                uci=uci,
                san=san,
                move_frame_index=move_frame_index,
                side_to_move=side_to_move,
                fen_before=fen_before,
                fen_after=fen_after,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                is_capture=is_capture_raw,
            )
        )

    return normalized


def _normalize_hand_occlusion_spans(
    hand_occlusion_spans: list[dict[str, Any]],
) -> list[SavedHandOcclusionSpan]:
    if not isinstance(hand_occlusion_spans, list):
        raise ValueError("hand_occlusion_spans must be a list")

    normalized: list[SavedHandOcclusionSpan] = []
    for raw_record in hand_occlusion_spans:
        if not isinstance(raw_record, dict):
            raise ValueError("hand_occlusion_spans entries must be objects")

        start_frame_index = _required_nonnegative_int(
            raw_record.get("start_frame_index"),
            name="start_frame_index",
        )
        end_frame_index = _required_nonnegative_int(
            raw_record.get("end_frame_index"),
            name="end_frame_index",
        )
        if end_frame_index < start_frame_index:
            raise ValueError(
                "hand occlusion span end_frame_index must be greater than or equal to "
                "start_frame_index"
            )

        normalized.append(
            SavedHandOcclusionSpan(
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
            )
        )

    normalized.sort(key=lambda span: (span.start_frame_index, span.end_frame_index))
    return normalized


def _required_nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _required_nonnegative_int(value, name=name)


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


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
