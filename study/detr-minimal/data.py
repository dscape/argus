from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pipeline.physical.board_probe.board_data import (
    prepare_board_neighborhood_geometry,
    preprocess_board_neighborhood_image,
)
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.shared.real_board_data import load_real_board_rows
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader
from pipeline.shared import SQUARE_CLASS_NAMES
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REAL_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_REAL_EVAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_CATEGORY_NAMES = {
    "a-file-rook",
    "lateral-occlusion",
    "low-camera-angle",
    "dense-middlegame",
    "mid-move",
    "easy-stationary",
}
_FILES = "abcdefgh"
_RANKS = "12345678"
_SQUARE_OUTPUT_NAMES = tuple(f"{file}{rank}" for rank in _RANKS for file in _FILES) + ("no_square",)
_TYPE_CLASS_NAMES = ("no_piece", *SQUARE_CLASS_NAMES[1:])

NO_PIECE_TYPE_INDEX = 0
NO_SQUARE_INDEX = len(_SQUARE_OUTPUT_NAMES) - 1
TYPE_CLASS_NAMES = _TYPE_CLASS_NAMES
TYPE_CLASS_TO_INDEX = {name: index for index, name in enumerate(TYPE_CLASS_NAMES)}
SQUARE_OUTPUT_NAMES = _SQUARE_OUTPUT_NAMES
SQUARE_OUTPUT_TO_INDEX = {name: index for index, name in enumerate(SQUARE_OUTPUT_NAMES)}
DETECTION_CLASS_NAMES = TYPE_CLASS_NAMES[1:]
DETECTION_CLASS_TO_INDEX = {name: index for index, name in enumerate(DETECTION_CLASS_NAMES)}
DETECTION_CLASS_COUNT = len(DETECTION_CLASS_NAMES)


@dataclass(frozen=True)
class BoardRow:
    row_id: str
    corners: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    clip_path: str | None = None
    frame_index: int | None = None
    image_path: str | None = None
    source_video_id: str | None = None
    source_frame_index: int | None = None
    native_corners: tuple[tuple[float, float], ...] | None = None
    native_image_bbox: tuple[int, int, int, int] | None = None
    category: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class EvalPiece:
    piece_type: str
    square: str | None


@dataclass(frozen=True)
class EvalFrameRecord:
    frame_id: str
    image_path: str
    category: str
    corners: tuple[tuple[float, float], ...]
    pieces: tuple[EvalPiece, ...]
    placed_labels: tuple[int, ...]
    source_video_id: str | None = None
    source_frame_index: int | None = None
    notes: str | None = None


class DetrBoardDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        *,
        rows: list[BoardRow],
        image_size: int = 224,
        target_mode: str = "minimal_detr",
    ) -> None:
        if target_mode not in {"minimal_detr", "rt_detr"}:
            raise ValueError(f"Unsupported target_mode: {target_mode}")
        self.rows = rows
        self.image_size = image_size
        self.target_mode = target_mode
        self._clip_cache: dict[Path, dict[str, Any]] = {}
        self._native_loader = NativeFrameLoader()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.rows[index]
        frame_bgr, corners = load_row_frame_and_corners(
            row,
            clip_cache=self._clip_cache,
            native_loader=self._native_loader,
        )
        if self.target_mode == "minimal_detr":
            image, _scaled_corners = preprocess_board_neighborhood_image(
                frame_bgr,
                corners,
                size=self.image_size,
            )
            type_targets, square_targets = labels_to_piece_targets(row.labels)
            target = {
                "piece_types": torch.tensor(type_targets, dtype=torch.long),
                "square_indices": torch.tensor(square_targets, dtype=torch.long),
            }
            return image, target

        image, _scaled_corners, square_boxes = prepare_board_neighborhood_geometry(
            frame_bgr,
            corners,
            size=self.image_size,
        )
        class_labels, boxes = labels_to_detection_targets(
            row.labels,
            square_boxes=square_boxes,
            image_size=self.image_size,
        )
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        if box_tensor.numel() == 0:
            box_tensor = torch.empty((0, 4), dtype=torch.float32)
        target = {
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": box_tensor,
            "board_labels": torch.tensor(row.labels, dtype=torch.long),
            "square_boxes": square_boxes.to(torch.float32),
        }
        return image, target

    def close(self) -> None:
        self._native_loader.close()


def load_replay_rows(
    *,
    clips_dir: str | Path = _DEFAULT_REAL_CLIPS_DIR,
    eval_root: str | Path = _DEFAULT_REAL_EVAL_ROOT,
    frame_stride: int = 1,
    max_frames: int | None = None,
    seed: int = 42,
    exclude_move_neighborhood: int = 1,
) -> list[BoardRow]:
    rows = load_real_board_rows(
        clips_dir=clips_dir,
        eval_root=eval_root,
        frame_stride=frame_stride,
        max_frames=max_frames,
        seed=seed,
        exclude_move_neighborhood=exclude_move_neighborhood,
    )
    return [
        BoardRow(
            row_id=f"{row.clip_path}:{row.frame_index}",
            clip_path=row.clip_path,
            frame_index=row.frame_index,
            corners=tuple((float(x), float(y)) for x, y in row.corners),
            labels=tuple(int(value) for value in row.labels),
            source_video_id=row.source_video_id,
        )
        for row in rows
    ]


def load_annotation_rows(annotation_root: str | Path) -> list[BoardRow]:
    rows = load_annotated_oblique_rows(annotation_root)
    return [
        BoardRow(
            row_id=str(row.annotation_id),
            clip_path=str(row.clip_path),
            frame_index=int(row.frame_index),
            corners=tuple((float(x), float(y)) for x, y in row.corners),
            labels=tuple(int(value) for value in row.labels),
            source_video_id=row.source_video_id,
            source_frame_index=row.source_frame_index,
            native_corners=row.native_corners,
            native_image_bbox=row.native_image_bbox,
        )
        for row in rows
    ]


def select_rows(rows: list[BoardRow], *, max_rows: int | None, seed: int) -> list[BoardRow]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:max_rows]


def load_eval_records(labels_path: str | Path) -> list[EvalFrameRecord]:
    path = Path(labels_path)
    if not path.is_absolute():
        path = (_PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise ValueError(f"Eval labels not found: {path}")

    records: list[EvalFrameRecord] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        category = str(payload.get("category", "")).strip()
        if category not in _CATEGORY_NAMES:
            raise ValueError(f"Invalid eval category on line {line_number}: {category!r}")
        frame_id = str(payload.get("frame_id", "")).strip()
        image_path = str(payload.get("image_path", "")).strip()
        raw_corners = payload.get("corners")
        raw_pieces = payload.get("pieces")
        if not frame_id:
            raise ValueError(f"Eval row {line_number} is missing frame_id")
        if not image_path:
            raise ValueError(f"Eval row {line_number} is missing image_path")
        if not isinstance(raw_corners, list) or len(raw_corners) != 4:
            raise ValueError(f"Eval row {line_number} has invalid corners")
        if not isinstance(raw_pieces, list):
            raise ValueError(f"Eval row {line_number} has invalid pieces list")
        pieces: list[EvalPiece] = []
        for raw_piece in raw_pieces:
            if not isinstance(raw_piece, dict):
                raise ValueError(f"Eval row {line_number} has invalid piece payload")
            piece_type = str(raw_piece.get("type", "")).strip()
            if piece_type not in SQUARE_CLASS_NAMES[1:]:
                raise ValueError(f"Eval row {line_number} has invalid piece type: {piece_type!r}")
            square = raw_piece.get("square")
            if square is not None:
                square = str(square)
                square_name_to_board_index(square)
            pieces.append(EvalPiece(piece_type=piece_type, square=square))
        records.append(
            EvalFrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                category=category,
                corners=tuple((float(x), float(y)) for x, y in raw_corners),
                pieces=tuple(pieces),
                placed_labels=board_labels_from_eval_pieces(tuple(pieces)),
                source_video_id=(
                    None
                    if payload.get("source_video_id") is None
                    else str(payload.get("source_video_id"))
                ),
                source_frame_index=(
                    None
                    if payload.get("source_frame_index") is None
                    else int(payload.get("source_frame_index"))
                ),
                notes=None if payload.get("notes") is None else str(payload.get("notes")),
            )
        )
    return records


def labels_to_piece_targets(labels: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    piece_types: list[int] = []
    square_indices: list[int] = []
    for board_index, label in enumerate(labels):
        if label <= 0:
            continue
        piece_types.append(int(label))
        square_indices.append(board_index_to_square_output_index(board_index))
    return tuple(piece_types), tuple(square_indices)


def board_label_to_detection_class(label: int) -> int:
    if label <= 0 or label >= len(TYPE_CLASS_NAMES):
        raise ValueError(
            f"Expected placed piece label in [1, {len(TYPE_CLASS_NAMES) - 1}], got {label}"
        )
    return label - 1


def detection_class_to_board_label(class_index: int) -> int:
    if class_index < 0 or class_index >= DETECTION_CLASS_COUNT:
        raise ValueError(
            f"Expected detection class in [0, {DETECTION_CLASS_COUNT - 1}], got {class_index}"
        )
    return class_index + 1


def labels_to_detection_targets(
    labels: tuple[int, ...],
    *,
    square_boxes: torch.Tensor,
    image_size: int,
) -> tuple[tuple[int, ...], tuple[tuple[float, float, float, float], ...]]:
    if square_boxes.shape != (64, 4):
        raise ValueError(
            f"Expected square_boxes with shape (64, 4), got {tuple(square_boxes.shape)}"
        )
    class_labels: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    for square_index, label in enumerate(labels):
        if label <= 0:
            continue
        class_labels.append(board_label_to_detection_class(int(label)))
        boxes.append(
            xyxy_box_to_normalized_cxcywh(square_boxes[square_index], image_size=image_size)
        )
    return tuple(class_labels), tuple(boxes)


def xyxy_box_to_normalized_cxcywh(
    box: torch.Tensor | np.ndarray,
    *,
    image_size: int,
) -> tuple[float, float, float, float]:
    if image_size <= 0:
        raise ValueError(f"image_size must be > 0, got {image_size}")
    if isinstance(box, np.ndarray):
        values = [float(value) for value in box.tolist()]
    else:
        values = [float(value) for value in box.detach().cpu().tolist()]
    xmin, ymin, xmax, ymax = values
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)
    center_x = xmin + (width / 2.0)
    center_y = ymin + (height / 2.0)
    scale = float(image_size)
    return (
        center_x / scale,
        center_y / scale,
        width / scale,
        height / scale,
    )


def board_labels_from_eval_pieces(pieces: tuple[EvalPiece, ...]) -> tuple[int, ...]:
    labels = [0] * 64
    for piece in pieces:
        if piece.square is None:
            continue
        board_index = square_name_to_board_index(piece.square)
        labels[board_index] = TYPE_CLASS_TO_INDEX[piece.piece_type]
    return tuple(labels)


def placed_piece_tuples(pieces: tuple[EvalPiece, ...]) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        sorted(
            ((piece.piece_type, piece.square) for piece in pieces),
            key=lambda item: (item[0], "" if item[1] is None else item[1]),
        )
    )


def labels_to_piece_tuples(labels: tuple[int, ...]) -> tuple[tuple[str, str | None], ...]:
    pieces: list[tuple[str, str | None]] = []
    for board_index, label in enumerate(labels):
        if label <= 0:
            continue
        pieces.append((SQUARE_CLASS_NAMES[label], board_index_to_square_name(board_index)))
    return tuple(sorted(pieces))


def board_index_to_square_name(board_index: int) -> str:
    row_index = board_index // 8
    file_index = board_index % 8
    rank = 8 - row_index
    return f"{chr(ord('a') + file_index)}{rank}"


def square_name_to_board_index(square_name: str) -> int:
    if len(square_name) != 2 or square_name[0] < "a" or square_name[0] > "h":
        raise ValueError(f"Invalid square name: {square_name!r}")
    rank = int(square_name[1])
    if rank < 1 or rank > 8:
        raise ValueError(f"Invalid square name: {square_name!r}")
    file_index = ord(square_name[0]) - ord("a")
    row_index = 8 - rank
    return row_index * 8 + file_index


def board_index_to_square_output_index(board_index: int) -> int:
    return SQUARE_OUTPUT_TO_INDEX[board_index_to_square_name(board_index)]


def square_output_index_to_board_index(square_index: int) -> int | None:
    if square_index == NO_SQUARE_INDEX:
        return None
    if square_index < 0 or square_index >= NO_SQUARE_INDEX:
        raise ValueError(f"square_index out of range: {square_index}")
    return square_name_to_board_index(SQUARE_OUTPUT_NAMES[square_index])


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_PROJECT_ROOT / candidate).resolve()


def load_row_frame_and_corners(
    row: BoardRow,
    *,
    clip_cache: dict[Path, dict[str, Any]],
    native_loader: NativeFrameLoader,
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    if row.image_path is not None:
        image_path = resolve_project_path(row.image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return image, row.corners
    if (
        row.source_video_id is not None
        and row.source_frame_index is not None
        and row.native_corners is not None
        and row.native_image_bbox is not None
    ):
        frame = native_loader.load(
            source_video_id=row.source_video_id,
            source_frame_index=row.source_frame_index,
        )
        x_off, y_off, _width, _height = row.native_image_bbox
        corners = tuple((float(x + x_off), float(y + y_off)) for x, y in row.native_corners)
        return frame, corners
    if row.clip_path is None or row.frame_index is None:
        raise ValueError(f"Row has no image or clip payload: {row.row_id}")
    clip_path = resolve_project_path(row.clip_path)
    clip = clip_cache.get(clip_path)
    if clip is None:
        loaded = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid clip payload: {clip_path}")
        clip_cache[clip_path] = loaded
        clip = loaded
    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor):
        raise ValueError(f"Clip has no frames tensor: {clip_path}")
    frame = frames[int(row.frame_index)]
    return frame_tensor_to_bgr(frame), row.corners


def frame_tensor_to_bgr(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with 3 dims, got {tuple(frame.shape)}")
    if frame.shape[0] == 3:
        chw = frame
    elif frame.shape[-1] == 3:
        chw = frame.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB frame tensor, got {tuple(frame.shape)}")
    if chw.dtype == torch.uint8:
        rgb = chw.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        rgb = chw.to(torch.float32)
        if float(rgb.max().item()) <= 1.0:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


__all__ = [
    "BoardRow",
    "DETECTION_CLASS_COUNT",
    "DETECTION_CLASS_NAMES",
    "DETECTION_CLASS_TO_INDEX",
    "DetrBoardDataset",
    "EvalFrameRecord",
    "EvalPiece",
    "NO_PIECE_TYPE_INDEX",
    "NO_SQUARE_INDEX",
    "SQUARE_OUTPUT_NAMES",
    "TYPE_CLASS_NAMES",
    "board_label_to_detection_class",
    "board_index_to_square_output_index",
    "board_labels_from_eval_pieces",
    "detection_class_to_board_label",
    "labels_to_piece_targets",
    "labels_to_detection_targets",
    "labels_to_piece_tuples",
    "load_annotation_rows",
    "load_eval_records",
    "load_replay_rows",
    "placed_piece_tuples",
    "resolve_project_path",
    "select_rows",
    "square_name_to_board_index",
    "square_output_index_to_board_index",
    "xyxy_box_to_normalized_cxcywh",
]
