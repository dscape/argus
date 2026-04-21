from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pipeline.physical.piece_projection import (
    DEFAULT_PIECE_HEIGHT,
    camera_pose_from_corners,
    default_camera_matrix,
    extract_projected_piece_crop,
    piece_bbox_from_projection,
    project_piece_box,
    project_square_base_quad,
)
from pipeline.physical.piece_projection import (
    project_piece_bboxes as _project_piece_bboxes,
)
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.shared.real_board_data import load_real_board_rows
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader, preprocess_square_crop
from pipeline.shared import SQUARE_CLASS_NAMES
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REAL_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_REAL_EVAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_BODY_CLASS_NAME = "piece_body"
_TYPE_CLASS_NAMES = ("empty", *SQUARE_CLASS_NAMES[1:], _BODY_CLASS_NAME)
_CATEGORY_NAMES = {
    "a-file-rook",
    "lateral-occlusion",
    "low-camera-angle",
    "dense-middlegame",
    "mid-move",
    "easy-stationary",
}

EMPTY_TYPE_INDEX = 0
PIECE_BODY_TYPE_INDEX = len(_TYPE_CLASS_NAMES) - 1
CONCRETE_TYPE_INDICES = tuple(range(1, 13))
TYPE_CLASS_NAMES = _TYPE_CLASS_NAMES
TYPE_CLASS_TO_INDEX = {name: index for index, name in enumerate(TYPE_CLASS_NAMES)}
PROJECTED_CROP_MODE = "projected"
NARROW_BOTTOM_WIDTH_CROP_MODE = "narrow-bottom-width"
CROP_MODES = (PROJECTED_CROP_MODE, NARROW_BOTTOM_WIDTH_CROP_MODE)


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


@dataclass(frozen=True)
class SquareSampleIndex:
    row_index: int
    square_index: int
    type_label: int
    base_label: float


class BaseHeadSquareDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        rows: list[BoardRow],
        input_size: int = 224,
        piece_height: float = DEFAULT_PIECE_HEIGHT,
        augment: bool = False,
        body_repeat_factor: int = 1,
        body_overlap_threshold: float = 0.08,
        flip_left_half: bool = True,
        crop_mode: str = PROJECTED_CROP_MODE,
    ) -> None:
        if body_repeat_factor <= 0:
            raise ValueError(f"body_repeat_factor must be > 0, got {body_repeat_factor}")
        if not 0.0 <= body_overlap_threshold <= 1.0:
            raise ValueError(
                f"body_overlap_threshold must be in [0, 1], got {body_overlap_threshold}"
            )
        self.rows = rows
        self.input_size = input_size
        self.piece_height = piece_height
        self.augment = augment
        self.body_repeat_factor = body_repeat_factor
        self.body_overlap_threshold = body_overlap_threshold
        self.flip_left_half = flip_left_half
        self.crop_mode = validate_crop_mode(crop_mode)
        self._clip_cache: dict[Path, dict[str, Any]] = {}
        self._native_loader = NativeFrameLoader()
        self._row_targets: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
        self.indices: list[SquareSampleIndex] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.indices[index]
        row = self.rows[sample.row_index]
        frame_bgr, corners = load_row_frame_and_corners(
            row,
            clip_cache=self._clip_cache,
            native_loader=self._native_loader,
        )
        square_row = sample.square_index // 8
        square_col = sample.square_index % 8
        crop_bgr = extract_study_piece_crop(
            frame_bgr,
            corners,
            row=square_row,
            col=square_col,
            output_size=self.input_size,
            piece_height=self.piece_height,
            flip_left_half=self.flip_left_half,
            crop_mode=self.crop_mode,
        )
        image = preprocess_square_crop(crop_bgr, size=self.input_size, augment=self.augment)
        type_target = torch.tensor(sample.type_label, dtype=torch.long)
        base_target = torch.tensor(float(sample.base_label), dtype=torch.float32)
        return image, type_target, base_target

    def close(self) -> None:
        self._native_loader.close()

    def type_class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in TYPE_CLASS_NAMES}
        for sample in self.indices:
            counts[TYPE_CLASS_NAMES[sample.type_label]] += 1
        return counts

    def base_positive_fraction(self) -> float:
        if not self.indices:
            return 0.0
        positives = sum(1 for sample in self.indices if sample.base_label > 0.5)
        return positives / len(self.indices)

    def _build_index(self) -> None:
        for row_index, row in enumerate(self.rows):
            frame_bgr, corners = load_row_frame_and_corners(
                row,
                clip_cache=self._clip_cache,
                native_loader=self._native_loader,
            )
            type_labels, base_labels = derive_square_targets_from_geometry(
                labels=row.labels,
                corners=corners,
                frame_shape=frame_bgr.shape,
                piece_height=self.piece_height,
                body_overlap_threshold=self.body_overlap_threshold,
                crop_mode=self.crop_mode,
            )
            self._row_targets.append((type_labels, base_labels))
            for square_index, (type_label, base_label) in enumerate(zip(type_labels, base_labels)):
                repeat = (
                    self.body_repeat_factor
                    if type_label == PIECE_BODY_TYPE_INDEX and base_label < 0.5
                    else 1
                )
                for _ in range(repeat):
                    self.indices.append(
                        SquareSampleIndex(
                            row_index=row_index,
                            square_index=square_index,
                            type_label=type_label,
                            base_label=base_label,
                        )
                    )


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
            image_path=None,
            source_video_id=row.source_video_id,
            source_frame_index=None,
            corners=tuple((float(x), float(y)) for x, y in row.corners),
            labels=tuple(int(value) for value in row.labels),
        )
        for row in rows
    ]


def load_annotation_rows(annotation_root: str | Path) -> list[BoardRow]:
    rows = load_annotated_oblique_rows(annotation_root)
    result: list[BoardRow] = []
    for row in rows:
        result.append(
            BoardRow(
                row_id=str(row.annotation_id),
                clip_path=str(row.clip_path),
                frame_index=int(row.frame_index),
                image_path=None,
                source_video_id=row.source_video_id,
                source_frame_index=row.source_frame_index,
                corners=tuple((float(x), float(y)) for x, y in row.corners),
                labels=tuple(int(value) for value in row.labels),
                native_corners=row.native_corners,
                native_image_bbox=row.native_image_bbox,
            )
        )
    return result


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
                square_name_to_index(square)
            pieces.append(EvalPiece(piece_type=piece_type, square=square))
        placed_labels = board_labels_from_pieces(tuple(pieces))
        records.append(
            EvalFrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                category=category,
                corners=tuple((float(x), float(y)) for x, y in raw_corners),
                pieces=tuple(pieces),
                placed_labels=placed_labels,
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


def board_labels_from_pieces(pieces: tuple[EvalPiece, ...]) -> tuple[int, ...]:
    labels = [0] * 64
    for piece in pieces:
        if piece.square is None:
            continue
        square_index = square_name_to_index(piece.square)
        labels[square_index] = TYPE_CLASS_TO_INDEX[piece.piece_type]
    return tuple(labels)


def labels_to_piece_tuples(labels: tuple[int, ...]) -> tuple[tuple[str, str | None], ...]:
    pieces: list[tuple[str, str | None]] = []
    for square_index, label in enumerate(labels):
        if label <= 0:
            continue
        pieces.append((SQUARE_CLASS_NAMES[label], index_to_square_name(square_index)))
    return tuple(sorted(pieces))


def placed_piece_tuples(pieces: tuple[EvalPiece, ...]) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        sorted(
            ((piece.piece_type, piece.square) for piece in pieces),
            key=lambda item: (item[0], "" if item[1] is None else item[1]),
        )
    )


def square_name_to_index(square_name: str) -> int:
    if len(square_name) != 2 or square_name[0] < "a" or square_name[0] > "h":
        raise ValueError(f"Invalid square name: {square_name!r}")
    rank = int(square_name[1])
    if rank < 1 or rank > 8:
        raise ValueError(f"Invalid square name: {square_name!r}")
    file_index = ord(square_name[0]) - ord("a")
    row_index = 8 - rank
    return row_index * 8 + file_index


def index_to_square_name(square_index: int) -> str:
    if square_index < 0 or square_index >= 64:
        raise ValueError(f"square_index out of range: {square_index}")
    row_index = square_index // 8
    file_index = square_index % 8
    rank = 8 - row_index
    return f"{chr(ord('a') + file_index)}{rank}"


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


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_PROJECT_ROOT / candidate).resolve()


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


def validate_crop_mode(crop_mode: str) -> str:
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
    return crop_mode


def square_bottom_edge_x_bounds(
    corners: tuple[tuple[float, float], ...],
    *,
    row: int,
    col: int,
) -> tuple[float, float]:
    square_quad = project_square_base_quad(corners, row=row, col=col)
    edge_indices = max(
        ((0, 1), (1, 2), (2, 3), (3, 0)),
        key=lambda edge: float((square_quad[edge[0], 1] + square_quad[edge[1], 1]) / 2.0),
    )
    x1 = float(min(square_quad[edge_indices[0], 0], square_quad[edge_indices[1], 0]))
    x2 = float(max(square_quad[edge_indices[0], 0], square_quad[edge_indices[1], 0]))
    return x1, x2



def project_piece_bbox_for_crop_mode(
    corners: tuple[tuple[float, float], ...],
    *,
    frame_shape: tuple[int, ...],
    row: int,
    col: int,
    piece_height: float,
    crop_mode: str,
    pose: Any | None = None,
) -> tuple[float, float, float, float]:
    crop_mode = validate_crop_mode(crop_mode)
    if crop_mode == PROJECTED_CROP_MODE:
        if pose is None:
            return tuple(
                float(value)
                for value in _project_piece_bboxes(
                    corners,
                    frame_shape=frame_shape,
                    piece_height=piece_height,
                )[row * 8 + col].tolist()
            )
        projected = project_piece_box(
            pose,
            row=row,
            col=col,
            piece_height=piece_height,
            corners=corners,
        )
        return piece_bbox_from_projection(projected)

    if pose is None:
        pose = camera_pose_from_corners(corners, K=default_camera_matrix(frame_shape))
    projected = project_piece_box(
        pose,
        row=row,
        col=col,
        piece_height=piece_height,
        corners=corners,
    )
    xmin, ymin, xmax, ymax = piece_bbox_from_projection(projected)
    bottom_x1, bottom_x2 = square_bottom_edge_x_bounds(corners, row=row, col=col)
    return bottom_x1, ymin, bottom_x2, ymax



def project_piece_bboxes_for_crop_mode(
    corners: tuple[tuple[float, float], ...],
    *,
    frame_shape: tuple[int, ...],
    piece_height: float,
    crop_mode: str,
) -> np.ndarray:
    crop_mode = validate_crop_mode(crop_mode)
    if crop_mode == PROJECTED_CROP_MODE:
        return _project_piece_bboxes(
            corners,
            frame_shape=frame_shape,
            piece_height=piece_height,
        )
    pose = camera_pose_from_corners(corners, K=default_camera_matrix(frame_shape))
    projected_bboxes = np.empty((64, 4), dtype=np.float64)
    for square_index in range(64):
        projected_bboxes[square_index] = np.asarray(
            project_piece_bbox_for_crop_mode(
                corners,
                frame_shape=frame_shape,
                row=square_index // 8,
                col=square_index % 8,
                piece_height=piece_height,
                crop_mode=crop_mode,
                pose=pose,
            ),
            dtype=np.float64,
        )
    return projected_bboxes



def _axis_aligned_crop(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    output_size: int,
    flip_horizontally: bool,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    height, width = image_bgr.shape[:2]
    clipped_xmin = max(0, int(np.floor(xmin)))
    clipped_ymin = max(0, int(np.floor(ymin)))
    clipped_xmax = min(width, int(np.ceil(xmax)))
    clipped_ymax = min(height, int(np.ceil(ymax)))
    if clipped_xmax <= clipped_xmin or clipped_ymax <= clipped_ymin:
        return np.zeros((output_size, output_size, image_bgr.shape[2]), dtype=image_bgr.dtype)

    crop = image_bgr[clipped_ymin:clipped_ymax, clipped_xmin:clipped_xmax]
    crop_h, crop_w = crop.shape[:2]
    scale = float(output_size) / float(max(crop_h, crop_w))
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    interpolation = cv2.INTER_AREA if max(crop_h, crop_w) >= output_size else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    if flip_horizontally:
        resized = cv2.flip(resized, 1)

    canvas = np.zeros((output_size, output_size, image_bgr.shape[2]), dtype=image_bgr.dtype)
    y_offset = output_size - new_h
    canvas[y_offset : y_offset + new_h, 0:new_w] = resized
    return canvas



def extract_study_piece_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    *,
    row: int,
    col: int,
    output_size: int,
    piece_height: float = DEFAULT_PIECE_HEIGHT,
    flip_left_half: bool = True,
    crop_mode: str = PROJECTED_CROP_MODE,
) -> np.ndarray:
    crop_mode = validate_crop_mode(crop_mode)
    if crop_mode == PROJECTED_CROP_MODE:
        return extract_projected_piece_crop(
            image_bgr,
            corners,
            row=row,
            col=col,
            output_size=output_size,
            piece_height=piece_height,
            flip_left_half=flip_left_half,
        )
    bbox = project_piece_bbox_for_crop_mode(
        corners,
        frame_shape=image_bgr.shape,
        row=row,
        col=col,
        piece_height=piece_height,
        crop_mode=crop_mode,
    )
    flip = flip_left_half and col < 4
    return _axis_aligned_crop(
        image_bgr,
        bbox,
        output_size=output_size,
        flip_horizontally=flip,
    )



def derive_square_targets_from_geometry(
    *,
    labels: tuple[int, ...],
    corners: tuple[tuple[float, float], ...],
    frame_shape: tuple[int, ...],
    piece_height: float,
    body_overlap_threshold: float,
    crop_mode: str = PROJECTED_CROP_MODE,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    projected_bboxes = project_piece_bboxes_for_crop_mode(
        corners,
        frame_shape=frame_shape,
        piece_height=piece_height,
        crop_mode=crop_mode,
    )
    return derive_square_targets_from_bboxes(
        labels=labels,
        projected_bboxes=projected_bboxes,
        body_overlap_threshold=body_overlap_threshold,
    )


def derive_square_targets_from_bboxes(
    *,
    labels: tuple[int, ...],
    projected_bboxes: np.ndarray,
    body_overlap_threshold: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if len(labels) != 64:
        raise ValueError(f"Expected 64 labels, got {len(labels)}")
    if projected_bboxes.shape != (64, 4):
        raise ValueError(
            f"Expected projected_bboxes with shape (64, 4), got {projected_bboxes.shape}"
        )
    occupied_square_indices = [
        square_index for square_index, label in enumerate(labels) if label != 0
    ]
    type_labels: list[int] = []
    base_labels: list[float] = []
    for square_index, label in enumerate(labels):
        if label != 0:
            type_labels.append(int(label))
            base_labels.append(1.0)
            continue
        overlap = max_body_overlap_ratio(
            target_bbox=projected_bboxes[square_index],
            foreign_bboxes=np.asarray(
                [projected_bboxes[index] for index in occupied_square_indices],
                dtype=np.float64,
            ),
        )
        type_labels.append(
            PIECE_BODY_TYPE_INDEX if overlap >= body_overlap_threshold else EMPTY_TYPE_INDEX
        )
        base_labels.append(0.0)
    return tuple(type_labels), tuple(base_labels)


def max_body_overlap_ratio(*, target_bbox: np.ndarray, foreign_bboxes: np.ndarray) -> float:
    if foreign_bboxes.size == 0:
        return 0.0
    overlaps = [_bbox_overlap_ratio(target_bbox, foreign_bbox) for foreign_bbox in foreign_bboxes]
    return max(overlaps, default=0.0)


def _bbox_overlap_ratio(target_bbox: np.ndarray, foreign_bbox: np.ndarray) -> float:
    tx1, ty1, tx2, ty2 = [float(value) for value in target_bbox.tolist()]
    fx1, fy1, fx2, fy2 = [float(value) for value in foreign_bbox.tolist()]
    inter_x1 = max(tx1, fx1)
    inter_y1 = max(ty1, fy1)
    inter_x2 = min(tx2, fx2)
    inter_y2 = min(ty2, fy2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if target_area <= 1e-6:
        return 0.0
    return intersection / target_area


def infer_square_labels(type_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    type_indices = type_logits.argmax(dim=-1)
    base_present = torch.sigmoid(base_logits).squeeze(-1) > 0.5
    predicted = torch.zeros_like(type_indices)
    concrete_mask = (type_indices > EMPTY_TYPE_INDEX) & (type_indices < PIECE_BODY_TYPE_INDEX)
    keep_mask = concrete_mask & base_present
    predicted[keep_mask] = type_indices[keep_mask]
    return predicted


__all__ = [
    "BoardRow",
    "BaseHeadSquareDataset",
    "CONCRETE_TYPE_INDICES",
    "CROP_MODES",
    "EMPTY_TYPE_INDEX",
    "EvalFrameRecord",
    "EvalPiece",
    "NARROW_BOTTOM_WIDTH_CROP_MODE",
    "PIECE_BODY_TYPE_INDEX",
    "PROJECTED_CROP_MODE",
    "TYPE_CLASS_NAMES",
    "TYPE_CLASS_TO_INDEX",
    "board_labels_from_pieces",
    "derive_square_targets_from_bboxes",
    "derive_square_targets_from_geometry",
    "extract_study_piece_crop",
    "frame_tensor_to_bgr",
    "infer_square_labels",
    "index_to_square_name",
    "labels_to_piece_tuples",
    "load_annotation_rows",
    "load_eval_records",
    "load_replay_rows",
    "load_row_frame_and_corners",
    "placed_piece_tuples",
    "project_piece_bbox_for_crop_mode",
    "project_piece_bboxes_for_crop_mode",
    "resolve_project_path",
    "select_rows",
    "square_bottom_edge_x_bounds",
    "square_name_to_index",
    "validate_crop_mode",
]
