"""Datasets and preprocessing for piece-projection board-probe inputs."""

from __future__ import annotations

import json
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from argus.chess.board_state import fen_to_square_targets
from pipeline.physical.piece_projection import (
    extract_board_neighborhood_crop,
    project_piece_bboxes,
    transform_projected_bboxes_to_crop_space,
)
from pipeline.physical.shared import splits
from pipeline.physical.shared.source_video_paths import resolve_source_video_path
from pipeline.shared import SQUARE_CLASS_NAMES

INPUT_SIZE = 224
NUM_SQUARE_CLASSES = len(SQUARE_CLASS_NAMES)
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_DEFAULT_TRAIN_ROOT = _PROJECT_ROOT / "data" / "physical" / "train"
_DEFAULT_SYNTHETIC_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train"
_DEFAULT_BOARD_CROP_MARGIN = 0.18


@dataclass(frozen=True)
class PhysicalEvalBoardRow:
    annotation_id: str
    board_path: str
    labels: tuple[int, ...]
    source_video_id: str | None
    corners: tuple[tuple[float, float], ...]
    clip_path: str | None = None
    frame_index: int | None = None
    source_frame_index: int | None = None
    native_corners: tuple[tuple[float, float], ...] | None = None
    native_image_bbox: tuple[int, int, int, int] | None = None
    clip_frame_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class PhysicalSyntheticBoardRow:
    clip_path: str
    frame_index: int
    labels: tuple[int, ...]


class PhysicalSyntheticClipBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Synthetic full-board frames with image-corner geometry."""

    def __init__(
        self,
        *,
        clips_dir: str | Path = _DEFAULT_SYNTHETIC_CLIPS_DIR,
        num_positions: int | None = None,
        image_size: int = INPUT_SIZE,
        seed: int = 42,
        rows: list[PhysicalSyntheticBoardRow] | None = None,
    ) -> None:
        self.clips_dir = Path(clips_dir)
        self.image_size = image_size
        self.seed = seed
        self.rows = rows or load_synthetic_board_rows(
            clips_dir=self.clips_dir,
            num_positions=num_positions,
            seed=seed,
        )
        self._clip_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        clip = self._load_clip(resolve_project_path(row.clip_path))
        frames = clip.get("frames")
        if not isinstance(frames, torch.Tensor):
            raise ValueError(f"Synthetic clip has no frames tensor: {row.clip_path}")
        frame_bgr = _frame_tensor_to_bgr(frames[row.frame_index])
        image = preprocess_board_image(frame_bgr, size=self.image_size)
        corners = _scaled_full_frame_corners(frame_bgr, size=self.image_size)
        targets = torch.tensor(row.labels, dtype=torch.long)
        return image, targets, corners

    def _load_clip(self, clip_path: Path) -> dict[str, Any]:
        cached = self._clip_cache.get(clip_path)
        if cached is not None:
            return cached
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip, dict):
            raise ValueError(f"Invalid synthetic clip payload: {clip_path}")
        self._clip_cache[clip_path] = clip
        return clip


class PhysicalAnnotatedBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Annotated real boards with projected piece-box geometry in board-crop space."""

    def __init__(
        self,
        *,
        annotation_root: str | Path,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalBoardRow] | None = None,
        crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
    ) -> None:
        self.annotation_root = Path(annotation_root)
        self.image_size = image_size
        self.crop_margin = crop_margin
        self.rows = rows or load_annotated_board_rows(self.annotation_root)
        self._clip_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_bgr = load_annotated_board_frame_bgr(row, clip_cache=self._clip_cache)
        image, _scaled_corners, piece_bboxes = preprocess_board_neighborhood_geometry(
            image_bgr,
            row.corners,
            size=self.image_size,
            crop_margin=self.crop_margin,
        )
        targets = torch.tensor(row.labels, dtype=torch.long)
        return image, targets, piece_bboxes


class PhysicalValBoardDataset(PhysicalAnnotatedBoardDataset):
    """Held-out real boards used for physical validation."""

    def __init__(
        self,
        *,
        annotation_root: str | Path = _DEFAULT_VAL_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalBoardRow] | None = None,
        crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
    ) -> None:
        super().__init__(
            annotation_root=annotation_root,
            image_size=image_size,
            rows=rows,
            crop_margin=crop_margin,
        )


class PhysicalTrainBoardDataset(PhysicalAnnotatedBoardDataset):
    """Manually labeled non-held-out boards for physical training."""

    def __init__(
        self,
        *,
        annotation_root: str | Path = _DEFAULT_TRAIN_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalBoardRow] | None = None,
        crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
    ) -> None:
        super().__init__(
            annotation_root=annotation_root,
            image_size=image_size,
            rows=rows,
            crop_margin=crop_margin,
        )


PhysicalEvalBoardDataset = PhysicalValBoardDataset
PhysicalManualTrainBoardDataset = PhysicalTrainBoardDataset


def load_annotated_board_rows(annotation_root: str | Path) -> list[PhysicalEvalBoardRow]:
    splits.ensure_annotation_layout_migrated()
    annotations_path = Path(annotation_root) / "board_annotations.jsonl"
    if not annotations_path.exists():
        raise ValueError(f"Physical board annotations not found: {annotations_path}")

    rows: list[PhysicalEvalBoardRow] = []
    for line in annotations_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        raw_labels = payload.get("labels")
        raw_corners = payload.get("corners")
        if (
            not isinstance(raw_labels, list)
            or len(raw_labels) != 64
            or any(label is None for label in raw_labels)
            or not isinstance(raw_corners, list)
            or len(raw_corners) != 4
        ):
            continue
        if payload.get("clip_path") is None or payload.get("frame_index") is None:
            continue
        source_video_id = (
            str(payload["source_video_id"])
            if payload.get("source_video_id") is not None
            else None
        )
        native_corners = None
        raw_native_corners = payload.get("native_corners")
        if isinstance(raw_native_corners, list) and len(raw_native_corners) == 4:
            native_corners = tuple((float(x), float(y)) for x, y in raw_native_corners)
        native_image_bbox = None
        raw_native_image_bbox = payload.get("native_image_bbox")
        if isinstance(raw_native_image_bbox, list) and len(raw_native_image_bbox) == 4:
            native_image_bbox = tuple(int(value) for value in raw_native_image_bbox)
        clip_frame_size = None
        raw_clip_frame_size = payload.get("clip_frame_size")
        if isinstance(raw_clip_frame_size, list) and len(raw_clip_frame_size) == 2:
            clip_frame_size = (int(raw_clip_frame_size[0]), int(raw_clip_frame_size[1]))

        corners = tuple((float(x), float(y)) for x, y in raw_corners)
        if native_corners is not None and native_image_bbox is not None:
            x_off, y_off, _width, _height = native_image_bbox
            corners = tuple((float(x + x_off), float(y + y_off)) for x, y in native_corners)

        rows.append(
            PhysicalEvalBoardRow(
                annotation_id=str(payload["annotation_id"]),
                board_path=str(payload["rectified_board_path"]),
                labels=tuple(int(label) for label in raw_labels),
                source_video_id=source_video_id,
                corners=corners,
                clip_path=str(payload["clip_path"]),
                frame_index=int(payload["frame_index"]),
                source_frame_index=(
                    int(payload["source_frame_index"])
                    if payload.get("source_frame_index") is not None
                    else None
                ),
                native_corners=native_corners,
                native_image_bbox=native_image_bbox,
                clip_frame_size=clip_frame_size,
            )
        )
    return rows


def load_synthetic_board_rows(
    *,
    clips_dir: str | Path = _DEFAULT_SYNTHETIC_CLIPS_DIR,
    num_positions: int | None = None,
    seed: int = 42,
) -> list[PhysicalSyntheticBoardRow]:
    clip_dir_path = Path(clips_dir)
    if not clip_dir_path.is_absolute():
        clip_dir_path = (_PROJECT_ROOT / clip_dir_path).resolve()
    if not clip_dir_path.exists():
        raise ValueError(f"Synthetic clips directory not found: {clip_dir_path}")

    rows: list[PhysicalSyntheticBoardRow] = []
    for clip_path in sorted(clip_dir_path.glob("*.pt")):
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip, dict):
            continue
        frames = clip.get("frames")
        fens = clip.get("fens")
        if not isinstance(frames, torch.Tensor) or not isinstance(fens, list):
            continue
        board_flipped = bool(clip.get("board_flipped", False))
        frame_count = min(int(frames.shape[0]), len(fens))
        for frame_index in range(frame_count):
            fen = fens[frame_index]
            if not isinstance(fen, str) or not fen:
                continue
            labels = fen_to_square_targets(fen, board_flipped=board_flipped)
            rows.append(
                PhysicalSyntheticBoardRow(
                    clip_path=path_for_storage(clip_path.resolve()),
                    frame_index=frame_index,
                    labels=tuple(int(value) for value in labels.tolist()),
                )
            )

    if num_positions is None:
        return rows
    if num_positions < 0:
        raise ValueError(f"num_positions must be >= 0, got {num_positions}")
    if num_positions > len(rows):
        raise ValueError(
            "Requested more labeled synthetic frames than available in"
            f" {clip_dir_path}: requested {num_positions}, found {len(rows)}"
        )
    rng = random.Random(seed)
    sampled_rows = rows.copy()
    rng.shuffle(sampled_rows)
    return sampled_rows[:num_positions]


def load_val_board_rows(
    annotation_root: str | Path = _DEFAULT_VAL_ROOT,
) -> list[PhysicalEvalBoardRow]:
    return load_annotated_board_rows(annotation_root)


def load_eval_board_rows(
    eval_root: str | Path = _DEFAULT_VAL_ROOT,
) -> list[PhysicalEvalBoardRow]:
    return load_annotated_board_rows(eval_root)


def preprocess_board_image(image_bgr: np.ndarray, *, size: int = INPUT_SIZE) -> torch.Tensor:
    """Resize BGR input, convert to RGB, and apply DINO/ImageNet normalization."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return normalize_rgb_tensor(tensor)


def prepare_board_neighborhood_geometry(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    size: int = INPUT_SIZE,
    crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crop = extract_board_neighborhood_crop(image_bgr, corners, crop_margin=crop_margin)
    rgb = cv2.cvtColor(crop.image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    height, width = crop.image_bgr.shape[:2]
    scaled_corners = crop.corners.copy()
    scaled_corners[:, 0] *= float(size) / max(float(width), 1.0)
    scaled_corners[:, 1] *= float(size) / max(float(height), 1.0)

    piece_bboxes = project_piece_bboxes(corners, frame_shape=image_bgr.shape)
    scaled_piece_bboxes = transform_projected_bboxes_to_crop_space(
        piece_bboxes,
        crop,
        output_shape=size,
    )
    return (
        tensor,
        torch.from_numpy(scaled_corners.astype(np.float32)),
        torch.from_numpy(scaled_piece_bboxes.astype(np.float32)),
    )


def prepare_board_neighborhood_image(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    size: int = INPUT_SIZE,
    crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor, scaled_corners, _scaled_piece_bboxes = prepare_board_neighborhood_geometry(
        image_bgr,
        corners,
        size=size,
        crop_margin=crop_margin,
    )
    return tensor, scaled_corners


def preprocess_board_neighborhood_image(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    size: int = INPUT_SIZE,
    crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor, scaled_corners = prepare_board_neighborhood_image(
        image_bgr,
        corners,
        size=size,
        crop_margin=crop_margin,
    )
    return normalize_rgb_tensor(tensor), scaled_corners


def preprocess_board_neighborhood_geometry(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    size: int = INPUT_SIZE,
    crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensor, scaled_corners, scaled_piece_bboxes = prepare_board_neighborhood_geometry(
        image_bgr,
        corners,
        size=size,
        crop_margin=crop_margin,
    )
    return normalize_rgb_tensor(tensor), scaled_corners, scaled_piece_bboxes


def normalize_rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to an RGB tensor in [0, 1]."""
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def path_for_storage(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_PROJECT_ROOT / candidate).resolve()


def _scaled_full_frame_corners(frame_bgr: np.ndarray, *, size: int) -> torch.Tensor:
    height, width = frame_bgr.shape[:2]
    scale_x = float(size) / max(float(width), 1.0)
    scale_y = float(size) / max(float(height), 1.0)
    return torch.tensor(
        [
            [0.0, 0.0],
            [(width - 1.0) * scale_x, 0.0],
            [(width - 1.0) * scale_x, (height - 1.0) * scale_y],
            [0.0, (height - 1.0) * scale_y],
        ],
        dtype=torch.float32,
    )


class _NativeFrameLoader:
    def __init__(self, *, capacity: int = 64) -> None:
        self._capacity = capacity
        self._frames: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._lock = threading.Lock()

    def load(self, *, source_video_id: str, source_frame_index: int) -> np.ndarray:
        video_path = resolve_source_video_path(source_video_id)
        key = (video_path, int(source_frame_index))
        with self._lock:
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
                return cached.copy()
            capture = self._captures.get(video_path)
            if capture is None:
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise ValueError(f"Failed to open video {video_path}")
                self._captures[video_path] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Failed to read frame {source_frame_index} from {video_path}")
            cached_frame = frame.copy()
            self._frames[key] = cached_frame
            if len(self._frames) > self._capacity:
                self._frames.popitem(last=False)
            return cached_frame.copy()


_NATIVE_FRAME_LOADER = _NativeFrameLoader()


def load_annotated_board_frame_bgr(
    row: PhysicalEvalBoardRow,
    *,
    clip_cache: dict[Path, dict[str, Any]],
) -> np.ndarray:
    if row.source_video_id is not None and row.source_frame_index is not None:
        return _NATIVE_FRAME_LOADER.load(
            source_video_id=row.source_video_id,
            source_frame_index=row.source_frame_index,
        )
    if row.clip_path is None or row.frame_index is None:
        raise ValueError(
            "Annotated board rows require clip_path/frame_index or source-video metadata"
        )
    return _load_clip_frame_bgr(row, clip_cache=clip_cache)


def _load_clip_frame_bgr(
    row: PhysicalEvalBoardRow,
    *,
    clip_cache: dict[Path, dict[str, Any]],
) -> np.ndarray:
    clip_path = resolve_project_path(row.clip_path or "")
    clip = clip_cache.get(clip_path)
    if clip is None:
        loaded = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid real clip: {row.clip_path}")
        clip_cache[clip_path] = loaded
        clip = loaded

    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor):
        raise ValueError(f"Clip has no frames tensor: {row.clip_path}")
    frame = frames[int(row.frame_index or 0)]
    return _frame_tensor_to_bgr(frame)


def _frame_tensor_to_bgr(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with 3 dims, got {tuple(frame.shape)}")
    if frame.shape[0] == 3:
        chw = frame
    elif frame.shape[-1] == 3:
        chw = frame.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB frame tensor, got {tuple(frame.shape)}")

    rgb = chw.to(torch.float32)
    if float(rgb.max().item()) <= 1.0:
        rgb = rgb * 255.0
    rgb_uint8 = rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
