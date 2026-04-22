"""Per-square datasets for the two-stage classifier.

Samples are individual (board, square) tuples, so the loader can shuffle and
rebalance across squares. Crops use the projection-based extractor
(``extract_projected_piece_crop`` / ``extract_projected_occupancy_crop``) which
recovers the camera pose from the 4 board corners and projects the piece's 3D
bounding box to image space. This adapts to each frame's actual camera angle.

Native-resolution video frames are used when the annotation provides
``source_video_id`` + ``source_frame_index`` + ``native_image_bbox`` (the
standard for physical-tracking data). A small LRU cache amortizes the video
reads across the 64 samples per board.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from pipeline.physical.piece_projection import (
    DEFAULT_PIECE_HEIGHT,
    extract_projected_occupancy_crop,
    extract_projected_piece_crop,
)
from pipeline.physical.shared.annotation_rows import (
    PhysicalObliqueBoardRow,
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.shared.source_video_paths import resolve_source_video_path
from pipeline.physical.two_stage.classifiers import (
    OCCUPANCY_NUM_CLASSES,
    PIECE_NUM_CLASSES,
    square_class_to_occupancy_label,
    square_class_to_piece_label,
)
from pipeline.shared import SQUARE_CLASS_NAMES, fen_to_square_labels

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OCCUPANCY_CROP_SIZE = 112
DEFAULT_PIECE_CROP_SIZE = 224
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# A full 1920x1080 BGR frame is ~6 MB. Cache capacity of 2000 holds every frame
# in the physical splits (~1500 unique annotations total), so epoch 2+ incurs
# zero video reads. Drop to ~16 for memory-constrained environments.
_NATIVE_FRAME_CACHE_CAPACITY = 2000


@dataclass(frozen=True)
class SquareSampleIndex:
    """Lightweight pointer: (board row index, 0..63 square index, square class)."""

    row_index: int
    square_index: int
    square_class: int


_AUGMENTATION_PIPELINE = T.Compose(
    [
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0),
    ]
)


def preprocess_square_crop(
    crop_bgr: np.ndarray,
    *,
    size: int,
    augment: bool = False,
) -> torch.Tensor:
    """BGR crop -> normalized CHW float tensor at the requested size.

    When ``augment=True`` the crop is passed through a standard color/affine
    augmentation pipeline in [0, 1] space before ImageNet normalization.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    if augment:
        tensor = _AUGMENTATION_PIPELINE(tensor)
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


class NativeFrameLoader:
    """Load native-resolution video frames with an LRU cache.

    Frames are cached by ``(video_path, source_frame_index)``. Video captures
    are held open until ``close()`` to avoid the cost of re-opening files.
    Access is serialized because OpenCV `VideoCapture` is not thread-safe.
    """

    def __init__(self, *, capacity: int = _NATIVE_FRAME_CACHE_CAPACITY) -> None:
        self._capacity = capacity
        self._frames: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            for capture in self._captures.values():
                capture.release()
            self._captures.clear()
            self._frames.clear()

    def __del__(self) -> None:
        self.close()

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
                raise ValueError(
                    f"Failed to read frame {source_frame_index} from {video_path}"
                )
            cached_frame = frame.copy()
            self._frames[key] = cached_frame
            if len(self._frames) > self._capacity:
                self._frames.popitem(last=False)
            return cached_frame.copy()


def _row_has_native_metadata(row: PhysicalObliqueBoardRow) -> bool:
    return (
        row.source_video_id is not None
        and row.source_frame_index is not None
        and row.native_corners is not None
        and row.native_image_bbox is not None
    )


def _full_frame_corners(row: PhysicalObliqueBoardRow) -> tuple[tuple[float, float], ...]:
    if row.native_corners is None or row.native_image_bbox is None:
        raise ValueError(
            f"annotation {row.annotation_id} missing native_corners/native_image_bbox; "
            "cannot use projected crops without native-resolution metadata"
        )
    x_off, y_off, _, _ = row.native_image_bbox
    return tuple((float(c[0] + x_off), float(c[1] + y_off)) for c in row.native_corners)


def _path_for_storage(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _frame_tensor_hw(frames: torch.Tensor) -> tuple[int, int]:
    if frames.ndim < 4:
        raise ValueError(f"Expected frames tensor with at least 4 dims, got {tuple(frames.shape)}")
    frame = frames[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with 3 dims, got {tuple(frame.shape)}")
    if frame.shape[0] == 3:
        return int(frame.shape[1]), int(frame.shape[2])
    if frame.shape[-1] == 3:
        return int(frame.shape[0]), int(frame.shape[1])
    raise ValueError(f"Expected RGB frame tensor, got {tuple(frame.shape)}")


def _flatten_fen_labels(fen: str) -> tuple[int, ...]:
    return tuple(int(label) for row in fen_to_square_labels(fen) for label in row)


def load_synthetic_oblique_rows(clips_root: str | Path) -> list[PhysicalObliqueBoardRow]:
    clips_root_path = Path(clips_root)
    if not clips_root_path.is_absolute():
        clips_root_path = (_PROJECT_ROOT / clips_root_path).resolve()
    if not clips_root_path.exists():
        raise ValueError(f"Synthetic clips directory not found: {clips_root_path}")

    clip_paths = sorted(clips_root_path.glob("clip_*.pt"))
    if not clip_paths:
        raise ValueError(f"No synthetic clips found in {clips_root_path}")

    rows: list[PhysicalObliqueBoardRow] = []
    for clip_path in clip_paths:
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip, dict):
            raise ValueError(f"Invalid synthetic clip payload: {clip_path}")

        frames = clip.get("frames")
        fens = clip.get("fens")
        board_corners = clip.get("board_corners")
        if not isinstance(frames, torch.Tensor):
            raise ValueError(f"Synthetic clip has no frames tensor: {clip_path}")
        if not isinstance(fens, list):
            raise ValueError(f"Synthetic clip has no fens list: {clip_path}")
        if board_corners is None:
            raise ValueError(
                f"Synthetic clip is missing board_corners: {clip_path}. Regenerate the clips first."
            )

        corners_tensor = (
            board_corners.to(dtype=torch.float32)
            if isinstance(board_corners, torch.Tensor)
            else torch.as_tensor(board_corners, dtype=torch.float32)
        )
        if corners_tensor.ndim != 3 or tuple(corners_tensor.shape[1:]) != (4, 2):
            raise ValueError(
                f"Synthetic clip board_corners must have shape (T, 4, 2): {clip_path}"
            )

        frame_height, frame_width = _frame_tensor_hw(frames)
        frame_count = min(int(frames.shape[0]), len(fens), int(corners_tensor.shape[0]))
        clip_path_str = _path_for_storage(clip_path)
        for frame_index in range(frame_count):
            fen = fens[frame_index]
            if not isinstance(fen, str) or not fen.strip():
                raise ValueError(
                    f"Synthetic clip has invalid FEN at frame {frame_index}: {clip_path}"
                )
            corners = corners_tensor[frame_index].tolist()
            rows.append(
                PhysicalObliqueBoardRow(
                    annotation_id=f"{clip_path.stem}_frame{frame_index:04d}",
                    clip_path=clip_path_str,
                    frame_index=frame_index,
                    source_video_id=None,
                    corners=tuple((float(x), float(y)) for x, y in corners),
                    labels=_flatten_fen_labels(fen),
                    corner_space="clip_frame",
                    clip_frame_size=(frame_width, frame_height),
                    native_corners=None,
                    native_image_bbox=None,
                    source_frame_index=None,
                )
            )
    return rows


class _SquareSampleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Shared plumbing for occupancy and piece per-square datasets."""

    def __init__(
        self,
        *,
        rows: list[PhysicalObliqueBoardRow],
        indices: list[SquareSampleIndex],
        input_size: int,
        use_native_frames: bool,
        piece_height: float,
        augment: bool = False,
        allow_clip_fallback: bool = False,
    ) -> None:
        self.rows = rows
        self.indices = indices
        self.input_size = input_size
        self.use_native_frames = use_native_frames
        self.piece_height = piece_height
        self.augment = augment
        self.allow_clip_fallback = allow_clip_fallback
        self._clip_cache: dict[Path, dict[str, Any]] = {}
        self._frame_loader = NativeFrameLoader() if use_native_frames else None

    def __len__(self) -> int:
        return len(self.indices)

    def close(self) -> None:
        if self._frame_loader is not None:
            self._frame_loader.close()

    def _frame_and_corners(
        self, row_index: int
    ) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
        row = self.rows[row_index]
        if (
            self.use_native_frames
            and self._frame_loader is not None
            and _row_has_native_metadata(row)
        ):
            frame = self._frame_loader.load(
                source_video_id=str(row.source_video_id),
                source_frame_index=int(row.source_frame_index),
            )
            corners = _full_frame_corners(row)
            return frame, corners
        frame = _load_clip_frame_bgr(row, clip_cache=self._clip_cache)
        return frame, row.corners


class OccupancySquareDataset(_SquareSampleDataset):
    """One projected occupancy crop + binary occupied-vs-empty label per sample."""

    def __init__(
        self,
        *,
        rows: list[PhysicalObliqueBoardRow],
        input_size: int,
        use_native_frames: bool = True,
        piece_height: float = DEFAULT_PIECE_HEIGHT,
        augment: bool = False,
        pad_ratio: float = 0.3,
        allow_clip_fallback: bool = False,
    ) -> None:
        filtered = (
            rows
            if (not use_native_frames or allow_clip_fallback)
            else _rows_with_native_metadata(rows)
        )
        indices = [
            SquareSampleIndex(
                row_index=row_index,
                square_index=square_index,
                square_class=int(row.labels[square_index]),
            )
            for row_index, row in enumerate(filtered)
            for square_index in range(64)
        ]
        super().__init__(
            rows=filtered,
            indices=indices,
            input_size=input_size,
            use_native_frames=use_native_frames,
            piece_height=piece_height,
            augment=augment,
            allow_clip_fallback=allow_clip_fallback,
        )
        self.pad_ratio = pad_ratio

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.indices[index]
        frame_bgr, corners = self._frame_and_corners(sample.row_index)
        row = sample.square_index // 8
        col = sample.square_index % 8
        crop_bgr = extract_projected_occupancy_crop(
            frame_bgr,
            corners,
            row=row,
            col=col,
            output_size=self.input_size,
            pad_ratio=self.pad_ratio,
        )
        image = preprocess_square_crop(crop_bgr, size=self.input_size, augment=self.augment)
        label = torch.tensor(
            square_class_to_occupancy_label(sample.square_class),
            dtype=torch.long,
        )
        return image, label


class PieceSquareDataset(_SquareSampleDataset):
    """One projected piece crop + 12-class piece label per sample.

    Empty squares are skipped -- the occupancy classifier decides empty vs
    occupied.
    """

    def __init__(
        self,
        *,
        rows: list[PhysicalObliqueBoardRow],
        input_size: int,
        use_native_frames: bool = True,
        piece_height: float = DEFAULT_PIECE_HEIGHT,
        flip_left_half: bool = True,
        augment: bool = False,
        allow_clip_fallback: bool = False,
    ) -> None:
        filtered = (
            rows
            if (not use_native_frames or allow_clip_fallback)
            else _rows_with_native_metadata(rows)
        )
        indices = [
            SquareSampleIndex(
                row_index=row_index,
                square_index=square_index,
                square_class=int(row.labels[square_index]),
            )
            for row_index, row in enumerate(filtered)
            for square_index in range(64)
            if int(row.labels[square_index]) != 0
        ]
        super().__init__(
            rows=filtered,
            indices=indices,
            input_size=input_size,
            use_native_frames=use_native_frames,
            piece_height=piece_height,
            augment=augment,
            allow_clip_fallback=allow_clip_fallback,
        )
        self.flip_left_half = flip_left_half

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.indices[index]
        frame_bgr, corners = self._frame_and_corners(sample.row_index)
        row = sample.square_index // 8
        col = sample.square_index % 8
        crop_bgr = extract_projected_piece_crop(
            frame_bgr,
            corners,
            row=row,
            col=col,
            output_size=self.input_size,
            piece_height=self.piece_height,
            flip_left_half=self.flip_left_half,
        )
        image = preprocess_square_crop(crop_bgr, size=self.input_size, augment=self.augment)
        label = torch.tensor(
            square_class_to_piece_label(sample.square_class),
            dtype=torch.long,
        )
        return image, label


def _rows_with_native_metadata(
    rows: list[PhysicalObliqueBoardRow],
) -> list[PhysicalObliqueBoardRow]:
    return [row for row in rows if _row_has_native_metadata(row)]


def load_occupancy_dataset(
    annotation_root: str | Path,
    *,
    input_size: int,
    use_native_frames: bool = True,
) -> OccupancySquareDataset:
    rows = load_annotated_oblique_rows(annotation_root)
    return OccupancySquareDataset(
        rows=rows, input_size=input_size, use_native_frames=use_native_frames
    )


def load_piece_dataset(
    annotation_root: str | Path,
    *,
    input_size: int,
    use_native_frames: bool = True,
) -> PieceSquareDataset:
    rows = load_annotated_oblique_rows(annotation_root)
    return PieceSquareDataset(
        rows=rows, input_size=input_size, use_native_frames=use_native_frames
    )


def class_counts(dataset: _SquareSampleDataset) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in SQUARE_CLASS_NAMES}
    for sample in dataset.indices:
        counts[SQUARE_CLASS_NAMES[sample.square_class]] += 1
    return counts


__all__ = [
    "DEFAULT_OCCUPANCY_CROP_SIZE",
    "DEFAULT_PIECE_CROP_SIZE",
    "NativeFrameLoader",
    "OCCUPANCY_NUM_CLASSES",
    "OccupancySquareDataset",
    "PIECE_NUM_CLASSES",
    "PieceSquareDataset",
    "SquareSampleIndex",
    "class_counts",
    "load_occupancy_dataset",
    "load_piece_dataset",
    "load_synthetic_oblique_rows",
    "preprocess_square_crop",
]
