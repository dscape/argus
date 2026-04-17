"""Per-square datasets for the two-stage classifier.

Each sample is one square crop and its label, so the loader can shuffle and
rebalance across squares (not just across boards). Crops are computed on the
fly from the original oblique frame using the board homography stored in the
annotation. Nothing on disk needs to be regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.physical.oblique_square_context import (
    PhysicalObliqueBoardRow,
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.square_classifiers import (
    OCCUPANCY_NUM_CLASSES,
    PIECE_NUM_CLASSES,
    square_class_to_occupancy_label,
    square_class_to_piece_label,
)
from pipeline.physical.square_crop import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    extract_occupancy_crop,
    extract_piece_crop,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class SquareSampleIndex:
    """Lightweight pointer: (board row index, 0..63 square index, square class)."""

    row_index: int
    square_index: int
    square_class: int


def preprocess_square_crop(
    crop_bgr: np.ndarray,
    *,
    size: int,
) -> torch.Tensor:
    """BGR crop → normalized CHW float tensor at the requested size."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


class _SquareSampleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Shared plumbing for occupancy and piece per-square datasets."""

    def __init__(
        self,
        *,
        rows: list[PhysicalObliqueBoardRow],
        indices: list[SquareSampleIndex],
        input_size: int,
    ) -> None:
        self.rows = rows
        self.indices = indices
        self.input_size = input_size
        self._clip_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _load_frame_bgr(self, row_index: int) -> np.ndarray:
        row = self.rows[row_index]
        return _load_clip_frame_bgr(row, clip_cache=self._clip_cache)

    def _row_corners(
        self, row_index: int
    ) -> tuple[tuple[float, float], ...]:
        return self.rows[row_index].corners


class OccupancySquareDataset(_SquareSampleDataset):
    """One symmetric crop + binary occupied-vs-empty label per sample."""

    def __init__(
        self,
        *,
        rows: list[PhysicalObliqueBoardRow],
        input_size: int = DEFAULT_OCCUPANCY_CROP_SIZE,
    ) -> None:
        indices = [
            SquareSampleIndex(
                row_index=row_index,
                square_index=square_index,
                square_class=int(row.labels[square_index]),
            )
            for row_index, row in enumerate(rows)
            for square_index in range(64)
        ]
        super().__init__(rows=rows, indices=indices, input_size=input_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.indices[index]
        frame_bgr = self._load_frame_bgr(sample.row_index)
        row = sample.square_index // 8
        col = sample.square_index % 8
        crop_bgr = extract_occupancy_crop(
            frame_bgr,
            self._row_corners(sample.row_index),
            row=row,
            col=col,
            output_size=self.input_size,
        )
        image = preprocess_square_crop(crop_bgr, size=self.input_size)
        label = torch.tensor(
            square_class_to_occupancy_label(sample.square_class),
            dtype=torch.long,
        )
        return image, label


class PieceSquareDataset(_SquareSampleDataset):
    """One asymmetric crop + 12-class piece label per sample.

    Empty squares (``square_class == 0``) are skipped because they carry no
    piece signal; the occupancy classifier decides empty vs occupied.
    """

    def __init__(
        self,
        *,
        rows: list[PhysicalObliqueBoardRow],
        input_size: int = DEFAULT_PIECE_CROP_SIZE,
    ) -> None:
        indices = [
            SquareSampleIndex(
                row_index=row_index,
                square_index=square_index,
                square_class=int(row.labels[square_index]),
            )
            for row_index, row in enumerate(rows)
            for square_index in range(64)
            if int(row.labels[square_index]) != 0
        ]
        super().__init__(rows=rows, indices=indices, input_size=input_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.indices[index]
        frame_bgr = self._load_frame_bgr(sample.row_index)
        row = sample.square_index // 8
        col = sample.square_index % 8
        crop_bgr = extract_piece_crop(
            frame_bgr,
            self._row_corners(sample.row_index),
            row=row,
            col=col,
            output_size=self.input_size,
        )
        image = preprocess_square_crop(crop_bgr, size=self.input_size)
        label = torch.tensor(
            square_class_to_piece_label(sample.square_class),
            dtype=torch.long,
        )
        return image, label


def load_occupancy_dataset(
    annotation_root: str | Path,
    *,
    input_size: int = DEFAULT_OCCUPANCY_CROP_SIZE,
) -> OccupancySquareDataset:
    rows = load_annotated_oblique_rows(annotation_root)
    return OccupancySquareDataset(rows=rows, input_size=input_size)


def load_piece_dataset(
    annotation_root: str | Path,
    *,
    input_size: int = DEFAULT_PIECE_CROP_SIZE,
) -> PieceSquareDataset:
    rows = load_annotated_oblique_rows(annotation_root)
    return PieceSquareDataset(rows=rows, input_size=input_size)


def class_counts(dataset: _SquareSampleDataset) -> dict[str, int]:
    """Return per-class sample counts for logging (uses SQUARE_CLASS_NAMES)."""
    counts: dict[str, int] = {name: 0 for name in SQUARE_CLASS_NAMES}
    for sample in dataset.indices:
        counts[SQUARE_CLASS_NAMES[sample.square_class]] += 1
    return counts


__all__ = [
    "OCCUPANCY_NUM_CLASSES",
    "OccupancySquareDataset",
    "PIECE_NUM_CLASSES",
    "PieceSquareDataset",
    "SquareSampleIndex",
    "class_counts",
    "load_occupancy_dataset",
    "load_piece_dataset",
    "preprocess_square_crop",
]
