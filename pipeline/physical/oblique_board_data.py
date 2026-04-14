"""Whole-board oblique crops with geometry for physical board probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.physical.board_data import INPUT_SIZE, normalize_rgb_tensor
from pipeline.physical.oblique_square_context import (
    PhysicalObliqueBoardRow,
    PhysicalRealObliqueBoardRow,
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"


@dataclass(frozen=True)
class ObliqueBoardCrop:
    image_bgr: np.ndarray
    corners: np.ndarray


class PhysicalObliqueBoardCropOnlyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Drop geometry after oblique crop extraction while preserving row metadata."""

    def __init__(
        self,
        base_dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        self.base_dataset = base_dataset
        self.rows = getattr(base_dataset, "rows", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, labels, _corners = self.base_dataset[index]
        return image, labels


class PhysicalSyntheticObliqueBoardDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Synthetic whole-board images paired with full-frame board corners."""

    def __init__(self, base_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, labels = self.base_dataset[index]
        size = image.shape[-1]
        corners = torch.tensor(
            [
                [0.0, 0.0],
                [float(size - 1), 0.0],
                [float(size - 1), float(size - 1)],
                [0.0, float(size - 1)],
            ],
            dtype=torch.float32,
        )
        return image, labels, corners


class PhysicalAnnotatedObliqueBoardDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Annotated real boards represented as one oblique crop plus board corners."""

    def __init__(
        self,
        *,
        annotation_root: str | Path,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalObliqueBoardRow] | None = None,
        crop_margin: float = 0.18,
    ) -> None:
        self.annotation_root = Path(annotation_root)
        self.image_size = image_size
        self.crop_margin = crop_margin
        self.rows = rows or load_annotated_oblique_rows(self.annotation_root)
        self._clip_cache: dict[Path, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        frame_bgr = _load_clip_frame_bgr(row, clip_cache=self._clip_cache)
        image, corners = preprocess_oblique_board_image(
            frame_bgr,
            row.corners,
            size=self.image_size,
            crop_margin=self.crop_margin,
        )
        labels = torch.tensor(row.labels, dtype=torch.long)
        return image, labels, corners


class PhysicalEvalObliqueBoardDataset(PhysicalAnnotatedObliqueBoardDataset):
    def __init__(
        self,
        *,
        annotation_root: str | Path = _DEFAULT_VAL_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalObliqueBoardRow] | None = None,
        crop_margin: float = 0.18,
    ) -> None:
        super().__init__(
            annotation_root=annotation_root,
            image_size=image_size,
            rows=rows,
            crop_margin=crop_margin,
        )


class PhysicalRealObliqueBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Replay-derived real boards represented as one oblique crop plus board corners."""

    def __init__(
        self,
        *,
        rows: list[PhysicalRealObliqueBoardRow],
        image_size: int = INPUT_SIZE,
        crop_margin: float = 0.18,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.crop_margin = crop_margin
        self._clip_cache: dict[Path, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        frame_bgr = _load_clip_frame_bgr(row, clip_cache=self._clip_cache)
        image, corners = preprocess_oblique_board_image(
            frame_bgr,
            row.corners,
            size=self.image_size,
            crop_margin=self.crop_margin,
        )
        labels = torch.tensor(row.labels, dtype=torch.long)
        return image, labels, corners


def preprocess_oblique_board_image(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    size: int = INPUT_SIZE,
    crop_margin: float = 0.18,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop around an oblique board, resize to a square tensor, and scale corners."""
    crop = extract_oblique_board_crop(image_bgr, corners, crop_margin=crop_margin)
    rgb = cv2.cvtColor(crop.image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    height, width = crop.image_bgr.shape[:2]
    scaled_corners = crop.corners.copy()
    scaled_corners[:, 0] *= float(size) / max(float(width), 1.0)
    scaled_corners[:, 1] *= float(size) / max(float(height), 1.0)
    corners_tensor = torch.from_numpy(scaled_corners.astype(np.float32))
    return normalize_rgb_tensor(image_tensor), corners_tensor


def extract_oblique_board_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    crop_margin: float = 0.18,
) -> ObliqueBoardCrop:
    """Crop a full frame down to the board neighborhood and return relative corners."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {points.shape}")
    if crop_margin < 0.0:
        raise ValueError(f"crop_margin must be >= 0, got {crop_margin}")

    height, width = image_bgr.shape[:2]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    extent = np.maximum(max_xy - min_xy, 1.0)
    margin = extent * float(crop_margin)

    x1 = max(0, int(np.floor(min_xy[0] - margin[0])))
    y1 = max(0, int(np.floor(min_xy[1] - margin[1])))
    x2 = min(width, int(np.ceil(max_xy[0] + margin[0])))
    y2 = min(height, int(np.ceil(max_xy[1] + margin[1])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid board crop derived from corners")

    cropped = image_bgr[y1:y2, x1:x2].copy()
    relative_corners = points - np.array([x1, y1], dtype=np.float32)
    return ObliqueBoardCrop(image_bgr=cropped, corners=relative_corners)
