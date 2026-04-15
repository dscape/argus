"""Oblique per-square context crops from the original board frame.

These helpers keep the board in its original perspective image instead of first
warping the whole board to a top-down plane. The crop heuristic intentionally
keeps context above the square because tall pieces project into the row behind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.physical.square_data import INPUT_SIZE, preprocess_square_image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BOARD_GRID_SIZE = 8.0


@dataclass(frozen=True)
class PhysicalObliqueBoardRow:
    annotation_id: str
    clip_path: str
    frame_index: int
    source_video_id: str | None
    corners: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    corner_space: str = "clip_frame"
    clip_frame_size: tuple[int, int] | None = None
    native_corners: tuple[tuple[float, float], ...] | None = None
    native_image_bbox: tuple[int, int, int, int] | None = None
    source_frame_index: int | None = None


class PhysicalAnnotatedObliqueSquareContextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Annotated real boards represented as 64 oblique context crops."""

    def __init__(
        self,
        *,
        annotation_root: str | Path,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalObliqueBoardRow] | None = None,
    ) -> None:
        self.annotation_root = Path(annotation_root)
        self.image_size = image_size
        self.rows = rows or load_annotated_oblique_rows(self.annotation_root)
        self._clip_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        frame_bgr = _load_clip_frame_bgr(row, clip_cache=self._clip_cache)
        crops = extract_oblique_square_context_crops(frame_bgr, row.corners)
        batch = torch.stack(
            [preprocess_square_image(crop, size=self.image_size) for crop in crops],
            dim=0,
        )
        targets = torch.tensor(row.labels, dtype=torch.long)
        return batch, targets


class PhysicalEvalObliqueSquareContextDataset(PhysicalAnnotatedObliqueSquareContextDataset):
    def __init__(
        self,
        *,
        annotation_root: str | Path = _PROJECT_ROOT / "data" / "physical" / "val",
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalObliqueBoardRow] | None = None,
    ) -> None:
        super().__init__(annotation_root=annotation_root, image_size=image_size, rows=rows)


@dataclass(frozen=True)
class PhysicalRealObliqueBoardRow:
    clip_path: str
    frame_index: int
    source_video_id: str | None
    corners: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    source_channel_handle: str | None = None


class PhysicalRealObliqueSquareContextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Replay-derived real boards represented as 64 oblique context crops."""

    def __init__(
        self,
        *,
        rows: list[PhysicalRealObliqueBoardRow],
        image_size: int = INPUT_SIZE,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self._clip_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        frame_bgr = _load_clip_frame_bgr(row, clip_cache=self._clip_cache)
        crops = extract_oblique_square_context_crops(frame_bgr, row.corners)
        batch = torch.stack(
            [preprocess_square_image(crop, size=self.image_size) for crop in crops],
            dim=0,
        )
        targets = torch.tensor(row.labels, dtype=torch.long)
        return batch, targets


def load_annotated_oblique_rows(annotation_root: str | Path) -> list[PhysicalObliqueBoardRow]:
    annotations_path = Path(annotation_root) / "board_annotations.jsonl"
    if not annotations_path.exists():
        raise ValueError(f"Physical board annotations not found: {annotations_path}")

    rows: list[PhysicalObliqueBoardRow] = []
    for line in annotations_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        raw_labels = payload.get("labels")
        raw_corners = payload.get("corners")
        raw_clip_frame_size = payload.get("clip_frame_size")
        raw_native_corners = payload.get("native_corners")
        raw_native_image_bbox = payload.get("native_image_bbox")
        if (
            not isinstance(raw_labels, list)
            or len(raw_labels) != 64
            or any(label is None for label in raw_labels)
            or not isinstance(raw_corners, list)
            or len(raw_corners) != 4
        ):
            continue
        clip_frame_size = None
        if isinstance(raw_clip_frame_size, list) and len(raw_clip_frame_size) == 2:
            clip_frame_size = (int(raw_clip_frame_size[0]), int(raw_clip_frame_size[1]))
        native_corners = None
        if isinstance(raw_native_corners, list) and len(raw_native_corners) == 4:
            native_corners = tuple((float(x), float(y)) for x, y in raw_native_corners)
        native_image_bbox = None
        if isinstance(raw_native_image_bbox, list) and len(raw_native_image_bbox) == 4:
            native_image_bbox = tuple(int(value) for value in raw_native_image_bbox)
        rows.append(
            PhysicalObliqueBoardRow(
                annotation_id=str(payload["annotation_id"]),
                clip_path=str(payload["clip_path"]),
                frame_index=int(payload["frame_index"]),
                source_video_id=(
                    str(payload["source_video_id"])
                    if payload.get("source_video_id") is not None
                    else None
                ),
                corners=tuple((float(x), float(y)) for x, y in raw_corners),
                labels=tuple(int(label) for label in raw_labels),
                corner_space=str(payload.get("corner_space", "clip_frame")),
                clip_frame_size=clip_frame_size,
                native_corners=native_corners,
                native_image_bbox=native_image_bbox,
                source_frame_index=(
                    int(payload["source_frame_index"])
                    if payload.get("source_frame_index") is not None
                    else None
                ),
            )
        )
    rows.sort(key=lambda row: (row.clip_path, row.frame_index, row.annotation_id))
    return rows


def extract_oblique_square_context_crops(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    flip_left_half: bool = True,
    far_rank_top_margin: float = 1.65,
    near_rank_top_margin: float = 0.65,
    bottom_margin: float = 0.10,
    center_file_side_margin: float = 0.20,
    edge_file_side_margin: float = 0.45,
) -> list[np.ndarray]:
    """Extract 64 chesscog-style context crops from the original board frame.

    The heuristic is intentionally simple but explicit:
    - project each square back into the original image via the board homography
    - extend the crop upward more for far ranks than for near ranks
    - extend the crop width more for edge files than for center files
    - horizontally flip left-half crops so lateral context is more consistent
    """

    transform = board_to_image_transform(corners)
    height, width = image_bgr.shape[:2]
    crops: list[np.ndarray] = []

    for square_index in range(64):
        row = square_index // 8
        col = square_index % 8
        quad = project_square_quad(transform, row=row, col=col)
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]
        bbox_width = max(float(x_coords.max() - x_coords.min()), 1.0)
        bbox_height = max(float(y_coords.max() - y_coords.min()), 1.0)

        depth = 1.0 - (row / 7.0)
        side_position = abs(col - 3.5) / 3.5
        top_margin = _lerp(near_rank_top_margin, far_rank_top_margin, depth) * bbox_height
        side_margin = (
            _lerp(center_file_side_margin, edge_file_side_margin, side_position) * bbox_width
        )
        extra_bottom = bottom_margin * bbox_height

        x1 = int(np.floor(x_coords.min() - side_margin))
        x2 = int(np.ceil(x_coords.max() + side_margin))
        y1 = int(np.floor(y_coords.min() - top_margin))
        y2 = int(np.ceil(y_coords.max() + extra_bottom))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, max(x1 + 1, x2))
        y2 = min(height, max(y1 + 1, y2))
        crop = image_bgr[y1:y2, x1:x2].copy()
        if flip_left_half and col < 4:
            crop = cv2.flip(crop, 1)
        crops.append(crop)

    return crops


def board_to_image_transform(
    corners: tuple[tuple[float, float], ...] | list[list[float]],
) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {points.shape}")
    board_points = np.array(
        [
            [0.0, 0.0],
            [_BOARD_GRID_SIZE, 0.0],
            [_BOARD_GRID_SIZE, _BOARD_GRID_SIZE],
            [0.0, _BOARD_GRID_SIZE],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(board_points, points)


def project_square_quad(transform: np.ndarray, *, row: int, col: int) -> np.ndarray:
    board_quad = np.array(
        [
            [float(col), float(row)],
            [float(col + 1), float(row)],
            [float(col + 1), float(row + 1)],
            [float(col), float(row + 1)],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    return cv2.perspectiveTransform(board_quad, transform).reshape(4, 2)


def _load_clip_frame_bgr(
    row: PhysicalObliqueBoardRow | PhysicalRealObliqueBoardRow,
    *,
    clip_cache: dict[Path, dict[str, Any]],
) -> np.ndarray:
    clip_path = (_PROJECT_ROOT / row.clip_path).resolve()
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
    frame = frames[row.frame_index]
    rgb = _frame_tensor_to_rgb(frame)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _frame_tensor_to_rgb(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with 3 dims, got {tuple(frame.shape)}")
    if frame.shape[0] == 3:
        chw = frame
    elif frame.shape[-1] == 3:
        chw = frame.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB frame tensor, got {tuple(frame.shape)}")

    if chw.dtype == torch.uint8:
        array = chw.permute(1, 2, 0).cpu().numpy()
        return array.astype(np.uint8)

    rgb = chw.to(torch.float32)
    if float(rgb.max().item()) <= 1.0:
        rgb = rgb * 255.0
    array = rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy()
    return array.astype(np.uint8)


def _lerp(start: float, end: float, amount: float) -> float:
    return float(start + (end - start) * amount)
