"""Annotation-row loaders shared across physical-board tooling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


@dataclass(frozen=True)
class PhysicalRealObliqueBoardRow:
    clip_path: str
    frame_index: int
    source_video_id: str | None
    corners: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    source_channel_handle: str | None = None


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


__all__ = [
    "PhysicalObliqueBoardRow",
    "PhysicalRealObliqueBoardRow",
    "_load_clip_frame_bgr",
    "load_annotated_oblique_rows",
]
