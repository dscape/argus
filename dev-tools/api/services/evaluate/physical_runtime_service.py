"""Service layer for visualizing the physical runtime reader on held-out eval clips."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image
from pipeline.physical.board_data import PhysicalEvalBoardDataset
from pipeline.physical.runtime_visualization import (
    _collect_visualized_frames,
    _group_rows_by_clip,
    _select_clip_path,
    render_contact_sheet,
    render_visualized_runtime_frame,
)


def render_runtime_visualization(
    *,
    clip_path: str | None,
    frame_start: int,
    frame_count: int,
    panel_size: int = 240,
    device: str = "cpu",
) -> dict[str, Any]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be > 0, got {panel_size}")

    dataset = PhysicalEvalBoardDataset()
    rows_by_clip = _group_rows_by_clip(dataset.rows)
    selected_clip_path = _select_clip_path(rows_by_clip, clip_path=clip_path)
    clip_rows = rows_by_clip[selected_clip_path]
    visualized_frames = _collect_visualized_frames(
        clip_rows,
        frame_start=frame_start,
        frame_count=frame_count,
        device=device,
    )

    frame_images = [
        render_visualized_runtime_frame(frame, panel_size=panel_size)
        for frame in visualized_frames
    ]
    contact_sheet = render_contact_sheet(
        frame_images,
        clip_path=selected_clip_path,
        frame_start=frame_start,
        frame_count=len(visualized_frames),
    )

    return {
        "clip_path": selected_clip_path,
        "frame_start": frame_start,
        "frame_count": len(visualized_frames),
        "available_frame_count": len(clip_rows),
        "contact_sheet_b64": _image_to_base64(contact_sheet),
        "frames": [
            {
                "annotation_id": frame.annotation_id,
                "board_path": frame.board_path,
                "frame_index": frame.frame_index,
                "gt_change_count": frame.gt_change_count,
                "stateless_change_count": frame.stateless_change_count,
                "stateless_error_count": frame.stateless_error_count,
                "stateless_mean_confidence": round(frame.stateless_mean_confidence, 4),
                "temporal_change_count": frame.temporal_change_count,
                "temporal_error_count": frame.temporal_error_count,
                "temporal_mean_confidence": round(frame.temporal_mean_confidence, 4),
                "image_b64": _image_to_base64(frame_image),
            }
            for frame, frame_image in zip(visualized_frames, frame_images)
        ],
    }


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
