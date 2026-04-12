"""Service layer for building the held-out physical square evaluation set."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from api.services.data import clip_service
from pipeline.physical import eval_dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_REAL_CLIP_RE = re.compile(r"^clip_overlay_(?P<video_id>.+?)_clip(?P<clip_id>\d+)_\d+\.pt$")


def list_clip_files(
    clips_dir: str = "data/argus/train_real",
    *,
    limit: int = 200,
) -> dict[str, Any]:
    directory = _resolve_within_project(clips_dir)
    if not directory.exists():
        return {"clips_dir": str(directory.relative_to(_PROJECT_ROOT)), "clips": []}

    clips: list[dict[str, Any]] = []
    for path in sorted(directory.glob("clip_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        stat = path.stat()
        match = _REAL_CLIP_RE.match(path.name)
        clips.append(
            {
                "filename": path.name,
                "clip_path": str(path.relative_to(_PROJECT_ROOT)),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "source_video_id": match.group("video_id") if match else None,
                "clip_id": int(match.group("clip_id")) if match else None,
            }
        )

    return {
        "clips_dir": str(directory.relative_to(_PROJECT_ROOT)),
        "clips": clips,
    }


def get_annotation_summary() -> dict[str, Any]:
    return eval_dataset.get_annotation_summary()


def get_frame_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    resolved = _resolve_within_project(clip_path)
    return eval_dataset.load_board_annotation(str(resolved.relative_to(_PROJECT_ROOT)), frame_index)


def rectify_frame(
    session_id: str,
    frame_index: int,
    corners: list[list[float]],
    *,
    output_size: int = eval_dataset.DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    image_rgb = _get_clip_frame_rgb(session_id, frame_index)
    rectified_rgb = eval_dataset.rectify_board_image(
        image_rgb,
        corners,
        output_size=output_size,
    )
    return {
        "image_b64": _encode_rgb_png(rectified_rgb),
        "output_size": output_size,
    }


def save_annotation(
    session_id: str,
    clip_path: str,
    frame_index: int,
    corners: list[list[float]],
    labels: list[int | None],
    *,
    output_size: int = eval_dataset.DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    resolved_clip_path = _resolve_within_project(clip_path)
    image_rgb = _get_clip_frame_rgb(session_id, frame_index)
    source_video_id = _source_video_id_from_path(resolved_clip_path)
    annotation = eval_dataset.save_board_annotation(
        image_rgb,
        clip_path=str(resolved_clip_path.relative_to(_PROJECT_ROOT)),
        frame_index=frame_index,
        source_video_id=source_video_id,
        corners=corners,
        labels=labels,
        output_size=output_size,
    )
    return {
        "annotation": annotation,
        "summary": eval_dataset.get_annotation_summary(),
    }


def _resolve_within_project(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (_PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"Path is outside the project root: {path}")
    return candidate


def _get_clip_frame_rgb(session_id: str, frame_index: int) -> np.ndarray:
    session = clip_service.get_session(session_id)
    if session is None:
        raise ValueError("Clip session not found")

    clip = session["clip"]
    frames = clip.get("frames")
    if frames is None:
        raise ValueError("No frames in clip")
    if frame_index < 0 or frame_index >= frames.shape[0]:
        raise ValueError(f"Frame index {frame_index} out of range [0, {frames.shape[0]})")

    frame = frames[frame_index]
    if frame.dtype == torch.uint8:
        image_rgb = frame.permute(1, 2, 0).numpy()
    else:
        image_rgb = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return image_rgb


def _encode_rgb_png(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".png", image_bgr)
    if not encoded:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _source_video_id_from_path(path: Path) -> str | None:
    match = _REAL_CLIP_RE.match(path.name)
    return match.group("video_id") if match else None
