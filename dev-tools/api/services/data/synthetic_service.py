"""Helpers for scanning and inspecting generated synthetic clip datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from api.services.data import clip_service

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve(directory: str) -> Path:
    path = Path(directory)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def scan_directory(directory: str, expected_clips: int | None = None) -> dict[str, Any]:
    """Return lightweight filesystem stats for a synthetic clip directory."""
    dir_path = _resolve(directory)

    if not dir_path.exists():
        return {
            "directory": str(dir_path),
            "exists": False,
            "expected_clips": expected_clips,
            "clip_count": 0,
            "total_size_mb": 0.0,
            "clips": [],
        }

    clips: list[dict[str, Any]] = []
    total_size = 0

    for file_path in sorted(dir_path.glob("*.pt")):
        stat = file_path.stat()
        total_size += stat.st_size
        clips.append(
            {
                "filename": file_path.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )

    return {
        "directory": str(dir_path),
        "exists": True,
        "expected_clips": expected_clips,
        "clip_count": len(clips),
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "clips": clips,
    }


def get_clip_stats(directory: str) -> dict[str, Any]:
    """Compute aggregated stats by loading all clips in a directory."""
    dir_path = _resolve(directory)
    clip_files = sorted(dir_path.glob("*.pt"))

    if not clip_files:
        return {
            "clip_count": 0,
            "total_frames": 0,
            "avg_frames_per_clip": 0.0,
            "total_moves": 0,
            "avg_moves_per_clip": 0.0,
            "moves_per_clip_distribution": [],
            "avg_file_size_mb": 0.0,
            "total_size_mb": 0.0,
            "avg_legal_moves": None,
            "image_size": None,
            "clip_length": None,
        }

    total_frames = 0
    total_moves = 0
    moves_per_clip: list[int] = []
    legal_move_averages: list[float] = []
    total_size = 0
    image_size: list[int] | None = None
    clip_length: int | None = None

    try:
        from argus.chess.move_vocabulary import NO_MOVE_IDX
    except ImportError:
        NO_MOVE_IDX = 1968

    for file_path in clip_files:
        total_size += file_path.stat().st_size
        clip = torch.load(file_path, map_location="cpu", weights_only=True)

        if "frames" in clip:
            frames = clip["frames"]
            total_frames += frames.shape[0]
            if clip_length is None:
                clip_length = int(frames.shape[0])
            if image_size is None and frames.ndim == 4:
                image_size = [int(frames.shape[2]), int(frames.shape[3])]

        if "move_targets" in clip:
            move_targets = clip["move_targets"]
            move_count = int((move_targets != NO_MOVE_IDX).sum().item())
            total_moves += move_count
            moves_per_clip.append(move_count)
        else:
            moves_per_clip.append(0)

        if "legal_masks" in clip:
            average_legal_moves = float(
                clip["legal_masks"].float().sum(dim=1).mean().item()
            )
            legal_move_averages.append(average_legal_moves)

    clip_count = len(clip_files)
    avg_legal_moves = None
    if legal_move_averages:
        avg_legal_moves = round(
            sum(legal_move_averages) / len(legal_move_averages),
            1,
        )

    return {
        "clip_count": clip_count,
        "total_frames": total_frames,
        "avg_frames_per_clip": round(total_frames / clip_count, 1),
        "total_moves": total_moves,
        "avg_moves_per_clip": round(total_moves / clip_count, 1),
        "moves_per_clip_distribution": moves_per_clip,
        "avg_file_size_mb": round(total_size / clip_count / 1024 / 1024, 2),
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "avg_legal_moves": avg_legal_moves,
        "image_size": image_size,
        "clip_length": clip_length,
    }


def load_clip_from_path(filepath: str) -> str:
    """Load a clip from disk into a reusable clip inspection session."""
    path = _resolve(filepath)
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"Not a .pt file: {path}")

    return clip_service.create_session(path.read_bytes(), path.name)
