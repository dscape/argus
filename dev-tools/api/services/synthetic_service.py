"""Service layer for monitoring synthetic data generation progress."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from api.services import clip_service

# The uvicorn process CWD is /app/dev-tools, but data volumes are
# mounted under /app.  Resolve relative paths from the project root
# so that "data/train" maps to "/app/data/train".
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../dev-tools/api/services/file -> .../dev-tools/api -> .../dev-tools -> /app


def _resolve(directory: str) -> Path:
    """Resolve a directory path relative to the project root."""
    p = Path(directory)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def scan_directory(directory: str, expected_clips: int | None = None) -> dict[str, Any]:
    """Scan a directory for .pt clip files and return progress info.

    Lightweight — uses only os.stat, no torch loading.
    """
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

    pt_files = sorted(dir_path.glob("*.pt"))
    clips = []
    total_size = 0

    for f in pt_files:
        stat = f.stat()
        size = stat.st_size
        total_size += size
        clips.append({
            "filename": f.name,
            "size_mb": round(size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })

    return {
        "directory": str(dir_path),
        "exists": True,
        "expected_clips": expected_clips,
        "clip_count": len(clips),
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "clips": clips,
    }


def get_clip_stats(directory: str) -> dict[str, Any]:
    """Load all .pt clips and compute aggregated stats.

    Heavy operation — loads every clip via torch.load.
    """
    dir_path = _resolve(directory)
    pt_files = sorted(dir_path.glob("*.pt"))

    if not pt_files:
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
    moves_per_clip = []
    legal_moves_sums = []
    total_size = 0
    image_size = None
    clip_length = None

    try:
        from argus.chess.move_vocabulary import NO_MOVE_IDX
    except ImportError:
        NO_MOVE_IDX = 1968

    for f in pt_files:
        total_size += f.stat().st_size
        clip = torch.load(f, map_location="cpu", weights_only=False)

        if "frames" in clip:
            frames = clip["frames"]
            t = frames.shape[0]
            total_frames += t
            if clip_length is None:
                clip_length = t
            if image_size is None and frames.ndim == 4:
                image_size = [int(frames.shape[2]), int(frames.shape[3])]

        if "move_targets" in clip:
            move_targets = clip["move_targets"]
            n_moves = int((move_targets != NO_MOVE_IDX).sum().item())
            total_moves += n_moves
            moves_per_clip.append(n_moves)
        else:
            moves_per_clip.append(0)

        if "legal_masks" in clip:
            avg = float(clip["legal_masks"].float().sum(dim=1).mean().item())
            legal_moves_sums.append(avg)

    n = len(pt_files)
    return {
        "clip_count": n,
        "total_frames": total_frames,
        "avg_frames_per_clip": round(total_frames / n, 1),
        "total_moves": total_moves,
        "avg_moves_per_clip": round(total_moves / n, 1),
        "moves_per_clip_distribution": moves_per_clip,
        "avg_file_size_mb": round(total_size / n / 1024 / 1024, 2),
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "avg_legal_moves": round(sum(legal_moves_sums) / len(legal_moves_sums), 1) if legal_moves_sums else None,
        "image_size": image_size,
        "clip_length": clip_length,
    }


def load_clip_from_path(filepath: str) -> str:
    """Load a .pt file from disk into a clip_service session.

    Returns a session_id usable with /api/clips/{session_id}/* endpoints.
    """
    path = _resolve(filepath)
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"Not a .pt file: {path}")

    file_bytes = path.read_bytes()
    return clip_service.create_session(file_bytes, path.name)
