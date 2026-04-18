"""Source-video path helpers shared across physical-board tooling."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_source_video_path(source_video_id: str) -> Path:
    video_path = _PROJECT_ROOT / "data" / "videos" / source_video_id / f"{source_video_id}.mp4"
    if not video_path.exists():
        raise ValueError(f"Source video is missing for {source_video_id}: {video_path}")
    return video_path
