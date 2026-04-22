"""Source-video path helpers shared across physical-board tooling."""

from __future__ import annotations

from pathlib import Path

from pipeline.paths import find_video_file


def resolve_source_video_path(source_video_id: str) -> Path:
    video_path = find_video_file(source_video_id)
    if video_path is None:
        raise ValueError(f"Source video is missing for {source_video_id}")
    return video_path
