"""Prepare train/val datasets from generated real-video training clips."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def export_training_dataset(
    clips_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    """Export a video-disjoint train/val split from generated clip files."""
    clips_root = Path(clips_dir)
    output_root = Path(output_dir)
    clip_paths = sorted(clips_root.glob("clip_*.pt"))
    if not clip_paths:
        raise ValueError(f"No clip files found in {clips_root}")

    clips_by_video: dict[str, list[Path]] = defaultdict(list)
    for clip_path in clip_paths:
        clips_by_video[infer_source_video_id(clip_path)].append(clip_path)

    video_ids = sorted(clips_by_video)
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    val_count = _compute_val_count(len(video_ids), val_fraction)
    val_videos = set(video_ids[:val_count])
    train_videos = set(video_ids[val_count:])

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    _reset_split_dir(train_dir)
    _reset_split_dir(val_dir)

    train_entries = _materialize_split(
        train_dir,
        sorted(train_videos),
        clips_by_video,
        link_mode=link_mode,
    )
    val_entries = _materialize_split(
        val_dir,
        sorted(val_videos),
        clips_by_video,
        link_mode=link_mode,
    )

    manifest = {
        "clips_dir": str(clips_root),
        "output_dir": str(output_root),
        "seed": seed,
        "val_fraction": val_fraction,
        "link_mode": link_mode,
        "splits": {
            "train": train_entries,
            "val": val_entries,
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def infer_source_video_id(clip_path: str | Path) -> str:
    """Infer the source video id for a generated clip path.

    Prefers the canonical 11-character YouTube id when present so clips from
    DB sub-clips like ``clip_overlay_<video>_clip70_0.pt`` stay grouped by the
    original source video.
    """
    stem = Path(clip_path).stem
    prefix = "clip_overlay_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix) :]
        if len(remainder) >= 11:
            maybe_youtube_id = remainder[:11]
            if YOUTUBE_VIDEO_ID_RE.fullmatch(maybe_youtube_id):
                return maybe_youtube_id
        if "_" in remainder:
            return remainder.rsplit("_", 1)[0]
        return remainder
    return stem


def _compute_val_count(num_videos: int, val_fraction: float) -> int:
    if num_videos <= 1 or val_fraction <= 0:
        return 0
    val_count = int(round(num_videos * val_fraction))
    return min(max(val_count, 1), num_videos - 1)


def _reset_split_dir(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for clip_path in split_dir.glob("clip_*.pt"):
        clip_path.unlink()


def _materialize_split(
    split_dir: Path,
    video_ids: list[str],
    clips_by_video: dict[str, list[Path]],
    *,
    link_mode: str,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for video_id in video_ids:
        for src_path in sorted(clips_by_video[video_id]):
            dest_path = split_dir / src_path.name
            _materialize_clip(src_path, dest_path, link_mode=link_mode)
            entries.append(
                {
                    "clip": str(dest_path.relative_to(split_dir.parent)),
                    "source_video_id": video_id,
                }
            )
    return entries


def _materialize_clip(src_path: Path, dest_path: Path, *, link_mode: str) -> None:
    if dest_path.exists():
        dest_path.unlink()

    if link_mode == "copy":
        shutil.copy2(src_path, dest_path)
        return
    if link_mode != "hardlink":
        raise ValueError(f"Unsupported link mode: {link_mode}")

    try:
        os.link(src_path, dest_path)
    except OSError:
        shutil.copy2(src_path, dest_path)
