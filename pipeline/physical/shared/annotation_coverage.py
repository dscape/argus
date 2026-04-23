"""Per-channel annotation coverage and ROI actions for physical-board data.

Joins three sources:
  - Clips in `data/argus/train_real/*.pt` (identified by `source_channel_handle`)
  - Train-split annotations in `data/physical/train/board_annotations.jsonl`
  - Val-split annotations in `data/physical/val/board_annotations.jsonl`

Val annotations gate board-probe usability of a channel's clips
(`pipeline/physical/shared/real_board_data.py`), so the ROI action
prioritizes val coverage first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
DEFAULT_TRAIN_ANNOTATIONS = _PROJECT_ROOT / "data" / "physical" / "train" / "board_annotations.jsonl"
DEFAULT_VAL_ANNOTATIONS = _PROJECT_ROOT / "data" / "physical" / "val" / "board_annotations.jsonl"

DEFAULT_TRAIN_COVERAGE_TARGET = 5
DEFAULT_MIN_VAL_COVERAGE = 3


@dataclass(frozen=True)
class ChannelCoverage:
    channel: str
    clip_paths: tuple[str, ...]
    val_annotated_frames: int
    train_annotated_frames: int
    roi_action: str

    @property
    def clips(self) -> int:
        return len(self.clip_paths)

    @property
    def probe_usable_clips(self) -> int:
        return self.clips if self.val_annotated_frames > 0 else 0


@dataclass(frozen=True)
class CoverageReport:
    channels: tuple[ChannelCoverage, ...]
    clips_without_channel: tuple[str, ...]
    train_total: int
    val_total: int

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "channel": row.channel,
                "clips": row.clips,
                "val_annotated_frames": row.val_annotated_frames,
                "train_annotated_frames": row.train_annotated_frames,
                "probe_usable_clips": row.probe_usable_clips,
                "roi_action": row.roi_action,
                "clip_paths": list(row.clip_paths),
            }
            for row in self.channels
        ]


def compute_coverage(
    *,
    clips_dir: str | Path = DEFAULT_CLIPS_DIR,
    train_annotations_path: str | Path = DEFAULT_TRAIN_ANNOTATIONS,
    val_annotations_path: str | Path = DEFAULT_VAL_ANNOTATIONS,
    train_coverage_target: int = DEFAULT_TRAIN_COVERAGE_TARGET,
    min_val_coverage: int = DEFAULT_MIN_VAL_COVERAGE,
) -> CoverageReport:
    """Join clips on disk with train/val annotations and emit ROI actions."""
    clip_channels = _load_clip_channels(Path(clips_dir))
    train_counts_by_clip = _count_annotations_by_clip(Path(train_annotations_path))
    val_counts_by_clip = _count_annotations_by_clip(Path(val_annotations_path))

    channel_to_clips: dict[str, list[str]] = {}
    train_counts_by_channel: dict[str, int] = {}
    val_counts_by_channel: dict[str, int] = {}
    clips_without_channel: list[str] = []

    for clip_path, channel in clip_channels.items():
        if not channel:
            clips_without_channel.append(clip_path)
            continue
        channel_to_clips.setdefault(channel, []).append(clip_path)
        train_counts_by_channel[channel] = train_counts_by_channel.get(
            channel, 0
        ) + train_counts_by_clip.get(clip_path, 0)
        val_counts_by_channel[channel] = val_counts_by_channel.get(
            channel, 0
        ) + val_counts_by_clip.get(clip_path, 0)

    channels: list[ChannelCoverage] = []
    for channel, clip_paths in sorted(channel_to_clips.items()):
        val_count = val_counts_by_channel.get(channel, 0)
        train_count = train_counts_by_channel.get(channel, 0)
        action = _roi_action(
            clips=len(clip_paths),
            val_count=val_count,
            train_count=train_count,
            train_coverage_target=train_coverage_target,
            min_val_coverage=min_val_coverage,
        )
        channels.append(
            ChannelCoverage(
                channel=channel,
                clip_paths=tuple(sorted(clip_paths)),
                val_annotated_frames=val_count,
                train_annotated_frames=train_count,
                roi_action=action,
            )
        )

    channels.sort(key=_priority_sort_key)

    return CoverageReport(
        channels=tuple(channels),
        clips_without_channel=tuple(sorted(clips_without_channel)),
        train_total=sum(train_counts_by_clip.values()),
        val_total=sum(val_counts_by_clip.values()),
    )


ROI_NEEDS_VAL = "needs_val"
ROI_NEEDS_TRAIN = "needs_train"
ROI_ADD_DIVERSITY = "add_diversity"
ROI_SATURATED = "saturated"

_ROI_PRIORITY = {
    ROI_NEEDS_VAL: 0,
    ROI_NEEDS_TRAIN: 1,
    ROI_ADD_DIVERSITY: 2,
    ROI_SATURATED: 3,
}


def _roi_action(
    *,
    clips: int,
    val_count: int,
    train_count: int,
    train_coverage_target: int,
    min_val_coverage: int,
) -> str:
    if clips == 0:
        return ROI_SATURATED
    if val_count < min_val_coverage:
        return ROI_NEEDS_VAL
    if train_count < train_coverage_target:
        return ROI_NEEDS_TRAIN
    return ROI_ADD_DIVERSITY


def _priority_sort_key(row: ChannelCoverage) -> tuple[int, int, int, str]:
    return (
        _ROI_PRIORITY[row.roi_action],
        row.train_annotated_frames,
        -row.clips,
        row.channel,
    )


def _load_clip_channels(clips_dir: Path) -> dict[str, str | None]:
    if not clips_dir.exists():
        return {}
    channels: dict[str, str | None] = {}
    for clip_path in sorted(clips_dir.glob("clip_*.pt")):
        try:
            payload = torch.load(clip_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        handle = payload.get("source_channel_handle") if isinstance(payload, dict) else None
        if isinstance(handle, str) and handle:
            channel: str | None = handle
        else:
            channel = None
        rel = str(clip_path.resolve().relative_to(_PROJECT_ROOT.resolve()))
        channels[rel] = channel
    return channels


def _count_annotations_by_clip(annotations_path: Path) -> dict[str, int]:
    if not annotations_path.exists():
        return {}
    counts: dict[str, set[int]] = {}
    for line in annotations_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        clip_path = record.get("clip_path")
        if not isinstance(clip_path, str):
            continue
        frame_index = record.get("frame_index")
        if not isinstance(frame_index, int):
            continue
        counts.setdefault(clip_path, set()).add(frame_index)
    return {clip_path: len(frames) for clip_path, frames in counts.items()}
