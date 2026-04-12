"""Physical training-dataset helpers that enforce held-out eval exclusions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.overlay.training_dataset import export_training_dataset
from pipeline.physical.eval_dataset import get_held_out_source_video_ids


def export_physical_training_dataset(
    clips_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    """Export a train/val split while excluding held-out physical eval videos."""
    held_out_source_video_ids = set(get_held_out_source_video_ids())
    if not held_out_source_video_ids:
        raise ValueError(
            "No held-out physical eval source videos found in data/physical/eval; "
            "annotate the eval set before exporting a physical training split"
        )

    return export_training_dataset(
        clips_dir,
        output_dir,
        val_fraction=val_fraction,
        seed=seed,
        link_mode=link_mode,
        exclude_source_video_ids=held_out_source_video_ids,
    )
