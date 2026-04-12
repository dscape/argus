from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pipeline.physical.training_dataset import export_physical_training_dataset


def _write_clip(path: Path) -> None:
    torch.save(
        {
            "frames": torch.zeros(2, 3, 4, 4, dtype=torch.uint8),
            "move_targets": torch.zeros(2, dtype=torch.long),
            "detect_targets": torch.zeros(2, dtype=torch.float32),
            "legal_masks": torch.zeros(2, 4, dtype=torch.bool),
            "move_mask": torch.zeros(2, dtype=torch.bool),
        },
        path,
    )


def test_export_physical_training_dataset_requires_held_out_videos(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "pipeline.physical.training_dataset.get_held_out_source_video_ids",
        lambda: [],
    )

    with pytest.raises(ValueError, match="annotate the eval set"):
        export_physical_training_dataset(tmp_path / "clips", tmp_path / "dataset")


def test_export_physical_training_dataset_excludes_held_out_videos(
    monkeypatch,
    tmp_path: Path,
) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "clip_overlay_2wWUKmCBr6A_clip5_0.pt")
    _write_clip(clips_dir / "clip_overlay_7RaBQag34Hk_clip26_0.pt")

    monkeypatch.setattr(
        "pipeline.physical.training_dataset.get_held_out_source_video_ids",
        lambda: ["2wWUKmCBr6A"],
    )

    manifest = export_physical_training_dataset(clips_dir, tmp_path / "dataset")

    assert manifest["excluded_source_video_ids"] == ["2wWUKmCBr6A"]
    assert manifest["excluded_clip_count"] == 1
