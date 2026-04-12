from __future__ import annotations

from pathlib import Path

import torch
from pipeline.overlay.training_dataset import export_training_dataset, infer_source_video_id


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


def test_infer_source_video_id_groups_db_clip_suffixes() -> None:
    assert infer_source_video_id("clip_overlay_2wWUKmCBr6A_clip5_0.pt") == "2wWUKmCBr6A"
    assert infer_source_video_id("clip_overlay_YEjQAF0hbBs_1.pt") == "YEjQAF0hbBs"


def test_export_training_dataset_creates_video_disjoint_split(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "clip_overlay_2wWUKmCBr6A_clip5_0.pt")
    _write_clip(clips_dir / "clip_overlay_2wWUKmCBr6A_clip5_1.pt")
    _write_clip(clips_dir / "clip_overlay_7RaBQag34Hk_clip26_0.pt")
    _write_clip(clips_dir / "clip_overlay_YEjQAF0hbBs_1.pt")

    output_dir = tmp_path / "dataset"
    manifest = export_training_dataset(clips_dir, output_dir, val_fraction=0.34, seed=1)

    train_entries = manifest["splits"]["train"]
    val_entries = manifest["splits"]["val"]
    train_videos = {entry["source_video_id"] for entry in train_entries}
    val_videos = {entry["source_video_id"] for entry in val_entries}

    assert train_videos
    assert val_videos
    assert train_videos.isdisjoint(val_videos)
    assert sorted(path.name for path in (output_dir / "train").glob("clip_*.pt")) == sorted(
        Path(entry["clip"]).name for entry in train_entries
    )
    assert sorted(path.name for path in (output_dir / "val").glob("clip_*.pt")) == sorted(
        Path(entry["clip"]).name for entry in val_entries
    )
    assert (output_dir / "manifest.json").exists()


def test_export_training_dataset_keeps_single_video_in_train_when_split_requested(
    tmp_path: Path,
) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "clip_overlay_2wWUKmCBr6A_0.pt")

    output_dir = tmp_path / "dataset"
    manifest = export_training_dataset(clips_dir, output_dir, val_fraction=0.5, seed=1)

    assert len(manifest["splits"]["train"]) == 1
    assert manifest["splits"]["val"] == []


def test_export_training_dataset_excludes_held_out_source_videos(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "clip_overlay_2wWUKmCBr6A_clip5_0.pt")
    _write_clip(clips_dir / "clip_overlay_7RaBQag34Hk_clip26_0.pt")

    output_dir = tmp_path / "dataset"
    manifest = export_training_dataset(
        clips_dir,
        output_dir,
        exclude_source_video_ids={"2wWUKmCBr6A"},
    )

    assert manifest["excluded_source_video_ids"] == ["2wWUKmCBr6A"]
    assert manifest["excluded_clip_count"] == 1
    assert {
        entry["source_video_id"] for split in manifest["splits"].values() for entry in split
    } == {"7RaBQag34Hk"}
