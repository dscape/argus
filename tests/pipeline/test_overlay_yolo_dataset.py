"""Tests for YOLO overlay dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pipeline.overlay import yolo_dataset


def _write_image(path: Path, size: tuple[int, int]) -> None:
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_export_overlay_yolo_dataset_creates_video_disjoint_splits(tmp_path, monkeypatch) -> None:
    videos_dir = tmp_path / "videos"
    fixtures_dir = tmp_path / "fixtures"
    dataset_dir = tmp_path / "dataset"

    _write_image(videos_dir / "video1" / "hires" / "25pct.jpg", (1280, 720))
    _write_image(videos_dir / "video1" / "hires" / "50pct.jpg", (1280, 720))
    _write_image(videos_dir / "video2" / "hires" / "25pct.jpg", (1280, 720))
    _write_image(videos_dir / "video3" / "hires" / "25pct.jpg", (1280, 720))
    _write_image(videos_dir / "video4" / "fullres" / "50pct.jpg", (1920, 1080))
    _write_image(videos_dir / "video5" / "hires" / "50pct.jpg", (1280, 720))

    all_gt = {
        "video1/25pct": {
            "image": "video1/25pct.jpg",
            "has_overlay": True,
            "bbox": [128, 72, 640, 360],
            "frame_width": 1280,
            "frame_height": 720,
        },
        "video1/50pct": {
            "image": "video1/50pct.jpg",
            "has_overlay": False,
            "bbox": None,
            "frame_width": 1280,
            "frame_height": 720,
        },
        "video2/25pct": {
            "image": "video2/25pct.jpg",
            "has_overlay": True,
            "bbox": [64, 36, 320, 180],
            "frame_width": 1280,
            "frame_height": 720,
        },
        "video3/25pct": {
            "image": "video3/25pct.jpg",
            "has_overlay": False,
            "bbox": None,
            "frame_width": 1280,
            "frame_height": 720,
        },
        "video4/50pct": {
            "image": "video4/50pct.jpg",
            "has_overlay": False,
            "bbox": None,
            "frame_width": 1920,
            "frame_height": 1080,
        },
        "video5/50pct": {
            "image": "video5/50pct.jpg",
            "has_overlay": True,
            "bbox": [80, 40, 480, 480],
            "frame_width": 1280,
            "frame_height": 720,
        },
    }
    all_gt_path = tmp_path / "all_ground_truth.json"
    all_gt_path.write_text(json.dumps(all_gt))

    _write_image(fixtures_dir / "video5" / "25pct.jpg", (1280, 720))
    fixture_gt = {
        "video5/25pct": {
            "image": "video5/25pct.jpg",
            "has_overlay": True,
            "bbox": [320, 180, 320, 320],
            "frame_width": 1280,
            "frame_height": 720,
        }
    }
    fixture_gt_path = tmp_path / "fixture_ground_truth.json"
    fixture_gt_path.write_text(json.dumps(fixture_gt))

    monkeypatch.setattr(yolo_dataset, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(yolo_dataset, "FIXTURES_DIR", fixtures_dir)
    monkeypatch.setattr(
        yolo_dataset,
        "find_frame",
        lambda video_id, tier, label: videos_dir / video_id / tier / f"{label}.jpg",
    )

    export = yolo_dataset.export_overlay_yolo_dataset(
        dataset_dir=dataset_dir,
        source_ground_truth=all_gt_path,
        fixture_ground_truth=fixture_gt_path,
        val_fraction=0.5,
        seed=7,
    )

    assert export.train.images + export.val.images == 5
    assert export.train.positives + export.val.positives == 2
    assert export.train.negatives + export.val.negatives == 3
    assert export.train.images >= 1
    assert export.val.images >= 1
    assert export.test.images == 1
    assert export.test.positives == 1
    assert export.test.negatives == 0

    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
    test_labels = list((dataset_dir / "labels" / "test").glob("*.txt"))
    assert len(train_labels) == 1
    assert len(val_labels) == 1
    assert len(test_labels) == 1

    positive_label_text = (train_labels[0] if train_labels else val_labels[0]).read_text().strip()
    parts = positive_label_text.split()
    assert parts[0] == "0"
    assert len(parts) == 5

    assert export.dataset_yaml.exists()
    yaml_text = export.dataset_yaml.read_text()
    assert "train: images/train" in yaml_text
    assert "test: images/test" in yaml_text

    manifest = json.loads(export.manifest_path.read_text())
    train_videos = {entry["key"].split("/")[0] for entry in manifest["splits"]["train"]}
    val_videos = {entry["key"].split("/")[0] for entry in manifest["splits"]["val"]}
    test_videos = {entry["key"].split("/")[0] for entry in manifest["splits"]["test"]}
    assert train_videos.isdisjoint(val_videos)
    assert "video1" in train_videos or "video1" in val_videos
    assert "video5" not in train_videos
    assert "video5" not in val_videos
    assert test_videos == {"video5"}
    assert len(manifest["splits"]["test"]) == 1
    assert manifest["splits"]["test"][0]["key"] == "video5/25pct"


def test_export_overlay_yolo_dataset_requires_usable_train_split(tmp_path, monkeypatch) -> None:
    source_ground_truth = tmp_path / "all_ground_truth.json"
    fixture_ground_truth = tmp_path / "fixture_ground_truth.json"
    videos_dir = tmp_path / "videos"
    fixture_ground_truth.write_text("{}")
    _write_image(videos_dir / "video1" / "hires" / "25pct.jpg", (1280, 720))
    monkeypatch.setattr(yolo_dataset, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(
        yolo_dataset,
        "find_frame",
        lambda video_id, tier, label: videos_dir / video_id / tier / f"{label}.jpg",
    )
    source_ground_truth.write_text(
        json.dumps(
            {
                "video1/25pct": {
                    "image": "video1/25pct.jpg",
                    "has_overlay": False,
                    "bbox": None,
                    "frame_width": 1280,
                    "frame_height": 720,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="no positive overlay labels"):
        yolo_dataset.export_overlay_yolo_dataset(
            dataset_dir=tmp_path / "dataset",
            source_ground_truth=source_ground_truth,
            fixture_ground_truth=fixture_ground_truth,
            val_fraction=0.0,
        )


def test_export_overlay_yolo_dataset_requires_ground_truth_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Ground truth file not found"):
        yolo_dataset.export_overlay_yolo_dataset(
            dataset_dir=tmp_path / "dataset",
            source_ground_truth=tmp_path / "missing.json",
            fixture_ground_truth=tmp_path / "fixture_ground_truth.json",
        )
