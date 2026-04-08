"""Tests for OTB YOLO dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pipeline.screen.otb_yolo_dataset import (
    FrameCandidate,
    PseudoLabel,
    export_otb_yolo_dataset,
)


class _FakeLabeler:
    def label_image(self, image: np.ndarray) -> PseudoLabel | None:
        return PseudoLabel(bbox=(10, 20, 60, 50), confidence=0.9, source="test")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _candidate(tmp_path: Path, video_id: str, label: str, tier: str = "lores") -> FrameCandidate:
    path = tmp_path / video_id / tier / f"{label}.jpg"
    _write_image(path)
    return FrameCandidate(video_id=video_id, label=label, image_path=path, tier=tier)


def test_export_otb_yolo_dataset_creates_video_disjoint_splits(tmp_path: Path) -> None:
    positives = [
        _candidate(tmp_path, "pos1", "25pct"),
        _candidate(tmp_path, "pos1", "50pct"),
        _candidate(tmp_path, "pos2", "25pct"),
        _candidate(tmp_path, "pos3", "25pct"),
    ]
    negatives = [
        _candidate(tmp_path, "neg1", "25pct"),
        _candidate(tmp_path, "neg2", "25pct"),
        _candidate(tmp_path, "neg3", "25pct"),
    ]

    export = export_otb_yolo_dataset(
        dataset_dir=tmp_path / "dataset",
        positives=positives,
        negatives=negatives,
        labeler=_FakeLabeler(),
        val_fraction=0.25,
        test_fraction=0.25,
        seed=7,
    )

    dataset_dir = tmp_path / "dataset"

    manifest = json.loads(export.manifest_path.read_text())
    split_videos = {
        split: {entry["video_id"] for entry in entries}
        for split, entries in manifest["splits"].items()
    }

    assert split_videos["train"].isdisjoint(split_videos["val"])
    assert split_videos["train"].isdisjoint(split_videos["test"])
    assert split_videos["val"].isdisjoint(split_videos["test"])
    assert export.train.positives > 0
    assert export.train.negatives >= 0

    positive_entries = [
        entry
        for entries in manifest["splits"].values()
        for entry in entries
        if entry["has_otb_board"]
    ]
    negative_entries = [
        entry
        for entries in manifest["splits"].values()
        for entry in entries
        if not entry["has_otb_board"]
    ]

    assert positive_entries
    assert negative_entries
    assert all(entry["label_path"] is not None for entry in positive_entries)
    assert all(entry["label_path"] is None for entry in negative_entries)
    combined_entries = positive_entries + negative_entries
    assert all((dataset_dir / entry["image"]).exists() for entry in combined_entries)
    assert all((dataset_dir / entry["label_path"]).exists() for entry in positive_entries)


def test_export_otb_yolo_dataset_requires_positive_labels(tmp_path: Path) -> None:
    positives = [_candidate(tmp_path, "pos1", "25pct")]
    negatives = [_candidate(tmp_path, "neg1", "25pct")]

    class EmptyLabeler:
        def label_image(self, image: np.ndarray) -> PseudoLabel | None:
            return None

    with pytest.raises(ValueError, match="no positive board labels"):
        export_otb_yolo_dataset(
            dataset_dir=tmp_path / "dataset",
            positives=positives,
            negatives=negatives,
            labeler=EmptyLabeler(),
            val_fraction=0.0,
            test_fraction=0.0,
        )
