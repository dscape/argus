from __future__ import annotations

import json
from pathlib import Path

import torch
from pipeline.physical.square_classifier_data import (
    class_counts,
    preprocess_square_crop,
)


def _make_annotation_bundle(tmp_path: Path, *, labels: list[int]) -> Path:
    clip_path = tmp_path / "data" / "argus" / "train_real" / "clip_test.pt"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    frame = torch.full((3, 64, 64), 127, dtype=torch.uint8)
    torch.save({"frames": torch.stack([frame], dim=0)}, clip_path)

    annotation_root = tmp_path / "data" / "physical" / "val"
    annotation_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "annotation_id": "ann-1",
        "clip_path": "data/argus/train_real/clip_test.pt",
        "frame_index": 0,
        "source_video_id": "video-1",
        "corners": [[8.0, 8.0], [56.0, 8.0], [56.0, 56.0], [8.0, 56.0]],
        "labels": labels,
    }
    (annotation_root / "board_annotations.jsonl").write_text(json.dumps(payload) + "\n")
    return annotation_root


def test_occupancy_dataset_yields_64_samples_per_board(tmp_path: Path) -> None:
    labels = [0] * 64
    labels[0] = 6  # white king on a8
    labels[63] = 12  # black king on h1
    annotation_root = _make_annotation_bundle(tmp_path, labels=labels)

    import pipeline.physical.oblique_square_context as oblique_context

    original_root = oblique_context._PROJECT_ROOT
    oblique_context._PROJECT_ROOT = tmp_path
    try:
        from pipeline.physical.square_classifier_data import load_occupancy_dataset

        dataset = load_occupancy_dataset(annotation_root, input_size=32)
        assert len(dataset) == 64
        image, label = dataset[0]
        assert image.shape == (3, 32, 32)
        assert label.item() == 1  # occupied (white king)

        empty_image, empty_label = dataset[1]
        assert empty_label.item() == 0  # empty
    finally:
        oblique_context._PROJECT_ROOT = original_root


def test_piece_dataset_skips_empty_squares_and_emits_12_class_labels(tmp_path: Path) -> None:
    labels = [0] * 64
    labels[0] = 6  # white king -> piece_label 5
    labels[1] = 1  # white pawn -> piece_label 0
    labels[2] = 12  # black king -> piece_label 11
    annotation_root = _make_annotation_bundle(tmp_path, labels=labels)

    import pipeline.physical.oblique_square_context as oblique_context

    original_root = oblique_context._PROJECT_ROOT
    oblique_context._PROJECT_ROOT = tmp_path
    try:
        from pipeline.physical.square_classifier_data import load_piece_dataset

        dataset = load_piece_dataset(annotation_root, input_size=32)
        assert len(dataset) == 3
        piece_labels = sorted(int(dataset[i][1].item()) for i in range(3))
        assert piece_labels == [0, 5, 11]
    finally:
        oblique_context._PROJECT_ROOT = original_root


def test_occupancy_dataset_class_counts_match_label_distribution(tmp_path: Path) -> None:
    labels = [0] * 64
    labels[0] = 1  # white pawn (class 1)
    labels[8] = 7  # black pawn (class 7)
    annotation_root = _make_annotation_bundle(tmp_path, labels=labels)

    import pipeline.physical.oblique_square_context as oblique_context

    original_root = oblique_context._PROJECT_ROOT
    oblique_context._PROJECT_ROOT = tmp_path
    try:
        from pipeline.physical.square_classifier_data import load_occupancy_dataset

        dataset = load_occupancy_dataset(annotation_root, input_size=32)
    finally:
        oblique_context._PROJECT_ROOT = original_root

    counts = class_counts(dataset)
    assert counts["empty"] == 62
    assert counts["P"] == 1
    assert counts["p"] == 1


def test_preprocess_square_crop_normalizes_values_into_standard_range() -> None:
    import numpy as np

    crop = np.full((64, 64, 3), 128, dtype=np.uint8)
    tensor = preprocess_square_crop(crop, size=32)
    assert tensor.shape == (3, 32, 32)
    assert torch.isfinite(tensor).all()
    # Mid-gray normalized through ImageNet stats stays within [-3, 3].
    assert tensor.abs().max().item() < 3.0
