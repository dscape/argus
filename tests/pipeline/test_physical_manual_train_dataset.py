from __future__ import annotations

import json

import numpy as np
import pipeline.physical.manual_train_dataset as manual_train_dataset


def _gradient_board(size: int = 128) -> np.ndarray:
    y = np.linspace(0, 255, size, dtype=np.uint8)
    x = np.linspace(0, 255, size, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv, yv, np.full_like(xv, 127)], axis=2)


def test_save_board_annotation_writes_manual_train_manifest_and_crops(
    tmp_path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "data" / "physical" / "train_manual"
    monkeypatch.setattr(manual_train_dataset, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manual_train_dataset, "DATASET_ROOT", dataset_root)
    monkeypatch.setattr(manual_train_dataset, "BOARDS_DIR", dataset_root / "boards")
    monkeypatch.setattr(manual_train_dataset, "SQUARES_DIR", dataset_root / "squares")
    monkeypatch.setattr(
        manual_train_dataset,
        "BOARD_ANNOTATIONS_PATH",
        dataset_root / "board_annotations.jsonl",
    )
    monkeypatch.setattr(
        manual_train_dataset,
        "SQUARE_MANIFEST_PATH",
        dataset_root / "square_manifest.jsonl",
    )

    labels = [None] * 64
    labels[0] = 4
    labels[63] = 10

    record = manual_train_dataset.save_board_annotation(
        _gradient_board(128),
        clip_path="data/argus/train_real/clip_overlay_demo_clip1_0.pt",
        frame_index=3,
        source_video_id="demo",
        corners=[[0, 0], [127, 0], [127, 127], [0, 127]],
        labels=labels,
        output_size=64,
    )

    assert record["annotation_id"] == "clip_overlay_demo_clip1_0_frame0003"
    assert record["labeled_square_count"] == 2
    assert (dataset_root / "boards" / "clip_overlay_demo_clip1_0_frame0003.jpg").exists()
    assert (dataset_root / "squares" / "clip_overlay_demo_clip1_0_frame0003_a8.jpg").exists()
    assert (dataset_root / "squares" / "clip_overlay_demo_clip1_0_frame0003_h1.jpg").exists()

    square_rows = [
        json.loads(line)
        for line in (dataset_root / "square_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(square_rows) == 2
    assert {row["split"] for row in square_rows} == {"train_manual"}

    summary = manual_train_dataset.get_annotation_summary()
    assert summary["dataset_root"] == "data/physical/train_manual"
    assert summary["board_annotation_count"] == 1
    assert summary["square_crop_count"] == 2
