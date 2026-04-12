from __future__ import annotations

import json

import numpy as np
import pipeline.physical.eval_dataset as eval_dataset


def _gradient_board(size: int = 128) -> np.ndarray:
    y = np.linspace(0, 255, size, dtype=np.uint8)
    x = np.linspace(0, 255, size, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv, yv, np.full_like(xv, 127)], axis=2)


def test_rectify_board_image_returns_square_rgb_image() -> None:
    image = _gradient_board(128)
    rectified = eval_dataset.rectify_board_image(
        image,
        corners=[[0, 0], [127, 0], [127, 127], [0, 127]],
        output_size=64,
    )

    assert rectified.shape == (64, 64, 3)


def test_save_board_annotation_writes_manifest_and_crops(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "data" / "physical" / "eval"
    monkeypatch.setattr(eval_dataset, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(eval_dataset, "DATASET_ROOT", dataset_root)
    monkeypatch.setattr(eval_dataset, "BOARDS_DIR", dataset_root / "boards")
    monkeypatch.setattr(eval_dataset, "SQUARES_DIR", dataset_root / "squares")
    monkeypatch.setattr(
        eval_dataset,
        "BOARD_ANNOTATIONS_PATH",
        dataset_root / "board_annotations.jsonl",
    )
    monkeypatch.setattr(
        eval_dataset,
        "SQUARE_MANIFEST_PATH",
        dataset_root / "square_manifest.jsonl",
    )

    labels = [None] * 64
    labels[0] = 4
    labels[63] = 10

    record = eval_dataset.save_board_annotation(
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
    assert {row["square_name"] for row in square_rows} == {"a8", "h1"}

    summary = eval_dataset.get_annotation_summary()
    assert summary["board_annotation_count"] == 1
    assert summary["square_crop_count"] == 2
    assert summary["class_counts"]["R"] == 1
    assert summary["class_counts"]["r"] == 1
