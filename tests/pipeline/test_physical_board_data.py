from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch
from pipeline.physical.board_probe.board_data import (
    INPUT_SIZE,
    PhysicalEvalBoardDataset,
    PhysicalEvalBoardRow,
    PhysicalManualTrainBoardDataset,
    PhysicalSyntheticClipBoardDataset,
    load_synthetic_board_rows,
    preprocess_board_image,
)


def _write_synthetic_clip(path, *, fens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = torch.zeros((len(fens), 3, 64, 64), dtype=torch.float32)
    torch.save({"frames": frames, "fens": fens, "board_flipped": False}, path)


def test_preprocess_board_image_outputs_normalized_chw() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    tensor = preprocess_board_image(image)

    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == torch.float32


def test_physical_synthetic_clip_board_dataset_returns_board_and_targets(tmp_path) -> None:
    clip_dir = tmp_path / "data" / "argus" / "train"
    _write_synthetic_clip(
        clip_dir / "clip_000000.pt",
        fens=[
            "8/8/8/8/8/8/8/K6k w - - 0 1",
            "8/8/8/8/8/8/8/K5k1 w - - 0 1",
        ],
    )

    dataset = PhysicalSyntheticClipBoardDataset(
        clips_dir=clip_dir,
        num_positions=2,
        image_size=64,
        seed=7,
    )

    image, targets, corners = dataset[0]

    assert image.shape == (3, 64, 64)
    assert targets.shape == (64,)
    assert corners.shape == (4, 2)


def test_load_synthetic_board_rows_rejects_request_larger_than_available(tmp_path) -> None:
    clip_dir = tmp_path / "data" / "argus" / "train"
    _write_synthetic_clip(
        clip_dir / "clip_000000.pt",
        fens=["8/8/8/8/8/8/8/K6k w - - 0 1"],
    )

    with pytest.raises(ValueError, match="Requested more labeled synthetic frames"):
        load_synthetic_board_rows(clips_dir=clip_dir, num_positions=2, seed=7)


def test_physical_eval_board_dataset_loads_board_neighborhoods(tmp_path) -> None:
    project_root = tmp_path
    board_path = project_root / "data" / "physical" / "val" / "boards" / "sample.jpg"
    board_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(board_path), np.full((32, 32, 3), 127, dtype=np.uint8))
    clip_path = project_root / "data" / "argus" / "train_real" / "clip.pt"
    clip_path.parent.mkdir(parents=True)
    torch.save({"frames": torch.zeros((1, 3, 32, 32), dtype=torch.uint8)}, clip_path)

    dataset = PhysicalEvalBoardDataset(
        rows=[
            PhysicalEvalBoardRow(
                annotation_id="ann-1",
                board_path="data/physical/val/boards/sample.jpg",
                labels=tuple([0] * 64),
                source_video_id="video-1",
                corners=((0.0, 0.0), (31.0, 0.0), (31.0, 31.0), (0.0, 31.0)),
                clip_path="data/argus/train_real/clip.pt",
                frame_index=0,
            )
        ],
        image_size=32,
    )

    import pipeline.physical.board_probe.board_data as board_data

    original_root = board_data._PROJECT_ROOT
    board_data._PROJECT_ROOT = project_root
    try:
        image, targets, piece_bboxes = dataset[0]
    finally:
        board_data._PROJECT_ROOT = original_root

    assert image.shape == (3, 32, 32)
    assert targets.shape == (64,)
    assert piece_bboxes.shape == (64, 4)


def test_physical_manual_train_board_dataset_loads_board_neighborhoods(tmp_path) -> None:
    project_root = tmp_path
    board_path = project_root / "data" / "physical" / "train" / "boards" / "sample.jpg"
    board_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(board_path), np.full((32, 32, 3), 127, dtype=np.uint8))
    clip_path = project_root / "data" / "argus" / "train_real" / "clip.pt"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": torch.zeros((1, 3, 32, 32), dtype=torch.uint8)}, clip_path)

    dataset = PhysicalManualTrainBoardDataset(
        rows=[
            PhysicalEvalBoardRow(
                annotation_id="ann-1",
                board_path="data/physical/train/boards/sample.jpg",
                labels=tuple([0] * 64),
                source_video_id="video-1",
                corners=((0.0, 0.0), (31.0, 0.0), (31.0, 31.0), (0.0, 31.0)),
                clip_path="data/argus/train_real/clip.pt",
                frame_index=0,
            )
        ],
        image_size=32,
    )

    import pipeline.physical.board_probe.board_data as board_data

    original_root = board_data._PROJECT_ROOT
    board_data._PROJECT_ROOT = project_root
    try:
        image, targets, piece_bboxes = dataset[0]
    finally:
        board_data._PROJECT_ROOT = original_root

    assert image.shape == (3, 32, 32)
    assert targets.shape == (64,)
    assert piece_bboxes.shape == (64, 4)


def test_load_annotated_board_rows_prefers_native_full_frame_corners(tmp_path) -> None:
    annotation_root = tmp_path / "data" / "physical" / "val"
    annotation_root.mkdir(parents=True)
    payload = {
        "annotation_id": "ann-1",
        "rectified_board_path": "data/physical/val/boards/sample.jpg",
        "labels": [0] * 64,
        "source_video_id": "video-1",
        "corners": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "clip_path": "data/argus/train_real/clip.pt",
        "frame_index": 0,
        "clip_frame_size": [224, 224],
        "native_corners": [[10, 20], [30, 40], [50, 60], [70, 80]],
        "native_image_bbox": [100, 200, 300, 400],
        "source_frame_index": 123,
    }
    (annotation_root / "board_annotations.jsonl").write_text(json.dumps(payload) + "\n")

    import pipeline.physical.board_probe.board_data as board_data

    rows = board_data.load_annotated_board_rows(annotation_root)

    assert len(rows) == 1
    assert rows[0].corners == ((110.0, 220.0), (130.0, 240.0), (150.0, 260.0), (170.0, 280.0))
    assert rows[0].source_frame_index == 123
    assert rows[0].native_image_bbox == (100, 200, 300, 400)


def test_load_annotated_board_frame_bgr_prefers_native_source_frame(monkeypatch) -> None:
    row = PhysicalEvalBoardRow(
        annotation_id="ann-1",
        board_path="data/physical/val/boards/sample.jpg",
        labels=tuple([0] * 64),
        source_video_id="video-1",
        corners=((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
        clip_path="data/argus/train_real/clip.pt",
        frame_index=0,
        source_frame_index=77,
    )

    import pipeline.physical.board_probe.board_data as board_data

    expected = np.full((8, 8, 3), 33, dtype=np.uint8)
    calls: list[tuple[str, int]] = []

    def fake_load(*, source_video_id: str, source_frame_index: int) -> np.ndarray:
        calls.append((source_video_id, source_frame_index))
        return expected

    monkeypatch.setattr(board_data._NATIVE_FRAME_LOADER, "load", fake_load)

    image = board_data.load_annotated_board_frame_bgr(row, clip_cache={})

    assert calls == [("video-1", 77)]
    assert np.array_equal(image, expected)
