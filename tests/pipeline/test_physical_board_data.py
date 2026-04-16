from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from pipeline.physical.board_data import (
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

    image, targets = dataset[0]

    assert image.shape == (3, 64, 64)
    assert targets.shape == (64,)


def test_load_synthetic_board_rows_rejects_request_larger_than_available(tmp_path) -> None:
    clip_dir = tmp_path / "data" / "argus" / "train"
    _write_synthetic_clip(
        clip_dir / "clip_000000.pt",
        fens=["8/8/8/8/8/8/8/K6k w - - 0 1"],
    )

    with pytest.raises(ValueError, match="Requested more labeled synthetic frames"):
        load_synthetic_board_rows(clips_dir=clip_dir, num_positions=2, seed=7)


def test_physical_eval_board_dataset_loads_rectified_boards(tmp_path) -> None:
    project_root = tmp_path
    board_path = project_root / "data" / "physical" / "val" / "boards" / "sample.jpg"
    board_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(board_path), np.full((32, 32, 3), 127, dtype=np.uint8))

    dataset = PhysicalEvalBoardDataset(
        rows=[
            PhysicalEvalBoardRow(
                annotation_id="ann-1",
                board_path="data/physical/val/boards/sample.jpg",
                labels=tuple([0] * 64),
                source_video_id="video-1",
            )
        ],
        image_size=32,
    )

    import pipeline.physical.board_data as board_data

    original_root = board_data._PROJECT_ROOT
    board_data._PROJECT_ROOT = project_root
    try:
        image, targets = dataset[0]
    finally:
        board_data._PROJECT_ROOT = original_root

    assert image.shape == (3, 32, 32)
    assert targets.shape == (64,)


def test_physical_manual_train_board_dataset_loads_rectified_boards(tmp_path) -> None:
    project_root = tmp_path
    board_path = project_root / "data" / "physical" / "train" / "boards" / "sample.jpg"
    board_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(board_path), np.full((32, 32, 3), 127, dtype=np.uint8))

    dataset = PhysicalManualTrainBoardDataset(
        rows=[
            PhysicalEvalBoardRow(
                annotation_id="ann-1",
                board_path="data/physical/train/boards/sample.jpg",
                labels=tuple([0] * 64),
                source_video_id="video-1",
            )
        ],
        image_size=32,
    )

    import pipeline.physical.board_data as board_data

    original_root = board_data._PROJECT_ROOT
    board_data._PROJECT_ROOT = project_root
    try:
        image, targets = dataset[0]
    finally:
        board_data._PROJECT_ROOT = original_root

    assert image.shape == (3, 32, 32)
    assert targets.shape == (64,)
