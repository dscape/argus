from __future__ import annotations

import cv2
import numpy as np
import torch
from pipeline.physical.board_data import (
    INPUT_SIZE,
    PhysicalEvalBoardDataset,
    PhysicalEvalBoardRow,
    PhysicalSyntheticBoardDataset,
    preprocess_board_image,
)


def test_preprocess_board_image_outputs_normalized_chw() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    tensor = preprocess_board_image(image)

    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == torch.float32


def test_physical_synthetic_board_dataset_returns_board_and_targets() -> None:
    dataset = PhysicalSyntheticBoardDataset(num_positions=2, image_size=64, seed=7)

    image, targets = dataset[0]

    assert image.shape == (3, 64, 64)
    assert targets.shape == (64,)


def test_physical_eval_board_dataset_loads_rectified_boards(tmp_path) -> None:
    project_root = tmp_path
    board_path = project_root / "data" / "physical" / "eval" / "boards" / "sample.jpg"
    board_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(board_path), np.full((32, 32, 3), 127, dtype=np.uint8))

    dataset = PhysicalEvalBoardDataset(
        rows=[
            PhysicalEvalBoardRow(
                annotation_id="ann-1",
                board_path="data/physical/eval/boards/sample.jpg",
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
