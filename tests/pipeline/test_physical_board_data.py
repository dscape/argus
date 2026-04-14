from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from PIL import Image
from pipeline.physical.board_data import (
    INPUT_SIZE,
    PhysicalEvalBoardDataset,
    PhysicalEvalBoardRow,
    PhysicalManualTrainBoardDataset,
    PhysicalSyntheticBoardDataset,
    _apply_piece_rectification_artifacts,
    augment_physical_board_image,
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


def test_augment_physical_board_image_preserves_shape_and_dtype() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)

    augmented = augment_physical_board_image(image, random.Random(7))

    assert augmented.shape == image.shape
    assert augmented.dtype == np.uint8


def test_apply_piece_rectification_artifacts_preserves_rgba_shape() -> None:
    piece_layer = np.zeros((64, 64, 4), dtype=np.uint8)
    piece_layer[16:48, 28:36, :3] = 255
    piece_layer[16:48, 28:36, 3] = 255

    distorted = _apply_piece_rectification_artifacts(
        Image.fromarray(piece_layer, "RGBA"),
        random.Random(7),
    )

    assert distorted.size == (64, 64)
    assert np.array(distorted).shape == piece_layer.shape


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
