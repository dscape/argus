from __future__ import annotations

import cv2
import numpy as np
import torch
from pipeline.physical.square_data import (
    INPUT_SIZE,
    PhysicalEvalRow,
    PhysicalEvalSquareDataset,
    PhysicalSyntheticSquareDataset,
    preprocess_square_image,
    split_rectified_board_into_squares,
)


def test_preprocess_square_image_outputs_normalized_chw() -> None:
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    tensor = preprocess_square_image(image)

    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == torch.float32


def test_split_rectified_board_into_squares_returns_64_crops() -> None:
    board = np.zeros((160, 160, 3), dtype=np.uint8)
    crops = split_rectified_board_into_squares(board)

    assert len(crops) == 64
    assert crops[0].shape == (20, 20, 3)


def test_physical_synthetic_square_dataset_shape() -> None:
    dataset = PhysicalSyntheticSquareDataset(num_samples_per_class=1, image_size=64, seed=7)

    tensor, label = dataset[0]

    assert len(dataset) == 13
    assert tensor.shape == (3, 64, 64)
    assert label == 0


def test_physical_eval_square_dataset_loads_manifest_rows(tmp_path) -> None:
    project_root = tmp_path
    crop_path = project_root / "data" / "physical" / "eval" / "squares" / "sample_a8.jpg"
    crop_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(crop_path), np.full((16, 16, 3), 127, dtype=np.uint8))

    dataset = PhysicalEvalSquareDataset(
        rows=[
            PhysicalEvalRow(
                annotation_id="ann-1",
                crop_path="data/physical/eval/squares/sample_a8.jpg",
                label_index=4,
                label_name="R",
                source_video_id="video-1",
                square_index=0,
            )
        ],
        image_size=32,
    )

    # Patch module project root after construction so __getitem__ resolves into tmp_path.
    import pipeline.physical.square_data as square_data

    original_root = square_data._PROJECT_ROOT
    square_data._PROJECT_ROOT = project_root
    try:
        tensor, label = dataset[0]
    finally:
        square_data._PROJECT_ROOT = original_root

    assert tensor.shape == (3, 32, 32)
    assert label == 4
