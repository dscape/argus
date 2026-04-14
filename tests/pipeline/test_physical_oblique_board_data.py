from __future__ import annotations

import numpy as np
from pipeline.physical.oblique_board_data import (
    extract_oblique_board_crop,
    preprocess_oblique_board_image,
)


def test_extract_oblique_board_crop_returns_relative_corners() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    corners = ((30.0, 20.0), (90.0, 20.0), (90.0, 80.0), (30.0, 80.0))

    crop = extract_oblique_board_crop(image, corners, crop_margin=0.1)

    assert crop.image_bgr.shape == (72, 72, 3)
    assert np.allclose(
        crop.corners,
        np.array([[6.0, 6.0], [66.0, 6.0], [66.0, 66.0], [6.0, 66.0]], dtype=np.float32),
    )


def test_preprocess_oblique_board_image_returns_square_tensor_and_scaled_corners() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    corners = ((30.0, 20.0), (90.0, 20.0), (90.0, 80.0), (30.0, 80.0))

    tensor, scaled_corners = preprocess_oblique_board_image(
        image,
        corners,
        size=36,
        crop_margin=0.1,
    )

    assert tensor.shape == (3, 36, 36)
    assert scaled_corners.shape == (4, 2)
    assert float(scaled_corners.min().item()) >= 0.0
    assert float(scaled_corners.max().item()) <= 36.0
