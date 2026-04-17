from __future__ import annotations

import numpy as np
from pipeline.physical.square_crop import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    extract_all_occupancy_crops,
    extract_all_piece_crops,
    extract_occupancy_crop,
    extract_piece_crop,
)

# Axis-aligned 80x80 board inside a 128x128 image: each square is 10 px.
# board (col=0, row=0) -> image (10, 20); board (col=8, row=8) -> image (90, 100).
_CORNERS = ((10.0, 20.0), (90.0, 20.0), (90.0, 100.0), (10.0, 100.0))


def _blank_image(size: int = 128) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _paint_marker(
    image: np.ndarray, *, center_xy: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cx, cy = center_xy
    x0 = max(cx - 1, 0)
    y0 = max(cy - 1, 0)
    x1 = min(cx + 2, image.shape[1])
    y1 = min(cy + 2, image.shape[0])
    image[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)


def _has_color(image: np.ndarray, *, channel: int, threshold: int = 100) -> bool:
    other_channels = [c for c in range(image.shape[2]) if c != channel]
    return bool(
        np.any(
            (image[:, :, channel] >= threshold)
            & np.all(image[:, :, other_channels] < threshold // 2, axis=-1)
        )
    )


def _has_color_in_region(
    image: np.ndarray,
    *,
    channel: int,
    x_range: tuple[float, float],
    threshold: int = 100,
) -> bool:
    x_lo = int(round(x_range[0] * image.shape[1]))
    x_hi = int(round(x_range[1] * image.shape[1]))
    return _has_color(image[:, x_lo:x_hi], channel=channel, threshold=threshold)


def test_extract_occupancy_crop_has_expected_shape_and_includes_square_center() -> None:
    image = _blank_image()
    # Board (col=3, row=3) center -> image (10 + 3.5 * 10, 20 + 3.5 * 10) = (45, 55).
    _paint_marker(image, center_xy=(45, 55), color=(0, 0, 255))  # red in BGR is channel 2

    crop = extract_occupancy_crop(image, _CORNERS, row=3, col=3)

    assert crop.shape == (DEFAULT_OCCUPANCY_CROP_SIZE, DEFAULT_OCCUPANCY_CROP_SIZE, 3)
    assert _has_color(crop, channel=2)


def test_extract_occupancy_crop_pads_symmetrically() -> None:
    image = _blank_image()
    # Mark a point 0.3 squares above rank 8 on file a: image (10, 20 - 3) = (10, 17).
    # Occupancy pad is 0.5 square -> for (row=0, col=0) the crop covers board y in [-0.5, 1.5].
    # That includes the marker (at board row = -0.3).
    _paint_marker(image, center_xy=(10, 17), color=(0, 0, 255))

    crop = extract_occupancy_crop(image, _CORNERS, row=0, col=0)

    assert _has_color(crop, channel=2)


def test_extract_piece_crop_has_expected_shape() -> None:
    image = _blank_image()
    crop = extract_piece_crop(image, _CORNERS, row=0, col=0)
    assert crop.shape == (DEFAULT_PIECE_CROP_SIZE, DEFAULT_PIECE_CROP_SIZE, 3)


def test_extract_piece_crop_far_rank_includes_overhang_above_rank_8() -> None:
    image = _blank_image()
    # 0.3 squares above rank 8 on file a: image (10, 20 - 3) = (10, 17).
    # For (row=0, col=0) top_ext=3 covers board y in [-3, 1] -> includes the marker.
    _paint_marker(image, center_xy=(10, 17), color=(0, 0, 255))

    crop = extract_piece_crop(image, _CORNERS, row=0, col=0)

    assert _has_color(crop, channel=2)


def test_extract_piece_crop_near_rank_excludes_same_overhang() -> None:
    image = _blank_image()
    # Same marker at image (10, 17) - well above the board.
    _paint_marker(image, center_xy=(10, 17), color=(0, 0, 255))

    # For (row=7, col=0) top_ext=1 covers board y in [6, 8] -> image y in [80, 100].
    # The marker at y=17 is outside.
    crop = extract_piece_crop(image, _CORNERS, row=7, col=0)

    assert not _has_color(crop, channel=2)


def test_extract_piece_crop_flips_left_half_so_extension_lands_right_of_square() -> None:
    # Marker 0.5 squares to the LEFT of file a on rank 8: image (5, 25) = board (-0.5, 0.5).
    # For (row=0, col=0) the local-canvas layout has the square on the LEFT half and the
    # left-side extension on the RIGHT half AFTER the horizontal flip. The local canvas
    # (96 px wide) is pasted at the bottom-left of the 192 px big canvas, so the extension
    # region spans big-canvas x fraction (0.25, 0.5). The target-square region spans (0.0,
    # 0.25), and (0.5, 1.0) is zero-padded.
    image = _blank_image()
    _paint_marker(image, center_xy=(5, 25), color=(0, 0, 255))

    crop = extract_piece_crop(image, _CORNERS, row=0, col=0)

    assert _has_color_in_region(crop, channel=2, x_range=(0.25, 0.5))
    assert not _has_color_in_region(crop, channel=2, x_range=(0.0, 0.25))
    assert not _has_color_in_region(crop, channel=2, x_range=(0.5, 1.0))


def test_extract_piece_crop_right_half_keeps_extension_right_of_square() -> None:
    # Marker 0.5 squares to the RIGHT of file h on rank 8: image (95, 25) = board (8.5, 0.5).
    # For (row=0, col=7) no flip is applied. The target square maps to local x in [0, 48]
    # and the right-side extension to [48, 96], pasted at the bottom-left of the 192 px big
    # canvas. The marker in the extension thus lands at big-canvas x fraction (0.25, 0.5).
    image = _blank_image()
    _paint_marker(image, center_xy=(95, 25), color=(0, 0, 255))

    crop = extract_piece_crop(image, _CORNERS, row=0, col=7)

    assert _has_color_in_region(crop, channel=2, x_range=(0.25, 0.5))
    assert not _has_color_in_region(crop, channel=2, x_range=(0.0, 0.25))
    assert not _has_color_in_region(crop, channel=2, x_range=(0.5, 1.0))


def test_extract_piece_crop_zero_pads_unused_canvas_area() -> None:
    # Fill the whole image with a non-zero color so any in-crop pixel would also
    # be non-zero. Zero-padded regions of the canvas must remain strictly zero.
    image = np.full((128, 128, 3), 200, dtype=np.uint8)

    # Middle square (row=3, col=4): right_ext is small (0.25), top_ext ~1.57.
    crop = extract_piece_crop(image, _CORNERS, row=3, col=4)

    # Top-right corner pixel of the canvas lies outside both extensions,
    # so it must be the zero-padded background.
    assert tuple(int(value) for value in crop[0, -1]) == (0, 0, 0)


def test_extract_all_occupancy_crops_returns_64_crops() -> None:
    image = _blank_image()
    crops = extract_all_occupancy_crops(image, _CORNERS)
    assert len(crops) == 64
    assert all(
        crop.shape == (DEFAULT_OCCUPANCY_CROP_SIZE, DEFAULT_OCCUPANCY_CROP_SIZE, 3)
        for crop in crops
    )


def test_extract_all_piece_crops_returns_64_crops() -> None:
    image = _blank_image()
    crops = extract_all_piece_crops(image, _CORNERS)
    assert len(crops) == 64
    assert all(
        crop.shape == (DEFAULT_PIECE_CROP_SIZE, DEFAULT_PIECE_CROP_SIZE, 3) for crop in crops
    )
