from __future__ import annotations

import numpy as np
from pipeline.overlay.board_crop import find_board_grid_in_crop
from pipeline.overlay.grid_detector import GridResult


def _make_inset_checkerboard(
    *,
    image_side: int = 400,
    board_side: int = 384,
    border: int = 8,
) -> np.ndarray:
    image = np.full((image_side, image_side, 3), 120, dtype=np.uint8)
    square_side = board_side // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 80
            y1 = border + row * square_side
            x1 = border + col * square_side
            image[y1 : y1 + square_side, x1 : x1 + square_side] = color
    return image


def test_find_board_grid_in_crop_returns_strict_grid_when_available(monkeypatch) -> None:
    sentinel = GridResult(
        v_lines=[12, 108, 204, 300, 396, 492, 588, 684, 780],
        h_lines=[6, 102, 198, 294, 390, 486, 582, 678, 774],
        sq_size=96,
    )

    def fake_detect_grid(_image: np.ndarray, allow_uniform: bool = True) -> GridResult | None:
        return sentinel if not allow_uniform else None

    monkeypatch.setattr("pipeline.overlay.board_crop.detect_grid", fake_detect_grid)

    result = find_board_grid_in_crop(np.zeros((786, 789, 3), dtype=np.uint8))

    assert result is sentinel


def test_find_board_grid_in_crop_localizes_inset_square_when_strict_detection_fails(
    monkeypatch,
) -> None:
    image = _make_inset_checkerboard()

    def fake_detect_grid(_image: np.ndarray, allow_uniform: bool = True) -> GridResult | None:
        return None

    monkeypatch.setattr("pipeline.overlay.board_crop.detect_grid", fake_detect_grid)

    result = find_board_grid_in_crop(image)

    assert result is not None
    assert abs(result.v_lines[0] - 8) <= 1
    assert abs(result.h_lines[0] - 8) <= 1
    assert abs(result.v_lines[-1] - 392) <= 1
    assert abs(result.h_lines[-1] - 392) <= 1
    assert abs(result.sq_size - 48) <= 1
