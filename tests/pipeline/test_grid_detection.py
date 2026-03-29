"""Tests for pipeline.overlay.grid_detector — exact grid detection on reference frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pipeline.overlay.grid_detector import (
    GridResult,
    _clamp_grid_to_image,
    _uniform_grid,
    detect_grid,
    find_board_in_frame,
)

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frames"

with open(FIXTURES_DIR / "ground_truth.json") as _f:
    GROUND_TRUTH: dict[str, dict] = json.load(_f)

# Test-specific thresholds (not part of the shared ground truth).
SQ_SIZES = {"O8Z": 132, "7Ra": 132, "2wW": 132, "Ov8": 73}
TOLERANCE = 8


def _load_frame(entry: dict) -> np.ndarray:
    """Load a pre-extracted frame from the fixtures directory."""
    path = FIXTURES_DIR / entry["image"]
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None, f"Cannot load fixture image: {path}"
    return frame


def _assert_grid_close(
    result: GridResult,
    expected_v: list[int],
    expected_h: list[int],
    expected_sq: int,
    tolerance: int,
    label: str,
) -> None:
    """Assert that *result* matches expected grid within *tolerance* pixels."""
    assert len(result.v_lines) == 9, f"{label}: expected 9 v_lines, got {len(result.v_lines)}"
    assert len(result.h_lines) == 9, f"{label}: expected 9 h_lines, got {len(result.h_lines)}"

    for i, (got, exp) in enumerate(zip(result.v_lines, expected_v)):
        assert abs(got - exp) <= tolerance, (
            f"{label}: v_lines[{i}] = {got}, expected {exp} (±{tolerance})"
        )

    for i, (got, exp) in enumerate(zip(result.h_lines, expected_h)):
        assert abs(got - exp) <= tolerance, (
            f"{label}: h_lines[{i}] = {got}, expected {exp} (±{tolerance})"
        )

    assert abs(result.sq_size - expected_sq) <= tolerance, (
        f"{label}: sq_size = {result.sq_size}, expected {expected_sq} (±{tolerance})"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGridDetection:
    """Grid detection must find the exact 8×8 grid in each reference frame."""

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW", "Ov8"])
    def test_find_board_in_frame(self, name: str) -> None:
        """find_board_in_frame returns correct grid for each frame."""
        info = GROUND_TRUTH[name]
        frame = _load_frame(info)

        result = find_board_in_frame(frame)

        assert result is not None, f"{name}: find_board_in_frame returned None"
        _assert_grid_close(
            result,
            info["grid"]["v_lines"],
            info["grid"]["h_lines"],
            SQ_SIZES[name],
            TOLERANCE,
            name,
        )

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW"])
    def test_detect_grid_on_overlay_crop(self, name: str) -> None:
        """detect_grid works directly on the overlay crop for left-side boards."""
        info = GROUND_TRUTH[name]
        frame = _load_frame(info)
        overlay = frame[0:1080, 0:1080]

        result = detect_grid(overlay)

        assert result is not None, f"{name}: detect_grid on overlay crop returned None"
        assert len(result.v_lines) == 9
        assert len(result.h_lines) == 9

    def test_grid_produces_64_squares(self) -> None:
        """crop_squares returns an 8×8 list of non-empty images."""
        info = GROUND_TRUTH["O8Z"]
        frame = _load_frame(info)
        result = find_board_in_frame(frame)
        assert result is not None

        squares = result.crop_squares(frame)
        assert len(squares) == 8
        for r, row in enumerate(squares):
            assert len(row) == 8, f"row {r}: expected 8 cols, got {len(row)}"
            for c, sq in enumerate(row):
                assert sq.size > 0, f"square ({r},{c}) is empty"


class TestClampGridToImage:
    """Grid clamping must fix lines that extend past the image boundary."""

    def test_h_lines_shifted_when_overshoot_exceeds_half_square(self) -> None:
        """h_lines overshooting by more than half a square get shifted back."""
        # 9 lines with 100px spacing, starting at 120 — last line at 920
        # Image is only 800px tall, overshoot = 920 - 799 = 121 > 50 (half)
        # shift = ceil(121/100)*100 = 200
        grid = GridResult(
            v_lines=list(range(0, 801, 100)),
            h_lines=[120 + i * 100 for i in range(9)],
            sq_size=100,
        )
        result = _clamp_grid_to_image(grid, h=800, w=800)
        # Shift of 200 → first line = 120-200 = -80, clamped to 0
        assert result.h_lines[0] == 0
        assert result.h_lines[1] == 20  # 220 - 200
        assert result.h_lines[-1] == 720  # 920 - 200
        # v_lines untouched (they fit)
        assert result.v_lines == list(range(0, 801, 100))

    def test_h_lines_not_shifted_when_overshoot_under_half_square(self) -> None:
        """Small overshoot (< half square) should NOT trigger a shift."""
        # Last line at 808, image 800px, overshoot = 9 < 50 (half)
        grid = GridResult(
            v_lines=list(range(8, 809, 100)),
            h_lines=list(range(8, 809, 100)),
            sq_size=100,
        )
        original_h = list(grid.h_lines)
        result = _clamp_grid_to_image(grid, h=800, w=800)
        assert result.h_lines == original_h  # no shift

    def test_negative_first_line_clamped_to_zero(self) -> None:
        """Negative first h_line gets clamped to 0."""
        grid = GridResult(
            v_lines=list(range(0, 577, 72)),
            h_lines=[-2, 71, 144, 217, 290, 363, 436, 509, 582],
            sq_size=73,
        )
        result = _clamp_grid_to_image(grid, h=576, w=576)
        assert result.h_lines[0] == 0
        # Rest unchanged (overshoot 582-575=7 < 36 half)
        assert result.h_lines[1] == 71

    def test_v_lines_shifted_when_overshooting(self) -> None:
        """v_lines are clamped the same way as h_lines."""
        grid = GridResult(
            v_lines=[130 + i * 100 for i in range(9)],
            h_lines=list(range(0, 801, 100)),
            sq_size=100,
        )
        result = _clamp_grid_to_image(grid, h=800, w=800)
        # v overshoot = 930 - 799 = 131 > 50 → shift by 200? No: round up
        # (131 + 99) // 100 * 100 = 200.  130 - 200 = -70 → clamped to 0
        assert result.v_lines[0] == 0
        assert result.v_lines[-1] == 730  # 930 - 200

    def test_no_clamping_when_grid_fits(self) -> None:
        """Grid that fits entirely within the image is left unchanged."""
        lines = list(range(0, 801, 100))
        grid = GridResult(v_lines=list(lines), h_lines=list(lines), sq_size=100)
        result = _clamp_grid_to_image(grid, h=810, w=810)
        assert result.v_lines == lines
        assert result.h_lines == lines


class TestUniformGridFallback:
    """Uniform grid fallback for borderless/flat board themes."""

    def test_uniform_grid_on_square_image(self) -> None:
        """Square image produces a valid uniform 8×8 grid."""
        result = _uniform_grid(400, 400)
        assert result is not None
        assert len(result.v_lines) == 9
        assert len(result.h_lines) == 9
        assert result.v_lines[0] == 0
        assert result.v_lines[-1] == 400
        assert result.h_lines[0] == 0
        assert result.h_lines[-1] == 400
        assert result.sq_size == 50

    def test_uniform_grid_rejected_for_wide_image(self) -> None:
        """Non-square image (ratio > 1.3) returns None."""
        result = _uniform_grid(400, 600)
        assert result is None

    def test_uniform_grid_accepted_for_slightly_rectangular(self) -> None:
        """Slightly rectangular image (ratio ≤ 1.3) still works."""
        result = _uniform_grid(520, 400)
        assert result is not None
        assert len(result.v_lines) == 9
        assert len(result.h_lines) == 9

    def test_detect_grid_falls_back_to_uniform(self) -> None:
        """detect_grid returns uniform grid for a plain square image."""
        # Create a plain gray image with no edges — Sobel + Hough will fail
        img = np.full((400, 400, 3), 180, dtype=np.uint8)
        result = detect_grid(img)
        assert result is not None
        assert result.sq_size == 50
        assert result.v_lines[0] == 0
        assert result.v_lines[-1] == 400
