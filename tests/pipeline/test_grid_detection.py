"""Tests for pipeline.overlay.grid_detector — exact grid detection on reference frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pipeline.overlay.grid_detector import GridResult, detect_grid, find_board_in_frame

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
