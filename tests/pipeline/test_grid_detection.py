"""Tests for pipeline.overlay.grid_detector — exact grid detection on reference videos."""

import cv2
import numpy as np
import pytest
from pipeline.overlay.grid_detector import GridResult, detect_grid, find_board_in_frame

# ---------------------------------------------------------------------------
# Video paths & ground-truth grid positions
# ---------------------------------------------------------------------------

VIDEO_DIR = "data/videos/STLChessClub"

# Manually verified grid coordinates (9 vertical + 9 horizontal boundaries).
# For O8Z / 7Ra / 2wW the overlay occupies the left ~1080×1080 of the 1920×1080 frame.
# For Ov8 the board is a small overlay on the right side of the frame.
GRID_TRUTH: dict[str, dict] = {
    "O8Z": {
        "video": f"{VIDEO_DIR}/O8ZwstOxG_A.mp4",
        "timestamp": 60,
        "v_lines": [56, 188, 320, 452, 584, 716, 848, 980, 1112],
        "h_lines": [10, 142, 274, 406, 538, 670, 802, 934, 1066],
        "sq_size": 132,
        "tolerance": 8,
    },
    "7Ra": {
        "video": f"{VIDEO_DIR}/7RaBQag34Hk.mp4",
        "timestamp": 60,
        "v_lines": [56, 188, 320, 452, 584, 716, 848, 980, 1112],
        "h_lines": [10, 142, 274, 406, 538, 670, 802, 934, 1066],
        "sq_size": 132,
        "tolerance": 8,
    },
    "2wW": {
        "video": f"{VIDEO_DIR}/2wWUKmCBr6A.mp4",
        "timestamp": 60,
        "v_lines": [56, 188, 320, 452, 584, 716, 848, 980, 1112],
        "h_lines": [10, 142, 274, 406, 538, 670, 802, 934, 1066],
        "sq_size": 132,
        "tolerance": 8,
    },
    "Ov8": {
        "video": f"{VIDEO_DIR}/Ov8PXnJp1PU.mp4",
        "timestamp": 60,
        "v_lines": [1279, 1352, 1425, 1498, 1571, 1644, 1717, 1790, 1863],
        "h_lines": [219, 292, 365, 438, 511, 584, 657, 730, 803],
        "sq_size": 73,
        "tolerance": 8,
    },
}


def _extract_frame(video_path: str, timestamp: int) -> np.ndarray:
    """Extract a single frame from *video_path* at *timestamp* seconds."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Failed to read frame at {timestamp}s from {video_path}"
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
    """Grid detection must find the exact 8×8 grid in each reference video."""

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW", "Ov8"])
    def test_find_board_in_frame(self, name: str) -> None:
        """find_board_in_frame returns correct grid for each video."""
        info = GRID_TRUTH[name]
        frame = _extract_frame(info["video"], info["timestamp"])

        result = find_board_in_frame(frame)

        assert result is not None, f"{name}: find_board_in_frame returned None"
        _assert_grid_close(
            result,
            info["v_lines"],
            info["h_lines"],
            info["sq_size"],
            info["tolerance"],
            name,
        )

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW"])
    def test_detect_grid_on_overlay_crop(self, name: str) -> None:
        """detect_grid works directly on the overlay crop for left-side boards."""
        info = GRID_TRUTH[name]
        frame = _extract_frame(info["video"], info["timestamp"])
        overlay = frame[0:1080, 0:1080]

        result = detect_grid(overlay)

        assert result is not None, f"{name}: detect_grid on overlay crop returned None"
        assert len(result.v_lines) == 9
        assert len(result.h_lines) == 9

    def test_grid_produces_64_squares(self) -> None:
        """crop_squares returns an 8×8 list of non-empty images."""
        info = GRID_TRUTH["O8Z"]
        frame = _extract_frame(info["video"], info["timestamp"])
        result = find_board_in_frame(frame)
        assert result is not None

        squares = result.crop_squares(frame)
        assert len(squares) == 8
        for r, row in enumerate(squares):
            assert len(row) == 8, f"row {r}: expected 8 cols, got {len(row)}"
            for c, sq in enumerate(row):
                assert sq.size > 0, f"square ({r},{c}) is empty"
