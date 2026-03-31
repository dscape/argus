"""Tests for pipeline.overlay.scanner grid detection functions.

Verifies that the vectorized implementations of compute_grid_regularity,
check_alternating_pattern, and detect_overlay_in_frame produce correct
results on synthetic board images.

Also includes regression tests for specific video clips that previously
produced incorrect overlay detections.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from pipeline.overlay.grid_detector import detect_grid
from pipeline.overlay.scanner import (
    MIN_LOW_VARIANCE_RATIO,
    OverlayDetection,
    check_alternating_pattern,
    compute_grid_regularity,
    detect_overlay_in_frame,
)


def _make_rendered_board(size: int = 256, light: int = 240, dark: int = 100) -> np.ndarray:
    """Create a perfect 8x8 rendered chess board (grayscale).

    Solid-fill squares with zero intra-cell variance — the ideal
    input for overlay detection.
    """
    board = np.zeros((size, size), dtype=np.uint8)
    sq = size // 8
    for row in range(8):
        for col in range(8):
            val = light if (row + col) % 2 == 0 else dark
            board[row * sq : (row + 1) * sq, col * sq : (col + 1) * sq] = val
    return board


def _make_rendered_board_bgr(size: int = 256) -> np.ndarray:
    """Create a perfect 8x8 rendered chess board (BGR)."""
    board = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8
    light = np.array([181, 217, 240], dtype=np.uint8)
    dark = np.array([99, 136, 181], dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            color = light if (row + col) % 2 == 0 else dark
            board[row * sq : (row + 1) * sq, col * sq : (col + 1) * sq] = color
    return board


def _make_noise_image(size: int = 256, seed: int = 42) -> np.ndarray:
    """Create a random noise image (grayscale) — high variance per cell."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (size, size), dtype=np.uint8)


# ---------------------------------------------------------------------------
# compute_grid_regularity
# ---------------------------------------------------------------------------


class TestComputeGridRegularity:
    """Test the vectorized compute_grid_regularity function."""

    def test_perfect_board_high_score(self):
        """A perfect rendered board should have regularity near 1.0."""
        board = _make_rendered_board(256)
        score = compute_grid_regularity(board)
        assert score >= 0.9, f"Expected >= 0.9, got {score}"

    def test_noise_image_low_score(self):
        """Random noise should have low regularity (high variance per cell)."""
        noise = _make_noise_image(256)
        score = compute_grid_regularity(noise)
        assert score < MIN_LOW_VARIANCE_RATIO, f"Expected < {MIN_LOW_VARIANCE_RATIO}, got {score}"

    def test_bgr_input_matches_gray(self):
        """BGR and grayscale inputs should give equivalent results."""
        board_bgr = _make_rendered_board_bgr(256)
        import cv2

        board_gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
        score_bgr = compute_grid_regularity(board_bgr)
        score_gray = compute_grid_regularity(board_gray)
        assert abs(score_bgr - score_gray) < 1e-6

    def test_empty_region_returns_zero(self):
        """Empty array returns 0.0."""
        empty = np.array([], dtype=np.uint8).reshape(0, 0)
        assert compute_grid_regularity(empty) == 0.0

    def test_too_small_region_returns_zero(self):
        """Region smaller than 32x32 returns 0.0."""
        small = np.zeros((16, 16), dtype=np.uint8)
        assert compute_grid_regularity(small) == 0.0

    def test_uniform_image_high_score(self):
        """A uniform (solid color) image has zero variance — all cells pass."""
        uniform = np.full((256, 256), 128, dtype=np.uint8)
        score = compute_grid_regularity(uniform)
        assert score == 1.0

    def test_score_range(self):
        """Score should always be in [0.0, 1.0]."""
        for seed in range(5):
            rng = np.random.RandomState(seed)
            img = rng.randint(0, 256, (128, 128), dtype=np.uint8)
            score = compute_grid_regularity(img)
            assert 0.0 <= score <= 1.0

    def test_different_sizes(self):
        """Works correctly for various board sizes."""
        for size in [32, 64, 128, 256, 480]:
            board = _make_rendered_board(size)
            score = compute_grid_regularity(board)
            assert score >= 0.9, f"Size {size}: expected >= 0.9, got {score}"

    def test_non_square_region(self):
        """Works on non-square regions (uses integer division for cell size)."""
        board = _make_rendered_board(256)
        # Take a non-square crop
        region = board[:200, :256]
        score = compute_grid_regularity(region)
        # Should still find some low-variance cells
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# check_alternating_pattern
# ---------------------------------------------------------------------------


class TestCheckAlternatingPattern:
    """Test the vectorized check_alternating_pattern function."""

    def test_rendered_board_has_pattern(self):
        """A perfect chess board should show alternating pattern."""
        board = _make_rendered_board(256)
        assert check_alternating_pattern(board) is True

    def test_uniform_image_no_pattern(self):
        """A uniform image has no alternation."""
        uniform = np.full((256, 256), 128, dtype=np.uint8)
        assert check_alternating_pattern(uniform) is False

    def test_bgr_input_matches_gray(self):
        """BGR and grayscale inputs should give the same result."""
        board_bgr = _make_rendered_board_bgr(256)
        import cv2

        board_gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
        assert check_alternating_pattern(board_bgr) == check_alternating_pattern(board_gray)

    def test_noise_image(self):
        """Random noise may or may not have alternation — just shouldn't crash."""
        noise = _make_noise_image(256)
        result = check_alternating_pattern(noise)
        assert isinstance(result, bool)

    def test_small_contrast_no_pattern(self):
        """Board with very small contrast between light and dark squares."""
        board = _make_rendered_board(256, light=130, dark=128)
        # Difference of 2 is well below the 15 threshold
        assert check_alternating_pattern(board) is False

    def test_different_sizes(self):
        """Works on various sizes."""
        for size in [32, 64, 128, 256]:
            board = _make_rendered_board(size)
            assert check_alternating_pattern(board) is True, f"Size {size}: expected True"


# ---------------------------------------------------------------------------
# detect_overlay_in_frame
# ---------------------------------------------------------------------------


class TestDetectOverlayInFrame:
    """Test the full overlay detection pipeline on synthetic frames."""

    def test_frame_with_overlay_detected(self):
        """A frame containing a rendered board overlay should be detected."""
        # Natural background with overlay
        rng = np.random.RandomState(42)
        frame = rng.randint(50, 200, (360, 480, 3), dtype=np.uint8)
        overlay = _make_rendered_board_bgr(200)
        frame[80:280, 140:340] = overlay

        result = detect_overlay_in_frame(frame)
        assert result.found is True
        assert result.score > MIN_LOW_VARIANCE_RATIO
        assert result.bbox is not None
        assert result.frame_resolution == (480, 360)

    def test_natural_scene_no_overlay(self):
        """A natural scene without overlay should not be detected."""
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (360, 480, 3), dtype=np.uint8)
        result = detect_overlay_in_frame(frame)
        assert result.found is False
        assert result.frame_resolution == (480, 360)

    def test_full_screen_board_detected(self):
        """A full-screen rendered board should be detected."""
        board = _make_rendered_board_bgr(360)
        # Pad to 480 wide
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        frame[:, 60:420] = board

        result = detect_overlay_in_frame(frame)
        assert result.found is True
        assert result.score > MIN_LOW_VARIANCE_RATIO

    def test_result_dataclass_fields(self):
        """OverlayDetection has expected fields."""
        det = OverlayDetection(found=False, frame_resolution=(480, 360))
        assert det.found is False
        assert det.bbox is None
        assert det.score == 0.0
        assert det.frame_resolution == (480, 360)

    def test_overlay_with_different_size(self):
        """Detection works with different overlay sizes relative to frame."""
        rng = np.random.RandomState(99)
        frame = rng.randint(40, 200, (360, 480, 3), dtype=np.uint8)
        overlay = _make_rendered_board_bgr(160)
        frame[100:260, 160:320] = overlay

        result = detect_overlay_in_frame(frame)
        assert result.found is True
        assert result.bbox is not None


# ---------------------------------------------------------------------------
# Regression: real video clip overlay detection
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frames"


class TestOverlayDetectionRegression:
    """Regression tests for video clips that previously produced wrong detections.

    clip 21 (vkoTN5DxRS0): Full-board overlay on the left side. The seed
        detection found a sub-region but expansion failed to grow to the
        full board.  Fixed by adding Gaussian blur before grid detection
        in expansion and extending the multiplier range.

    clip 42 (Unu6antTBGs): Chess24 Norway Chess layout with rendered board
        on the left and camera views on the right.  The scanner incorrectly
        picked the camera views as the largest candidate.  Fixed by trying
        expansion from diverse seeds (including highest-scoring) and
        disabling the uniform grid fallback during expansion.

    clip 44 (ycitHs8_NY4): Low-res (640×360) OTB footage with a small
        chess.com green overlay on a monitor between the players.  The
        scanner must detect this small rendered overlay despite the
        challenging resolution and competing physical board.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_fixtures(self):
        """Skip if regression frames haven't been extracted."""
        needed = [
            FIXTURES_DIR / "vkoTN5DxRS0_t60.png",
            FIXTURES_DIR / "Unu6antTBGs_t60.png",
            FIXTURES_DIR / "ycitHs8_NY4_t170.png",
        ]
        for p in needed:
            if not p.exists():
                pytest.skip(f"Regression fixture not found: {p.name}")

    def test_vko_full_board_detected(self) -> None:
        """vkoTN5DxRS0: expansion must grow to the full board (~1056px)."""
        frame = cv2.imread(str(FIXTURES_DIR / "vkoTN5DxRS0_t60.png"))
        assert frame is not None

        det = detect_overlay_in_frame(frame)
        assert det.found, "vko: overlay should be found"
        x, y, w, h = det.bbox
        # Board should be large — at least 900px wide
        assert w >= 900, f"vko: bbox width {w} too small, expected >= 900"
        assert h >= 900, f"vko: bbox height {h} too small, expected >= 900"
        # Board should be on the left side of the frame
        assert x < 200, f"vko: bbox x={x} too far right, expected < 200"

        # Verify the crop produces a valid grid with 64 non-empty squares
        crop = frame[y : y + h, x : x + w]
        grid = detect_grid(crop)
        assert grid is not None, "vko: detect_grid on crop returned None"
        assert len(grid.v_lines) == 9
        assert len(grid.h_lines) == 9
        squares = grid.crop_squares(crop)
        for r in range(8):
            for c in range(8):
                assert squares[r][c].size > 0, f"vko: square ({r},{c}) is empty"

    def test_unu_board_not_camera_views(self) -> None:
        """Unu6antTBGs: detection must find the rendered board, not camera views."""
        frame = cv2.imread(str(FIXTURES_DIR / "Unu6antTBGs_t60.png"))
        assert frame is not None

        det = detect_overlay_in_frame(frame)
        assert det.found, "Unu: overlay should be found"
        x, y, w, h = det.bbox
        # The rendered board is in the left portion of the frame.
        # Camera views are at x > 560.  The bbox must start before
        # the camera view boundary.
        assert x < 300, f"Unu: bbox x={x} too far right (camera view area)"
        # Board should be at least 400px wide
        assert w >= 400, f"Unu: bbox width {w} too small, expected >= 400"

        # Verify a grid can be detected within the crop
        crop = frame[y : y + h, x : x + w]
        grid = detect_grid(crop)
        assert grid is not None, "Unu: detect_grid on crop returned None"
        assert len(grid.v_lines) == 9
        assert len(grid.h_lines) == 9

    def test_yci_chesscom_overlay_detected(self) -> None:
        """ycitHs8_NY4 t=170: small chess.com green overlay must be detected.

        The overlay is on a monitor between the players.  Despite the low
        resolution (640×360) and the presence of a physical chess board,
        the scanner should find the rendered overlay.
        """
        frame = cv2.imread(str(FIXTURES_DIR / "ycitHs8_NY4_t170.png"))
        assert frame is not None

        det = detect_overlay_in_frame(frame)
        assert det.found, "yci: chess.com overlay should be detected at t=170"
        assert det.bbox is not None
        x, y, w, h = det.bbox
        # The overlay is small at this resolution — should be 100-200px
        assert w >= 100, f"yci: bbox width {w} too small"
        assert h >= 100, f"yci: bbox height {h} too small"
