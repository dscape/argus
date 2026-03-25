"""Tests for pipeline.overlay.scanner grid detection functions.

Verifies that the vectorized implementations of compute_grid_regularity,
check_alternating_pattern, and detect_overlay_in_frame produce correct
results on synthetic board images.
"""

import numpy as np
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
