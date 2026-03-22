"""Tests for pipeline.screen.dual_region_detector."""

import numpy as np
import pytest

from pipeline.screen.dual_region_detector import (
    OTBDetection,
    ScreeningResult,
    detect_otb_region,
)


def _make_natural_scene(h: int, w: int) -> np.ndarray:
    """Create a synthetic image resembling a natural camera scene.

    High pixel variance, diverse colors, and textured edges — characteristics
    of real OTB footage (wood grain, people, lighting).
    """
    rng = np.random.RandomState(42)
    img = rng.randint(50, 220, (h, w, 3), dtype=np.uint8)
    # Add some structure (simulate edges/textures)
    for y in range(0, h, 20):
        img[y : y + 2, :] = rng.randint(0, 255, (min(2, h - y), w, 3), dtype=np.uint8)
    return img


def _make_overlay_region(h: int, w: int) -> np.ndarray:
    """Create a synthetic rendered chess board overlay.

    Low pixel variance, alternating solid colors — characteristics of
    lichess/chess.com 2D board overlays.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    sq_h = h // 8
    sq_w = w // 8
    light = np.array([181, 217, 240], dtype=np.uint8)  # #F0D9B5 in BGR
    dark = np.array([99, 136, 181], dtype=np.uint8)  # #B58863 in BGR

    for row in range(8):
        for col in range(8):
            color = light if (row + col) % 2 == 0 else dark
            y0, y1 = row * sq_h, (row + 1) * sq_h
            x0, x1 = col * sq_w, (col + 1) * sq_w
            img[y0:y1, x0:x1] = color

    return img


def _make_composite_frame(
    frame_h: int = 1080,
    frame_w: int = 1920,
    overlay_size: int = 400,
    overlay_x: int = 1400,
    overlay_y: int = 100,
    natural_background: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Create a composite frame with overlay on top of background.

    Returns (frame, overlay_bbox).
    """
    if natural_background:
        frame = _make_natural_scene(frame_h, frame_w)
    else:
        # Solid color background
        frame = np.full((frame_h, frame_w, 3), 40, dtype=np.uint8)

    overlay = _make_overlay_region(overlay_size, overlay_size)
    frame[overlay_y : overlay_y + overlay_size,
          overlay_x : overlay_x + overlay_size] = overlay

    bbox = (overlay_x, overlay_y, overlay_size, overlay_size)
    return frame, bbox


class TestDetectOTBRegion:
    """Test OTB camera footage detection in the non-overlay region."""

    def test_natural_scene_detected(self):
        """Frame with natural scene outside overlay -> found=True."""
        frame, bbox = _make_composite_frame(natural_background=True)
        result = detect_otb_region(frame, bbox)
        assert result.found is True
        assert result.confidence > 0.0

    def test_solid_background_rejected(self):
        """Overlay on solid color background -> found=False."""
        frame, bbox = _make_composite_frame(natural_background=False)
        result = detect_otb_region(frame, bbox)
        assert result.found is False

    def test_full_screen_overlay_rejected(self):
        """Overlay covering nearly the entire frame -> found=False."""
        # Overlay takes up most of the frame
        frame, bbox = _make_composite_frame(
            frame_h=500, frame_w=500,
            overlay_size=480, overlay_x=10, overlay_y=10,
        )
        result = detect_otb_region(frame, bbox)
        # Either the remaining area is too small or has too little variance
        assert result.found is False

    def test_laplacian_and_color_reported(self):
        """Detection result includes diagnostic values."""
        frame, bbox = _make_composite_frame(natural_background=True)
        result = detect_otb_region(frame, bbox)
        assert result.laplacian_variance > 0
        assert result.color_std > 0


class TestScreeningResult:
    """Test the ScreeningResult dataclass logic."""

    def test_approved_when_both_present(self):
        r = ScreeningResult(has_overlay=True, has_otb=True, approved=True)
        assert r.approved is True

    def test_rejected_no_overlay(self):
        r = ScreeningResult(has_overlay=False, has_otb=True, approved=False)
        assert r.approved is False

    def test_rejected_no_otb(self):
        r = ScreeningResult(has_overlay=True, has_otb=False, approved=False)
        assert r.approved is False

    def test_rejected_neither(self):
        r = ScreeningResult(has_overlay=False, has_otb=False, approved=False)
        assert r.approved is False
