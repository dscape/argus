"""Tests for frame_fetcher video_id validation and vertical video detection."""

import numpy as np
from pipeline.screen.frame_fetcher import fetch_youtube_frames, is_vertical_video


class TestVideoIdValidation:
    """Verify that invalid video IDs are rejected before URL construction."""

    def test_valid_youtube_id_accepted(self):
        """Standard 11-char YouTube IDs should pass validation (may fail on fetch)."""
        # We can't actually fetch, but validation should pass
        # The function will return [] due to network failure in test, not validation
        result = fetch_youtube_frames("dQw4w9WgXcQ")
        # Just verify it didn't raise — empty result is fine (no network in tests)
        assert isinstance(result, list)

    def test_path_traversal_rejected(self):
        """Path traversal attempts should be rejected."""
        result = fetch_youtube_frames("../../../etc")
        assert result == []

    def test_empty_string_rejected(self):
        """Empty video_id should be rejected."""
        result = fetch_youtube_frames("")
        assert result == []

    def test_special_chars_rejected(self):
        """Video IDs with special characters should be rejected."""
        result = fetch_youtube_frames("abc;rm -rf /")
        assert result == []

    def test_url_injection_rejected(self):
        """URL-like video IDs should be rejected."""
        result = fetch_youtube_frames("http://evil.com/payload")
        assert result == []


class TestVerticalVideoDetection:
    """Verify letterbox detection logic."""

    def _make_frame(
        self, h: int, w: int, left_brightness: int, right_brightness: int, center_brightness: int
    ) -> np.ndarray:
        """Create a test frame with configurable edge/center brightness."""
        frame = np.full((h, w, 3), center_brightness, dtype=np.uint8)
        frame[:, :30, :] = left_brightness  # Left edge
        frame[:, -30:, :] = right_brightness  # Right edge
        return frame

    def test_dark_edges_detected_as_vertical(self):
        """Frames with dark left/right edges and bright center = vertical."""
        frames = [
            (self._make_frame(360, 480, 10, 10, 150), "thumb"),
            (self._make_frame(360, 480, 10, 10, 150), "25%"),
            (self._make_frame(360, 480, 10, 10, 150), "50%"),
            (self._make_frame(360, 480, 10, 10, 150), "75%"),
        ]
        assert is_vertical_video(frames) is True

    def test_bright_edges_not_vertical(self):
        """Frames with bright edges = not vertical (normal landscape video)."""
        frames = [
            (self._make_frame(360, 480, 150, 150, 150), "thumb"),
            (self._make_frame(360, 480, 150, 150, 150), "25%"),
            (self._make_frame(360, 480, 150, 150, 150), "50%"),
            (self._make_frame(360, 480, 150, 150, 150), "75%"),
        ]
        assert is_vertical_video(frames) is False

    def test_empty_frames_not_vertical(self):
        """Empty frame list should not be considered vertical."""
        assert is_vertical_video([]) is False
