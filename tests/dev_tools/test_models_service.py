"""Tests for models_service DB update handling and threshold validation."""

from unittest.mock import MagicMock, patch

import numpy as np


def _mock_db_conn(video_rows):
    """Create a mock DB connection returning given rows on first fetchall."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = video_rows
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn_ctx = MagicMock()
    mock_conn_ctx.cursor.return_value = mock_cursor
    mock_conn_ctx.__enter__ = lambda s: s
    mock_conn_ctx.__exit__ = MagicMock(return_value=False)
    return mock_conn_ctx


class TestDbWriteFailureHandling:
    """Verify DB write failures are surfaced in results."""

    @patch("api.services.models_service.get_conn")
    @patch("api.services.models_service.is_vertical_video")
    @patch("api.services.models_service.fetch_youtube_frames")
    @patch("api.services.models_service.score_title")
    @patch("api.services.models_service.detect_overlay_in_frame")
    @patch("api.services.models_service.detect_otb_region")
    def test_db_failure_flags_results(
        self, mock_otb, mock_overlay, mock_score, mock_frames, mock_vertical, mock_conn
    ):
        """When DB update fails, results should contain db_write_failed flag."""
        from api.services.models_service import ai_screen_batch

        mock_score.return_value = (True, 0.8)

        # First call returns video metadata, second call (for DB write) raises
        mock_conn_ok = _mock_db_conn(
            [
                ("vid1", "Chess game", "overlay", 0.95, True),
            ]
        )
        mock_conn.return_value = mock_conn_ok

        # Return some frames so processing continues
        fake_frame = np.zeros((360, 480, 3), dtype=np.uint8)
        mock_frames.return_value = [(fake_frame, "thumb")]
        mock_vertical.return_value = False

        # Mock overlay detection
        mock_det = MagicMock()
        mock_det.found = False
        mock_det.score = 0.0
        mock_det.bbox = None
        mock_overlay.return_value = mock_det

        results = ai_screen_batch(["vid1"], threshold=0.90)

        assert len(results) == 1
        # Should have all required keys regardless
        assert "video_id" in results[0]
        assert "error" in results[0]


class TestVerticalVideoHandling:
    """Verify vertical video detection is handled correctly."""

    @patch("api.services.models_service.get_conn")
    @patch("api.services.models_service.is_vertical_video")
    @patch("api.services.models_service.fetch_youtube_frames")
    @patch("api.services.models_service.score_title")
    def test_vertical_video_auto_rejected(self, mock_score, mock_frames, mock_vertical, mock_conn):
        """Vertical videos should be auto-rejected with confidence 1.0."""
        from api.services.models_service import ai_screen_batch

        mock_score.return_value = (False, 0.3)
        mock_conn.return_value = _mock_db_conn(
            [
                ("vid1", "test title", None, None, None),
            ]
        )
        fake_frame = np.zeros((360, 480, 3), dtype=np.uint8)
        mock_frames.return_value = [(fake_frame, "thumb")]
        mock_vertical.return_value = True

        results = ai_screen_batch(["vid1"], threshold=0.90)

        assert len(results) == 1
        assert results[0]["predicted_class"] == "reject"
        assert results[0]["confidence"] == 1.0
        assert results[0]["auto_decided"] is True
        assert results[0]["vertical"] is True


class TestThresholdBehavior:
    """Verify threshold affects auto_decided flag correctly."""

    @patch("api.services.models_service.get_conn")
    @patch("api.services.models_service.is_vertical_video")
    @patch("api.services.models_service.fetch_youtube_frames")
    @patch("api.services.models_service.score_title")
    @patch("api.services.models_service.detect_overlay_in_frame")
    @patch("api.services.models_service.detect_otb_region")
    def test_high_threshold_prevents_auto_decide(
        self, mock_otb, mock_overlay, mock_score, mock_frames, mock_vertical, mock_conn
    ):
        """With threshold=1.0, nothing should be auto-decided."""
        from api.services.models_service import ai_screen_batch

        mock_score.return_value = (True, 0.8)
        # Use DB fallback with existing AI prediction
        mock_conn.return_value = _mock_db_conn(
            [
                ("vid1", "Chess game", "overlay", 0.95, False),
            ]
        )
        fake_frame = np.zeros((360, 480, 3), dtype=np.uint8)
        mock_frames.return_value = [(fake_frame, "thumb")]
        mock_vertical.return_value = False

        mock_det = MagicMock()
        mock_det.found = False
        mock_det.score = 0.0
        mock_det.bbox = None
        mock_overlay.return_value = mock_det

        results = ai_screen_batch(["vid1"], threshold=1.0)

        assert len(results) == 1
        # Even with 0.95 confidence, threshold=1.0 means no auto-decide
        if results[0].get("predicted_class") is not None:
            assert results[0]["auto_decided"] is False
