"""Tests for AI screening batch error handling.

Verifies that ai_screen_batch returns per-video errors instead of raising
a blanket exception when individual videos fail.
"""

from unittest.mock import MagicMock, patch


class TestAiScreenBatchErrorHandling:
    """Verify that individual video failures don't crash the whole batch."""

    @patch("api.services.evaluate.models_service.get_conn")
    @patch("api.services.evaluate.models_service.fetch_youtube_frames")
    def test_single_video_failure_returns_error_entry(self, mock_frames, mock_conn):
        """If fetch_youtube_frames raises for one video, the batch should
        return an error entry for that video, not raise an exception."""
        from api.services.evaluate.models_service import ai_screen_batch

        # DB returns metadata for our test video
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("vid1", "test title", None, None, None),
        ]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.cursor.return_value = mock_cursor
        mock_conn_ctx.__enter__ = lambda s: s
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_conn_ctx

        # Simulate fetch_youtube_frames raising an exception
        mock_frames.side_effect = RuntimeError("Network timeout")

        results = ai_screen_batch(["vid1"], threshold=0.90)

        assert len(results) == 1
        assert results[0]["video_id"] == "vid1"
        assert "error" in results[0]
        assert "Processing failed" in results[0]["error"]

    @patch("api.services.evaluate.models_service.get_conn")
    @patch("api.services.evaluate.models_service.fetch_youtube_frames")
    def test_fetch_returns_none_auto_rejects(self, mock_frames, mock_conn):
        """If fetch_youtube_frames returns empty, the video is auto-rejected."""
        from api.services.evaluate.models_service import ai_screen_batch

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("vid2", "another title", None, None, None),
        ]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.cursor.return_value = mock_cursor
        mock_conn_ctx.__enter__ = lambda s: s
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_conn_ctx

        mock_frames.return_value = []

        results = ai_screen_batch(["vid2"], threshold=0.90)

        assert len(results) == 1
        assert results[0]["video_id"] == "vid2"
        assert results[0]["predicted_class"] == "reject"
        assert results[0]["confidence"] == 1.0
        assert results[0]["auto_decided"] is True
        assert results[0]["error"] is None
