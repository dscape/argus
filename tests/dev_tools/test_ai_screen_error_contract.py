"""Tests for AI screening batch error contract.

Verifies that ai_screen_batch returns consistent dict shapes regardless
of success/failure, with all fields always present.
"""

from unittest.mock import MagicMock, patch

# All result dicts must contain these keys
REQUIRED_KEYS = {
    "video_id",
    "predicted_class",
    "confidence",
    "auto_decided",
    "vertical",
    "title_score",
    "max_ovl_score",
    "max_otb_score",
    "model_version",
    "error",
}


def _mock_db_conn(video_rows):
    """Create a mock DB connection that returns the given rows."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = video_rows
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn_ctx = MagicMock()
    mock_conn_ctx.cursor.return_value = mock_cursor
    mock_conn_ctx.__enter__ = lambda s: s
    mock_conn_ctx.__exit__ = MagicMock(return_value=False)
    return mock_conn_ctx


class TestErrorContractConsistency:
    """All results must have the same set of keys regardless of outcome."""

    @patch("api.services.evaluate.models_service.get_conn")
    @patch("api.services.evaluate.models_service.fetch_youtube_frames")
    @patch("api.services.evaluate.models_service.score_title")
    def test_thumbnail_failure_has_all_keys(self, mock_score, mock_frames, mock_conn):
        """When thumbnails can't be fetched, result still has all fields."""
        from api.services.evaluate.models_service import ai_screen_batch

        mock_score.return_value = (False, 0.3)
        mock_conn.return_value = _mock_db_conn([("vid1", "test", None, None, None)])
        mock_frames.return_value = []

        results = ai_screen_batch(["vid1"], threshold=0.90)
        assert len(results) == 1
        assert set(results[0].keys()) >= REQUIRED_KEYS
        assert results[0]["error"] is None
        assert results[0]["predicted_class"] == "reject"
        assert results[0]["auto_decided"] is True

    @patch("api.services.evaluate.models_service.get_conn")
    @patch("api.services.evaluate.models_service.fetch_youtube_frames")
    @patch("api.services.evaluate.models_service.score_title")
    def test_exception_has_all_keys(self, mock_score, mock_frames, mock_conn):
        """When processing raises an exception, result still has all fields."""
        from api.services.evaluate.models_service import ai_screen_batch

        mock_score.return_value = (False, 0.3)
        mock_conn.return_value = _mock_db_conn([("vid1", "test", None, None, None)])
        mock_frames.side_effect = RuntimeError("boom")

        results = ai_screen_batch(["vid1"], threshold=0.90)
        assert len(results) == 1
        assert set(results[0].keys()) >= REQUIRED_KEYS
        assert results[0]["error"] is not None
