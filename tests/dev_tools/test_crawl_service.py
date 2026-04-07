"""Tests for crawl_service frame fetching filters."""

import importlib
import sys
import types
from unittest.mock import MagicMock, call, patch


def _mock_db_conn(video_rows: list[tuple[str]]) -> tuple[MagicMock, MagicMock]:
    """Create a mock DB connection returning the given video rows."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = video_rows
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.__exit__.return_value = False

    mock_conn_ctx = MagicMock()
    mock_conn_ctx.cursor.return_value = mock_cursor
    mock_conn_ctx.__enter__.return_value = mock_conn_ctx
    mock_conn_ctx.__exit__.return_value = False
    return mock_conn_ctx, mock_cursor


def _import_crawl_service():
    """Import crawl_service with lightweight stand-ins for optional deps."""
    sys.modules.pop("api.services.videos.crawl_service", None)
    with patch.dict(
        sys.modules,
        {
            "pipeline.db.connection": types.SimpleNamespace(get_conn=MagicMock()),
            "pipeline.screen.title_filter": types.SimpleNamespace(
                score_title=MagicMock(return_value=(False, 0.0))
            ),
        },
    ):
        return importlib.import_module("api.services.videos.crawl_service")


class TestFetchFramesForChannel:
    """Verify channel frame fetching uses the intended video filter."""

    def test_includes_approved_null_layout_as_overlay(self):
        """Approved videos with NULL layout_type should still be fetched."""
        crawl_service = _import_crawl_service()

        mock_conn, mock_cursor = _mock_db_conn([("vid-null",), ("vid-overlay",)])
        mock_fetch = MagicMock(
            side_effect=[
                [("25pct", 1280, 720), ("50pct", 1280, 720)],
                [("25pct", 1280, 720)],
            ]
        )

        with (
            patch.object(crawl_service, "get_conn", return_value=mock_conn),
            patch.dict(
                sys.modules,
                {
                    "pipeline.screen.frame_fetcher": types.SimpleNamespace(
                        fetch_overlay_frames=mock_fetch
                    )
                },
            ),
        ):
            result = crawl_service.fetch_frames_for_channel("channel-123", hires=True)

        query, params = mock_cursor.execute.call_args[0]
        assert "screening_status = 'approved'" in query
        assert "(layout_type = 'overlay' OR layout_type IS NULL)" in query
        assert params == ("channel-123",)
        assert mock_fetch.call_args_list == [
            call("vid-null", hires=True),
            call("vid-overlay", hires=True),
        ]
        assert result == {
            "channel_id": "channel-123",
            "videos_processed": 2,
            "frames_fetched": 3,
        }

    def test_returns_zero_when_no_matching_videos(self):
        """No matching approved overlay videos should short-circuit cleanly."""
        crawl_service = _import_crawl_service()

        mock_conn, _ = _mock_db_conn([])
        mock_fetch = MagicMock()

        with (
            patch.object(crawl_service, "get_conn", return_value=mock_conn),
            patch.dict(
                sys.modules,
                {
                    "pipeline.screen.frame_fetcher": types.SimpleNamespace(
                        fetch_overlay_frames=mock_fetch
                    )
                },
            ),
        ):
            result = crawl_service.fetch_frames_for_channel("channel-123", hires=False)

        mock_fetch.assert_not_called()
        assert result == {
            "channel_id": "channel-123",
            "videos_processed": 0,
            "frames_fetched": 0,
        }
