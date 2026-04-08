"""Tests for overlay_test_service sampling and candidate selection."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

from api.services.evaluate import overlay_test_service


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


class TestVideoHasUnsavedFrames:
    """Verify frame availability checks handle partially saved videos."""

    @patch.object(overlay_test_service, "_resolve_frame_path")
    def test_partial_saves_still_count_as_candidates(self, mock_resolve) -> None:
        """A video stays eligible while it still has an unsaved extraction frame."""
        mock_resolve.side_effect = lambda _video_id, frame_name: (
            object() if frame_name in ("25pct", "50pct") else None
        )

        assert overlay_test_service._video_has_unsaved_frames("vid-partial", {"vid-partial:25pct"})


class TestGetExtractionCandidates:
    """Verify extraction candidate queries follow implicit-overlay semantics."""

    @patch.object(overlay_test_service, "_video_has_unsaved_frames")
    @patch.object(
        overlay_test_service,
        "_get_saved_frame_keys",
        return_value={"vid-null-layout:25pct"},
    )
    @patch.object(overlay_test_service, "get_conn")
    def test_includes_approved_null_layout_videos(
        self,
        mock_get_conn,
        _mock_saved_keys,
        mock_has_unsaved,
    ) -> None:
        """Approved videos with NULL layout_type should still be queried."""
        mock_conn_ctx, mock_cursor = _mock_db_conn(
            [("vid-null-layout",), ("vid-overlay",), ("vid-otb",)]
        )
        mock_get_conn.return_value = mock_conn_ctx
        mock_has_unsaved.side_effect = lambda vid, _saved: vid != "vid-otb"

        result = overlay_test_service.get_extraction_candidates(limit=2)

        query, params = mock_cursor.execute.call_args[0]
        assert "screening_status = 'approved'" in query
        assert "(layout_type = 'overlay' OR layout_type IS NULL)" in query
        assert params == (6,)
        assert result == ["vid-null-layout", "vid-overlay"]
        assert mock_has_unsaved.call_args_list == [
            call("vid-null-layout", {"vid-null-layout:25pct"}),
            call("vid-overlay", {"vid-null-layout:25pct"}),
        ]

    @patch.object(
        overlay_test_service,
        "_fixture_candidate_video_ids",
        return_value=["fixture-video"],
    )
    @patch.object(
        overlay_test_service,
        "_get_saved_frame_keys",
        return_value=set(),
    )
    @patch.object(overlay_test_service, "get_conn")
    def test_falls_back_to_fixture_videos_when_db_candidates_are_empty(
        self,
        mock_get_conn,
        _mock_saved_keys,
        _mock_fixture_candidates,
    ) -> None:
        mock_conn_ctx, _mock_cursor = _mock_db_conn([])
        mock_get_conn.return_value = mock_conn_ctx

        result = overlay_test_service.get_extraction_candidates(limit=2)

        assert result == ["fixture-video"]


class TestFixtureFallbacks:
    def test_sample_board_filenames_falls_back_to_committed_fixture_boards(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        synthetic_dir = tmp_path / "synthetic"
        fixture_dir = tmp_path / "fixtures"
        real_dir = tmp_path / "real"
        synthetic_dir.mkdir()
        fixture_dir.mkdir()
        (fixture_dir / "fixture-board.jpeg").write_bytes(b"fixture")

        monkeypatch.setattr(overlay_test_service, "CHESS_POSITIONS_TEST_DIR", synthetic_dir)
        monkeypatch.setattr(overlay_test_service, "BOARD_FIXTURES_DIR", fixture_dir)
        monkeypatch.setattr(overlay_test_service, "REAL_OVERLAY_TEST_DIR", real_dir)

        assert overlay_test_service.sample_board_filenames(limit=1) == ["fixture-board.jpeg"]

    def test_resolve_board_image_path_falls_back_to_fixture_boards(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        synthetic_dir = tmp_path / "synthetic"
        fixture_dir = tmp_path / "fixtures"
        synthetic_dir.mkdir()
        fixture_dir.mkdir()
        expected = fixture_dir / "fixture-board.jpeg"
        expected.write_bytes(b"fixture")

        monkeypatch.setattr(overlay_test_service, "CHESS_POSITIONS_TEST_DIR", synthetic_dir)
        monkeypatch.setattr(overlay_test_service, "BOARD_FIXTURES_DIR", fixture_dir)

        assert overlay_test_service.resolve_board_image_path("fixture-board.jpeg") == expected

    def test_resolve_frame_path_falls_back_to_fixture_frames(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        frame_path = tmp_path / "fixture-video" / "25pct.jpg"
        frame_path.parent.mkdir(parents=True)
        frame_path.write_bytes(b"fixture")

        monkeypatch.setattr(overlay_test_service, "FIXTURE_FRAMES_DIR", tmp_path)
        monkeypatch.setattr(overlay_test_service, "find_frame", lambda *_args, **_kwargs: None)

        assert overlay_test_service._resolve_frame_path("fixture-video", "25pct") == frame_path
