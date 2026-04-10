"""Tests for overlay_test_service sampling and candidate selection."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import chess
import cv2
import numpy as np
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


def _encode_image_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _make_checkerboard_overlay(
    *,
    board_side: int = 320,
    margin_left: int = 8,
    margin_top: int = 10,
    overlay_side: int = 340,
) -> np.ndarray:
    overlay = np.zeros((overlay_side, overlay_side, 3), dtype=np.uint8)
    sq = board_side // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 80
            y1 = margin_top + row * sq
            x1 = margin_left + col * sq
            overlay[y1 : y1 + sq, x1 : x1 + sq] = color
    return overlay


def _fail_if_called(_video_id: str, _frame_name: str):
    raise AssertionError("should not load frame")


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

    @patch.object(overlay_test_service, "_video_has_frames")
    @patch.object(overlay_test_service, "get_conn")
    def test_includes_approved_null_layout_videos(
        self,
        mock_get_conn,
        mock_has_frames,
    ) -> None:
        """Approved videos with NULL layout_type should still be queried."""
        mock_conn_ctx, mock_cursor = _mock_db_conn(
            [("vid-null-layout",), ("vid-overlay",), ("vid-otb",)]
        )
        mock_get_conn.return_value = mock_conn_ctx
        mock_has_frames.side_effect = lambda vid: vid != "vid-otb"

        result = overlay_test_service.get_extraction_candidates(limit=2)

        query, params = mock_cursor.execute.call_args[0]
        assert "screening_status = 'approved'" in query
        assert "(layout_type = 'overlay' OR layout_type IS NULL)" in query
        assert params == (6,)
        assert result == ["vid-null-layout", "vid-overlay"]
        assert mock_has_frames.call_args_list == [
            call("vid-null-layout"),
            call("vid-overlay"),
        ]

    @patch.object(
        overlay_test_service,
        "_fixture_candidate_video_ids",
        return_value=["fixture-video"],
    )
    @patch.object(overlay_test_service, "get_conn")
    def test_falls_back_to_fixture_videos_when_db_candidates_are_empty(
        self,
        mock_get_conn,
        _mock_fixture_candidates,
    ) -> None:
        mock_conn_ctx, _mock_cursor = _mock_db_conn([])
        mock_get_conn.return_value = mock_conn_ctx

        result = overlay_test_service.get_extraction_candidates(limit=2)

        assert result == ["fixture-video"]

    def test_iter_extractable_frame_names_prefers_fixture_overlay_frames(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            overlay_test_service,
            "_load_fixture_frame_ground_truth",
            lambda: {
                "video-1/25pct": {"has_overlay": False},
                "video-1/50pct": {"has_overlay": True, "bbox": [1, 2, 3, 4]},
                "video-1/75pct": {"has_overlay": True, "bbox": [1, 2, 3, 4]},
            },
        )
        monkeypatch.setattr(
            overlay_test_service,
            "_resolve_frame_path",
            lambda _video_id, frame_name: Path(f"/tmp/{frame_name}.jpg"),
        )

        assert overlay_test_service._iter_extractable_frame_names("video-1") == [
            "50pct",
            "75pct",
        ]

    def test_iter_extractable_frame_names_uses_lores_frames_when_fixture_only_has_negative_sample(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            overlay_test_service,
            "_load_fixture_frame_ground_truth",
            lambda: {
                "video-1/50pct": {"has_overlay": False},
            },
        )
        monkeypatch.setattr(
            overlay_test_service,
            "find_frame",
            lambda _video_id, tier, frame_name: (
                Path(f"/tmp/{tier}/{frame_name}.jpg") if tier == "lores" else None
            ),
        )
        monkeypatch.setattr(overlay_test_service, "FIXTURE_FRAMES_DIR", Path("/tmp/missing"))

        assert overlay_test_service._iter_extractable_frame_names("video-1") == [
            "25pct",
            "50pct",
            "75pct",
        ]


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

    def test_sample_board_filenames_ignores_local_real_samples(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        synthetic_dir = tmp_path / "synthetic"
        fixture_dir = tmp_path / "fixtures"
        real_dir = tmp_path / "real"
        synthetic_dir.mkdir()
        fixture_dir.mkdir()
        real_dir.mkdir()
        (synthetic_dir / "synthetic-board.jpeg").write_bytes(b"fixture")
        (real_dir / "real-board.jpeg").write_bytes(b"fixture")

        monkeypatch.setattr(overlay_test_service, "CHESS_POSITIONS_TEST_DIR", synthetic_dir)
        monkeypatch.setattr(overlay_test_service, "BOARD_FIXTURES_DIR", fixture_dir)
        monkeypatch.setattr(overlay_test_service, "REAL_OVERLAY_TEST_DIR", real_dir)

        assert overlay_test_service.sample_board_filenames(limit=5) == ["synthetic-board.jpeg"]

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


class TestBoardCropReuse:
    def test_detect_overlay_from_frames_returns_refined_board_crop(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        overlay = _make_checkerboard_overlay()
        frame[20:360, 30:370] = overlay
        frame_path = tmp_path / "frame.jpg"
        assert cv2.imwrite(str(frame_path), frame)

        monkeypatch.setattr(
            overlay_test_service,
            "detect_overlay_runtime",
            lambda _frame: SimpleNamespace(found=True, bbox=(30, 20, 340, 340)),
        )
        monkeypatch.setattr(
            overlay_test_service,
            "_resolve_frame_path",
            lambda _video_id, _frame_name: frame_path,
        )
        monkeypatch.setattr(overlay_test_service, "_get_saved_frame_keys", lambda: set())

        result = overlay_test_service.detect_overlay_from_frames("video-1")

        assert result["status"] == "detected"
        crop = overlay_test_service._decode_image_b64(result["image_b64"])
        assert crop.shape[0] < overlay.shape[0]
        assert crop.shape[1] < overlay.shape[1]
        assert abs(crop.shape[0] - 320) <= 12
        assert abs(crop.shape[1] - 320) <= 12
        assert result["overlay_detect_ms"] >= 0.0
        assert result["grid_detect_ms"] >= 0.0

    def test_detect_overlay_from_frames_uses_fixture_overlay_frame_and_keeps_saved_samples(
        self,
        monkeypatch,
    ) -> None:
        board = np.full((320, 320, 3), 180, dtype=np.uint8)

        monkeypatch.setattr(
            overlay_test_service,
            "_video_has_frames",
            lambda _video_id: True,
        )
        monkeypatch.setattr(
            overlay_test_service,
            "_iter_extractable_frame_names",
            lambda _video_id: ["50pct"],
        )
        monkeypatch.setattr(
            overlay_test_service,
            "_load_board_crop_for_video_frame",
            lambda _video_id, _frame_name: overlay_test_service.BoardCropResult(
                board_crop=board,
                overlay_detect_ms=0.0,
                grid_detect_ms=1.2,
                detector_found=False,
                warning=(
                    "Runtime detector missed a known overlay; showing the fixture crop for review."
                ),
            ),
        )
        monkeypatch.setattr(
            overlay_test_service,
            "_is_saved_frame",
            lambda _video_id, _frame_name: True,
        )

        result = overlay_test_service.detect_overlay_from_frames("video-1")

        assert result["frame_key"] == "video-1:50pct"
        assert result["status"] == "warning"
        assert result["detector_found"] is False
        assert result["already_saved"] is True
        assert "Runtime detector missed" in result["warning"]
        crop = overlay_test_service._decode_image_b64(result["image_b64"])
        assert crop.shape[:2] == board.shape[:2]

    def test_classify_overlay_fen_uses_provided_board_image(self, monkeypatch) -> None:
        board = np.full((320, 320, 3), 180, dtype=np.uint8)
        image_b64 = _encode_image_b64(board)

        monkeypatch.setattr(
            overlay_test_service,
            "_load_board_crop_for_video_frame",
            _fail_if_called,
        )
        monkeypatch.setattr(
            overlay_test_service,
            "_read_board_crop_fen",
            lambda _image: chess.STARTING_BOARD_FEN,
        )

        result = overlay_test_service.classify_overlay_fen(
            "video-1",
            "25pct",
            image_b64=image_b64,
        )

        assert result["predicted_fen"] == chess.STARTING_BOARD_FEN
        assert result["status"] == "ok"
        assert result["piece_classify_ms"] >= 0.0

    def test_save_confirmed_frame_extractions_uses_provided_board_image(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        board = np.full((320, 320, 3), 180, dtype=np.uint8)
        image_b64 = _encode_image_b64(board)

        monkeypatch.setattr(overlay_test_service, "REAL_OVERLAY_TEST_DIR", tmp_path)
        monkeypatch.setattr(
            overlay_test_service,
            "_load_board_crop_for_video_frame",
            _fail_if_called,
        )

        result = overlay_test_service.save_confirmed_frame_extractions(
            [
                {
                    "video_id": "video-1",
                    "frame_name": "25pct",
                    "fen": chess.STARTING_BOARD_FEN,
                    "image_b64": image_b64,
                }
            ]
        )

        assert result == {"saved": 1, "errors": []}
        saved_path = tmp_path / "f_video-1_25pct_rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.jpg"
        saved = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)
        assert saved is not None
        assert saved.shape[:2] == board.shape[:2]
