"""Tests for video annotation service backend selection."""

import threading
import time
from pathlib import Path

import chess
import cv2
import numpy as np
from api.services.annotate import video_service
from pipeline.analysis.board_reading import CropReadResult
from pipeline.overlay.calibration import LayoutCalibration


def _write_test_video(path: Path, frame_count: int = 6, fps: float = 2.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 64),
    )
    for idx in range(frame_count):
        frame = np.full((64, 64, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _attach_calibration(session_id: str) -> None:
    video_service._sessions[session_id]["calibration"] = LayoutCalibration(
        overlay=(0, 0, 32, 32),
        camera=(32, 0, 32, 32),
        ref_resolution=(64, 64),
        board_flipped=False,
        board_theme="lichess_default",
    )


def test_read_overlay_at_frame_reports_read_method(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path, frame_count=2)

    session = video_service.open_video(str(video_path))
    session_id = session["session_id"]
    _attach_calibration(session_id)

    monkeypatch.setattr(
        video_service,
        "read_overlay_crop",
        lambda _crop, _config: CropReadResult(
            fen=chess.STARTING_BOARD_FEN,
            method="vlm_direct",
        ),
    )

    try:
        result = video_service.read_overlay_at_frame(session_id, 0, reader_backend="hybrid")
    finally:
        video_service.delete_session(session_id)

    assert result["fen"] == chess.STARTING_BOARD_FEN
    assert result["read_method"] == "vlm_direct"


def test_detect_moves_returns_reader_backend(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path, frame_count=6, fps=2.0)

    board = chess.Board()
    fen_start = board.board_fen()
    board.push(chess.Move.from_uci("e2e4"))
    fen_after = board.board_fen()
    fens = iter([fen_start, fen_start, fen_start, fen_after, fen_after, fen_after])

    session = video_service.open_video(str(video_path))
    session_id = session["session_id"]
    _attach_calibration(session_id)

    monkeypatch.setattr(
        video_service,
        "read_overlay_crop",
        lambda _crop, _config: CropReadResult(fen=next(fens), method="overlay"),
    )

    try:
        result = video_service.detect_moves(
            session_id,
            sample_fps=2.0,
            reader_backend="hybrid",
        )
    finally:
        video_service.delete_session(session_id)

    assert result["reader_backend"] == "hybrid"
    assert result["segments"][0]["moves"][0]["move_uci"] == "e2e4"


def test_detect_moves_job_polls_result(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path, frame_count=4)

    session = video_service.open_video(str(video_path))
    session_id = session["session_id"]
    _attach_calibration(session_id)

    started = threading.Event()
    release = threading.Event()

    def fake_detect_moves(
        _session_id,
        sample_fps=2.0,
        clip_id=None,
        reader_backend="overlay",
        progress_callback=None,
    ):
        assert _session_id == session_id
        assert sample_fps == 1.0
        assert clip_id is None
        assert reader_backend == "overlay"
        started.set()
        assert release.wait(timeout=1.0)
        if progress_callback is not None:
            progress_callback(1, 2, 0, 1)
            progress_callback(2, 2, 1, 2)
        return {
            "num_frames_sampled": 2,
            "num_readable": 2,
            "reader_backend": reader_backend,
            "segments": [],
        }

    monkeypatch.setattr(video_service, "detect_moves", fake_detect_moves)

    try:
        job = video_service.start_detect_moves_job(
            session_id,
            sample_fps=1.0,
            reader_backend="overlay",
        )
        assert started.wait(timeout=1.0)

        running = video_service.get_detect_moves_job(job["job_id"], session_id)
        assert running is not None
        assert running["status"] == "running"
        assert running["total_samples"] == 2
        assert running["completed_samples"] < 2

        release.set()

        done = None
        for _ in range(50):
            done = video_service.get_detect_moves_job(job["job_id"], session_id)
            if done is not None and done["status"] == "done":
                break
            time.sleep(0.01)

        assert done is not None
        assert done["status"] == "done"
        assert done["result"]["reader_backend"] == "overlay"
        assert done["completed_samples"] == 2
        assert done["num_readable"] == 2
    finally:
        video_service.delete_session(session_id)
