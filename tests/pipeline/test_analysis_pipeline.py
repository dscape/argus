"""Tests for the shared local video analysis pipeline."""

import chess
import numpy as np
import pipeline.analysis.pipeline as analysis_pipeline
from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.analysis.frame_extractor import FrameData


def test_video_analysis_pipeline_detects_moves(tmp_path, monkeypatch):
    board = chess.Board()
    fen_start = board.board_fen()
    board.push(chess.Move.from_uci("e2e4"))
    fen_after = board.board_fen()

    frames = [
        FrameData(index=i, timestamp=i / 2.0, image=np.zeros((8, 8, 3), dtype=np.uint8))
        for i in range(6)
    ]

    class FakeReader:
        def __init__(self):
            self._fens = iter([fen_start, fen_start, fen_start, fen_after, fen_after, fen_after])

        def read(self, _image):
            return type("ReadResult", (), {"fen": next(self._fens), "method": "overlay"})()

    monkeypatch.setattr(analysis_pipeline, "extract_frames", lambda *_args, **_kwargs: iter(frames))
    monkeypatch.setattr(analysis_pipeline, "build_frame_reader", lambda _config: FakeReader())

    config = VideoAnalysisConfig(annotate=False, scene_backend="none", output_dir=tmp_path)
    result = analysis_pipeline.VideoAnalysisPipeline(config).run(tmp_path / "demo.mp4")

    assert result.total_moves == 1
    assert len(result.pgn_files) == 1
    assert result.pgn_files[0].exists()
    assert "1. e4" in result.pgn_strings[0]


def test_read_overlay_crop_uses_vlm_fallback_when_hybrid(monkeypatch):
    from pipeline.analysis import board_reading
    from pipeline.mlx import vlm_analyzer

    config = VideoAnalysisConfig(reader_backend="hybrid", scene_backend="none", device="cpu")
    crop = np.zeros((32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(board_reading, "find_board_in_crop", lambda _crop: None)
    monkeypatch.setattr(
        vlm_analyzer,
        "read_board_position",
        lambda _frame, _config: chess.STARTING_BOARD_FEN,
    )

    result = board_reading.read_overlay_crop(crop, config)

    assert result.fen == chess.STARTING_BOARD_FEN
    assert result.method == "vlm_direct"
