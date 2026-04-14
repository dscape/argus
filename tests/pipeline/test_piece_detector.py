from __future__ import annotations

import chess
import numpy as np
from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.analysis.piece_detector import (
    BoardState,
    _detect_with_square_classifier,
    detect_pieces,
)


def test_detect_with_square_classifier_uses_physical_module(monkeypatch) -> None:
    import pipeline.physical.square_classifier as square_classifier

    monkeypatch.setattr(
        square_classifier,
        "read_fen_from_frame",
        lambda _board_crop, **_kwargs: chess.STARTING_BOARD_FEN,
    )

    result = _detect_with_square_classifier(
        np.zeros((32, 32, 3), dtype=np.uint8),
        device="cpu",
    )

    assert result == BoardState(
        fen=chess.STARTING_BOARD_FEN,
        method="physical_square_classifier",
    )


def test_detect_with_square_classifier_uses_sequence_reader_when_provided() -> None:
    class DummySequenceReader:
        def read_fen_from_frame(self, _board_crop: np.ndarray) -> str:
            return chess.STARTING_BOARD_FEN

    result = _detect_with_square_classifier(
        np.zeros((32, 32, 3), dtype=np.uint8),
        device="cpu",
        sequence_reader=DummySequenceReader(),
    )

    assert result == BoardState(
        fen=chess.STARTING_BOARD_FEN,
        method="physical_square_classifier",
    )


def test_detect_with_square_classifier_passes_corners_to_sequence_reader() -> None:
    calls: list[tuple[tuple[float, float], ...] | list[list[float]] | None] = []

    class DummySequenceReader:
        def read_fen_from_frame(
            self,
            _board_crop: np.ndarray,
            *,
            corners: tuple[tuple[float, float], ...] | list[list[float]],
        ) -> str:
            calls.append(corners)
            return chess.STARTING_BOARD_FEN

    result = _detect_with_square_classifier(
        np.zeros((32, 32, 3), dtype=np.uint8),
        device="cpu",
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        sequence_reader=DummySequenceReader(),
    )

    assert result == BoardState(
        fen=chess.STARTING_BOARD_FEN,
        method="physical_square_classifier",
    )
    assert calls == [((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))]


def test_detect_pieces_falls_back_to_vlm_when_physical_classifier_has_no_model(
    monkeypatch,
) -> None:
    import pipeline.analysis.piece_detector as piece_detector

    monkeypatch.setattr(
        piece_detector,
        "_detect_with_square_classifier",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        piece_detector,
        "_detect_with_vlm",
        lambda _board_rgb, _config: BoardState(
            fen=chess.STARTING_BOARD_FEN,
            method="vlm_direct",
        ),
    )

    result = detect_pieces(
        np.zeros((32, 32, 3), dtype=np.uint8),
        VideoAnalysisConfig(device="cpu"),
    )

    assert result == BoardState(
        fen=chess.STARTING_BOARD_FEN,
        method="vlm_direct",
    )
