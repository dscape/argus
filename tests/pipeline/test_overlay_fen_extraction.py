"""End-to-end tests for overlay FEN extraction on reference videos.

Each approach must achieve 100% square-level accuracy on all 4 ground truth
positions.  The test extracts a frame, detects the grid, classifies all 64
squares, and compares the resulting FEN against the verified ground truth.
"""

import chess
import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ground truth (verified by user)
# ---------------------------------------------------------------------------

VIDEO_DIR = "data/videos/STLChessClub"

GROUND_TRUTH: dict[str, dict] = {
    "O8Z": {
        "video": f"{VIDEO_DIR}/O8ZwstOxG_A.mp4",
        "timestamp": 60,
        "fen": "r3rn1k/p1pq1ppp/bp3n2/3p1N2/3P4/PPB3PB/4PP1P/R2QR1K1",
    },
    "7Ra": {
        "video": f"{VIDEO_DIR}/7RaBQag34Hk.mp4",
        "timestamp": 60,
        "fen": "r1bqkb1r/pp1n1pp1/2n1p2p/2ppP3/3P3P/2P2NP1/PP3P2/RNBQKB1R",
    },
    "2wW": {
        "video": f"{VIDEO_DIR}/2wWUKmCBr6A.mp4",
        "timestamp": 60,
        "fen": "r1b2rk1/pp3ppp/2n1pn2/2b5/2P5/5NP1/P2NPPBP/R1BR2K1",
    },
    "Ov8": {
        "video": f"{VIDEO_DIR}/Ov8PXnJp1PU.mp4",
        "timestamp": 60,
        "fen": "8/8/8/1p6/pBb4p/P4k2/7P/6K1",
    },
}


def _extract_frame(video_path: str, timestamp: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Failed to read frame at {timestamp}s from {video_path}"
    return frame


def _fen_to_board(fen: str) -> chess.Board:
    """Parse a piece-placement-only or full FEN string."""
    if " " not in fen:
        fen = fen + " w - - 0 1"
    return chess.Board(fen)


def _compare_boards(predicted: chess.Board, expected: chess.Board, label: str) -> None:
    """Assert that two boards are identical, with a helpful diff on failure."""
    mismatches: list[str] = []
    for sq in chess.SQUARES:
        p = predicted.piece_at(sq)
        e = expected.piece_at(sq)
        if p != e:
            sq_name = chess.square_name(sq)
            p_str = p.symbol() if p else "."
            e_str = e.symbol() if e else "."
            mismatches.append(f"  {sq_name}: got {p_str}, expected {e_str}")

    if mismatches:
        pred_str = "\n".join(f"    {line}" for line in str(predicted).split("\n"))
        exp_str = "\n".join(f"    {line}" for line in str(expected).split("\n"))
        diff = "\n".join(mismatches)
        pytest.fail(
            f"{label}: {len(mismatches)} square(s) wrong\n"
            f"  Predicted:\n{pred_str}\n"
            f"  Expected:\n{exp_str}\n"
            f"  Mismatches:\n{diff}"
        )


# ---------------------------------------------------------------------------
# Approach A — self-bootstrap heuristic reader
# ---------------------------------------------------------------------------


class TestApproachA:
    """Heuristic overlay reader (self-bootstrap templates from video)."""

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW", "Ov8"])
    def test_fen_exact_match(self, name: str) -> None:
        from pipeline.overlay.overlay_reader_v2 import read_fen_from_frame

        info = GROUND_TRUTH[name]
        frame = _extract_frame(info["video"], info["timestamp"])

        fen = read_fen_from_frame(frame, video_path=info["video"])
        assert fen is not None, f"{name}: read_fen_from_frame returned None"

        predicted = _fen_to_board(fen)
        expected = _fen_to_board(info["fen"])
        _compare_boards(predicted, expected, f"Approach A / {name}")


# ---------------------------------------------------------------------------
# Approach B — CNN piece classifier
# ---------------------------------------------------------------------------


class TestApproachB:
    """DINOv2-based CNN piece classifier."""

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW", "Ov8"])
    def test_fen_exact_match(self, name: str) -> None:
        from pipeline.overlay.piece_classifier import read_fen_from_frame

        info = GROUND_TRUTH[name]
        frame = _extract_frame(info["video"], info["timestamp"])

        fen = read_fen_from_frame(frame)
        assert fen is not None, f"{name}: read_fen_from_frame returned None"

        predicted = _fen_to_board(fen)
        expected = _fen_to_board(info["fen"])
        _compare_boards(predicted, expected, f"Approach B / {name}")
