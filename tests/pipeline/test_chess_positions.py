"""Test piece classification accuracy on chess-positions board fixtures.

Uses ``read_fen_with_grid()`` with the known 50×50 uniform grid, bypassing
grid detection entirely.  This isolates the classifier from grid detection.

Boards are added to ``tests/fixtures/boards/`` via
``scripts/eval_chess_positions.py --add-failures`` whenever accuracy drops.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pipeline.overlay.grid_detector import GridResult

from tests.pipeline.fen_helpers import compare_boards, fen_to_board

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "boards"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"


def _load_ground_truth() -> dict[str, dict]:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


GROUND_TRUTH = _load_ground_truth()


def _load_board(entry: dict) -> np.ndarray:
    path = FIXTURES_DIR / entry["image"]
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None, f"Cannot load fixture image: {path}"
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChessPositionsClassification:
    """Piece classifier must achieve 100% accuracy on all board fixtures."""

    @pytest.mark.skipif(not GROUND_TRUTH, reason="No board fixtures yet")
    @pytest.mark.parametrize("name", list(GROUND_TRUTH.keys()))
    def test_board(self, name: str) -> None:
        from pipeline.overlay.piece_classifier import read_fen_with_grid

        info = GROUND_TRUTH[name]
        frame = _load_board(info)

        g = info["grid"]
        sq_size = g["v_lines"][1] - g["v_lines"][0]
        grid = GridResult(g["v_lines"], g["h_lines"], sq_size)

        predicted_fen = read_fen_with_grid(frame, grid, detect_orientation=False)
        expected_fen = info["fen"]

        compare_boards(
            fen_to_board(predicted_fen),
            fen_to_board(expected_fen),
            name,
        )
