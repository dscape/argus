"""End-to-end tests for overlay FEN extraction on reference frames.

Each test must achieve 100% square-level accuracy on all 4 ground truth
positions.  The test loads a pre-extracted frame, detects the grid, classifies
all 64 squares, and compares the resulting FEN against the verified ground
truth.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.pipeline.fen_helpers import compare_boards, fen_to_board

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frames"

_gt_path = FIXTURES_DIR / "ground_truth.json"
GROUND_TRUTH: dict[str, dict] = json.loads(_gt_path.read_text()) if _gt_path.exists() else {}

pytestmark = pytest.mark.skipif(
    not {"O8Z", "7Ra", "2wW", "Ov8"}.issubset(GROUND_TRUTH.keys()),
    reason="Old fixture frames (O8Z, 7Ra, etc.) not present in ground_truth.json",
)


def _load_frame(entry: dict) -> np.ndarray:
    """Load a pre-extracted frame from the fixtures directory."""
    path = FIXTURES_DIR / entry["image"]
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None, f"Cannot load fixture image: {path}"
    return frame


class TestOverlayFenExtraction:
    """CNN piece classifier must achieve 100% accuracy on all reference frames."""

    @pytest.mark.parametrize("name", ["O8Z", "7Ra", "2wW", "Ov8"])
    def test_fen_exact_match(self, name: str) -> None:
        from pipeline.overlay.piece_classifier import read_fen_from_frame

        info = GROUND_TRUTH[name]
        frame = _load_frame(info)

        fen = read_fen_from_frame(frame)
        assert fen is not None, f"{name}: read_fen_from_frame returned None"

        predicted = fen_to_board(fen)
        expected = fen_to_board(info["fen"])
        compare_boards(predicted, expected, name)
