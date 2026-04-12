from __future__ import annotations

import pytest
from pipeline.shared import BoardObservation, fen_to_square_labels


def test_board_observation_requires_64_square_confidences() -> None:
    with pytest.raises(ValueError, match="64 values"):
        BoardObservation(fen="8/8/8/8/8/8/8/8", square_confidences=(1.0,))


def test_board_observation_rejects_invalid_confidence_range() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        BoardObservation(
            fen="8/8/8/8/8/8/8/8",
            square_confidences=(1.0,) * 63 + (1.1,),
        )


def test_fen_to_square_labels_accepts_full_fen() -> None:
    labels = fen_to_square_labels("8/8/8/3P4/8/8/8/8 w - - 0 1")

    assert labels[3][3] == 1
    assert sum(sum(row) for row in labels) == 1
