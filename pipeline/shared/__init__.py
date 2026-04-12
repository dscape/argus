"""Shared abstractions used by both overlay and physical board pipelines."""

from pipeline.shared.board_observation import BoardObservation
from pipeline.shared.board_state import NUM_SQUARE_CLASSES, SQUARE_CLASS_NAMES, fen_to_square_labels

__all__ = [
    "BoardObservation",
    "SQUARE_CLASS_NAMES",
    "NUM_SQUARE_CLASSES",
    "fen_to_square_labels",
]
