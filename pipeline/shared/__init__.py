"""Shared abstractions used by both overlay and physical board pipelines."""

from pipeline.shared.board_calibration import apply_board_logit_bias
from pipeline.shared.board_constraints import constrained_board_class_ids
from pipeline.shared.board_observation import BoardObservation
from pipeline.shared.board_smoothing import (
    AdaptiveBoardLogitsExponentialSmoother,
    BoardLogitsExponentialSmoother,
)
from pipeline.shared.board_state import NUM_SQUARE_CLASSES, SQUARE_CLASS_NAMES, fen_to_square_labels
from pipeline.shared.board_tracking import (
    BoardTrackerResult,
    LegalMoveStateTracker,
    LegalSequenceBeamDecoder,
    LookaheadLegalMoveStateTracker,
    SegmentalLegalSequenceDecoder,
    SequenceBeamSearchResult,
    SequenceTrackerFrameResult,
    board_to_class_ids,
    build_board_hypotheses,
    score_board_state,
    score_legal_move,
)

__all__ = [
    "BoardObservation",
    "apply_board_logit_bias",
    "AdaptiveBoardLogitsExponentialSmoother",
    "BoardLogitsExponentialSmoother",
    "SQUARE_CLASS_NAMES",
    "NUM_SQUARE_CLASSES",
    "constrained_board_class_ids",
    "fen_to_square_labels",
    "BoardTrackerResult",
    "LegalMoveStateTracker",
    "LookaheadLegalMoveStateTracker",
    "LegalSequenceBeamDecoder",
    "SegmentalLegalSequenceDecoder",
    "SequenceBeamSearchResult",
    "SequenceTrackerFrameResult",
    "board_to_class_ids",
    "build_board_hypotheses",
    "score_board_state",
    "score_legal_move",
]
