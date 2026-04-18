"""Default production decoder for physical board-probe logits."""

from __future__ import annotations

from typing import Any

import chess
import torch

from pipeline.shared import LookaheadLegalMoveStateTracker, SegmentalLegalSequenceDecoder
from pipeline.shared.board_tracking import SequenceTrackerFrameResult, score_board_state

PRODUCTION_DECODER_ID = "v282"
PRODUCTION_DECODER_DESCRIPTION = (
    "Hybrid lookahead + segmental decoder promoted from the autoresearch resweep over the "
    "refactor-era retrained board-probe logits."
)

LOOKAHEAD_WINDOW = 2
LOOKAHEAD_MARGIN = 10.0

SEGMENTAL_CONFIG: dict[str, Any] = {
    "beam_size": 8,
    "top_move_candidates": 2,
    "top_board_candidates": 2,
    "board_weight": 1.0,
    "move_weight": 0.0,
    "detect_weight": 0.0,
    "move_score_margin": 9.8957061767578125,
    "detect_peak_threshold": 0.1,
    "board_change_peak_threshold": 3.0 / 64.0,
    "min_event_separation": 2,
    "secondary_min_event_separation": 2,
    "secondary_peak_ratio": 0.95,
    "state_aware_proposal_passes": 0,
    "anomaly_change_evidence_threshold": 0.25,
    "anomaly_settled_gain_threshold": 0.0,
    "segment_board_drop_worst_frames": 1,
    "event_window_radius": 0,
    "max_event_proposals": 16,
    "diagnostic_settled_horizon": 8,
}

HYBRID_CONFIG: dict[str, Any] = {
    "min_total_score_gain": 150.0,
    "min_score_gain_per_added_move": 125.0,
}


def production_decoder_config() -> dict[str, Any]:
    return {
        "id": PRODUCTION_DECODER_ID,
        "description": PRODUCTION_DECODER_DESCRIPTION,
        "baseline": {
            "kind": "lookahead",
            "lookahead_window": LOOKAHEAD_WINDOW,
            "lookahead_margin": LOOKAHEAD_MARGIN,
        },
        "segmental": dict(SEGMENTAL_CONFIG),
        "selection": dict(HYBRID_CONFIG),
    }


def decode_sequence_with_production_decoder(
    logits_sequence: list[torch.Tensor] | torch.Tensor,
    *,
    initial_board_fen: str,
    initial_side_to_move: str | None,
) -> list[SequenceTrackerFrameResult]:
    logits_tensor = _logits_tensor(logits_sequence)
    baseline_results = LookaheadLegalMoveStateTracker(
        initial_board_fen,
        initial_side_to_move=initial_side_to_move,
        lookahead_window=LOOKAHEAD_WINDOW,
        move_score_margin=LOOKAHEAD_MARGIN,
    ).decode(list(logits_tensor))
    segmental_results = list(
        SegmentalLegalSequenceDecoder(
            initial_board_fen,
            initial_side_to_move=initial_side_to_move,
            **SEGMENTAL_CONFIG,
        ).decode(logits_tensor).frames
    )

    baseline_move_count = sum(result.move_uci is not None for result in baseline_results)
    segmental_move_count = sum(result.move_uci is not None for result in segmental_results)
    added_moves = segmental_move_count - baseline_move_count
    if added_moves <= 0:
        return baseline_results

    baseline_score = _sequence_board_score(logits_tensor, baseline_results)
    segmental_score = _sequence_board_score(logits_tensor, segmental_results)
    score_gain = segmental_score - baseline_score
    if score_gain < float(HYBRID_CONFIG["min_total_score_gain"]):
        return baseline_results
    score_gain_per_added_move = score_gain / added_moves
    if score_gain_per_added_move < float(HYBRID_CONFIG["min_score_gain_per_added_move"]):
        return baseline_results
    return segmental_results


def _sequence_board_score(
    logits_tensor: torch.Tensor,
    results: list[SequenceTrackerFrameResult],
) -> float:
    log_probs = [torch.log_softmax(logits, dim=1) for logits in logits_tensor]
    return float(
        sum(
            score_board_state(frame_log_probs, chess.Board(result.full_fen))
            for frame_log_probs, result in zip(log_probs, results)
        )
    )


def _logits_tensor(logits_sequence: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(logits_sequence, torch.Tensor):
        return logits_sequence.to(dtype=torch.float32)
    return torch.stack([logits.to(dtype=torch.float32) for logits in logits_sequence], dim=0)
