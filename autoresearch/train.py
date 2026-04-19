#!/usr/bin/env python3
"""Editable decoder experiment surface for Argus autoresearch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import chess
import torch

from prepare import (
    BEST_TRAIN_PATH,
    PreparedSequence,
    decode_sequence_with_lookahead,
    decode_sequence_with_segmental,
    evaluate_decoder,
    load_prepared_dataset,
    record_successful_run,
    snapshot_current_train,
    write_report_json,
)
from pipeline.shared.board_tracking import score_board_state

EXPERIMENT_NAME = "hybrid_lookahead_segmental_drop1_boardcands2_ratio95_props16_top2_detect10_minsep2_thresh3_total150_permove125_m9p8957061767578125_r1_beam8_v318"
EXPERIMENT_DESCRIPTION = (
    "Hybrid decoder keeps the baseline lookahead path unless a conservative board-only "
    "segmental pass wins the current score-gain gates, while following up the kept beam8 + props16 + top2 branch after top1 matched it, top3 was worse, boardcands3 snapped it toward the older v120-style tradeoff, boardcands1 was harmful, boardcands4 was worse, and the per-added-move gate stayed safe down to 93.75 before breaking at 87.5, and widening only the segmental event window radius from 0 to 1 on that active beam8 regime "
    "to test whether slightly broader temporal aggregation can recover ambiguity cases without collapsing the branch's lower false-change behavior."
)

DECODER_KIND = "hybrid"

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
    "event_window_radius": 1,
    "max_event_proposals": 16,
    "diagnostic_settled_horizon": 8,
}

HYBRID_CONFIG: dict[str, Any] = {
    "min_total_score_gain": 150.0,
    "min_score_gain_per_added_move": 125.0,
}


def experiment_config() -> dict[str, Any]:
    return {
        "baseline": {
            "kind": "lookahead",
            "lookahead_window": LOOKAHEAD_WINDOW,
            "lookahead_margin": LOOKAHEAD_MARGIN,
        },
        "segmental": dict(SEGMENTAL_CONFIG),
        "selection": dict(HYBRID_CONFIG),
    }


def decode_sequence(sequence: PreparedSequence):
    baseline_results = decode_sequence_with_lookahead(
        sequence,
        lookahead_window=LOOKAHEAD_WINDOW,
        lookahead_margin=LOOKAHEAD_MARGIN,
    )
    segmental_results = decode_sequence_with_segmental(sequence, **SEGMENTAL_CONFIG)

    baseline_move_count = sum(result.move_uci is not None for result in baseline_results)
    segmental_move_count = sum(result.move_uci is not None for result in segmental_results)
    added_moves = segmental_move_count - baseline_move_count
    if added_moves <= 0:
        return baseline_results

    baseline_score = sequence_board_score(sequence, baseline_results)
    segmental_score = sequence_board_score(sequence, segmental_results)
    score_gain = segmental_score - baseline_score
    score_gain_per_added_move = score_gain / added_moves
    if score_gain < float(HYBRID_CONFIG["min_total_score_gain"]):
        return baseline_results
    if score_gain_per_added_move < float(HYBRID_CONFIG["min_score_gain_per_added_move"]):
        return baseline_results
    return segmental_results


def sequence_board_score(
    sequence: PreparedSequence,
    results: list[Any],
) -> float:
    log_probs = [torch.log_softmax(logits, dim=1) for logits in sequence.logits]
    return float(
        sum(
            score_board_state(frame_log_probs, chess.Board(result.full_fen))
            for frame_log_probs, result in zip(log_probs, results)
        )
    )


def main() -> None:
    started_at = time.time()
    dataset = load_prepared_dataset()
    train_path = Path(__file__).resolve()
    snapshot_path = snapshot_current_train(train_path, EXPERIMENT_NAME)

    report = evaluate_decoder(
        dataset,
        decode_sequence=decode_sequence,
        decoder_name=DECODER_KIND,
        decoder_config=experiment_config(),
    )
    report["experiment_name"] = EXPERIMENT_NAME
    report["experiment_description"] = EXPERIMENT_DESCRIPTION
    report["elapsed_seconds"] = time.time() - started_at
    report["baseline_board_exact"] = float(dataset.baseline_report["metrics"]["board_exact_match"])
    report["delta_board_exact"] = (
        float(report["metrics"]["board_exact_match"]) - report["baseline_board_exact"]
    )
    report_path = write_report_json(report, snapshot_path)
    decision = record_successful_run(
        train_path=train_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        report=report,
        description=EXPERIMENT_DESCRIPTION,
    )

    print("---")
    print(f"experiment:               {EXPERIMENT_NAME}")
    print(f"decoder:                  {DECODER_KIND}")
    print(f"board_exact:              {report['metrics']['board_exact_match']:.6f}")
    print(f"non_empty_accuracy:       {report['metrics']['non_empty_accuracy']:.6f}")
    print(f"macro_f1:                 {report['metrics']['macro_f1']:.6f}")
    print(f"move_detection_recall:    {report['move_detection_recall']:.6f}")
    print(
        "static_false_change_rate: "
        f"{report['static_frame_false_change_rate']:.6f}"
    )
    print(f"baseline_board_exact:     {report['baseline_board_exact']:.6f}")
    print(f"delta_board_exact:        {report['delta_board_exact']:.6f}")
    print(f"status:                   {decision.status}")
    print(f"snapshot:                 {snapshot_path.relative_to(train_path.parent.parent)}")
    print(f"report:                   {report_path.relative_to(train_path.parent.parent)}")
    print(f"best_train:               {BEST_TRAIN_PATH.relative_to(train_path.parent.parent)}")
    print(f"restored_train:           {str(decision.restored_train).lower()}")
    print(f"total_seconds:            {report['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
