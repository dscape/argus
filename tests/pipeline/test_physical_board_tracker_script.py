from __future__ import annotations

from scripts.eval_physical_board_tracker import (
    FramePrediction,
    _match_frames,
    compute_tracker_sequence_metrics,
)


def test_match_frames_uses_tolerance_without_reusing_predictions() -> None:
    assert _match_frames([10, 20], [9, 21], tolerance=1) == 2
    assert _match_frames([10, 20], [9], tolerance=1) == 1


def test_compute_tracker_sequence_metrics_reports_recall_and_false_changes() -> None:
    predictions_by_clip = {
        "clip": [
            FramePrediction("ann-0", 0, (0, 0), (0, 0), None),
            FramePrediction("ann-1", 1, (0, 0), (0, 0), None),
            FramePrediction("ann-2", 2, (1, 0), (1, 0), "e2e4"),
            FramePrediction("ann-3", 3, (1, 0), (1, 1), "e7e5"),
        ]
    }

    move_recall, static_false_change_rate, diagnostics = compute_tracker_sequence_metrics(
        predictions_by_clip,
        tolerance=0,
    )

    assert move_recall == 1.0
    assert static_false_change_rate == 0.5
    assert diagnostics["matched_gt_change_frames"] == 1
