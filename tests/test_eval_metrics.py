from __future__ import annotations

import pytest
import torch

from argus.eval.metrics import compute_move_metrics


@pytest.mark.parametrize(
    ("threshold", "expected_f1"),
    [
        (0.5, 0.0),
        (0.3, 1.0),
    ],
)
def test_compute_move_metrics_respects_detection_threshold(
    threshold: float,
    expected_f1: float,
) -> None:
    metrics = compute_move_metrics(
        predictions=torch.tensor([[1]], dtype=torch.long),
        targets=torch.tensor([[1]], dtype=torch.long),
        detect_logits=torch.tensor([[-0.2]], dtype=torch.float32),
        detect_targets=torch.tensor([[1.0]], dtype=torch.float32),
        move_mask=torch.tensor([[True]], dtype=torch.bool),
        detect_threshold=threshold,
    )

    assert metrics["move_accuracy"] == pytest.approx(1.0)
    assert metrics["move_detection_f1"] == pytest.approx(expected_f1)
