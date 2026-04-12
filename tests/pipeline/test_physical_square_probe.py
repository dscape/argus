from __future__ import annotations

import torch
from pipeline.physical.square_probe import PhysicalSquareLinearProbe, evaluate_probe


def test_evaluate_probe_reports_perfect_metrics() -> None:
    probe = PhysicalSquareLinearProbe(embed_dim=2)
    with torch.no_grad():
        probe.classifier.weight.zero_()
        probe.classifier.bias.fill_(-10.0)
        probe.classifier.weight[0, 0] = 20.0
        probe.classifier.weight[1, 1] = 20.0
        probe.classifier.bias[0] = 0.0
        probe.classifier.bias[1] = 0.0

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    metrics = evaluate_probe(
        probe,
        features,
        labels,
        device=torch.device("cpu"),
        board_annotation_ids=["ann-1", "ann-1"],
    )

    assert metrics.accuracy == 1.0
    assert metrics.non_empty_accuracy == 1.0
    assert metrics.board_exact_match == 1.0
    assert metrics.class_accuracy["empty"] == 1.0
    assert metrics.class_accuracy["P"] == 1.0
