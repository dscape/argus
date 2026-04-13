from __future__ import annotations

import pytest
import torch
from pipeline.shared.board_smoothing import (
    AdaptiveBoardLogitsExponentialSmoother,
    BoardLogitsExponentialSmoother,
)


def test_board_logits_exponential_smoother_applies_causal_ema() -> None:
    smoother = BoardLogitsExponentialSmoother(alpha=0.25)
    first = torch.zeros((64, 13), dtype=torch.float32)
    second = torch.ones((64, 13), dtype=torch.float32)

    smoothed_first = smoother.update(first)
    smoothed_second = smoother.update(second)

    assert torch.allclose(smoothed_first, first)
    assert torch.allclose(smoothed_second, 0.25 * second + 0.75 * first)


def test_board_logits_exponential_smoother_reset_clears_state() -> None:
    smoother = BoardLogitsExponentialSmoother(alpha=0.5)
    smoother.update(torch.ones((64, 13), dtype=torch.float32))

    smoother.reset()
    smoothed = smoother.update(torch.zeros((64, 13), dtype=torch.float32))

    assert torch.allclose(smoothed, torch.zeros((64, 13), dtype=torch.float32))


def test_adaptive_board_logits_exponential_smoother_uses_high_alpha_for_move_like_changes() -> None:
    smoother = AdaptiveBoardLogitsExponentialSmoother(
        low_alpha=0.03,
        high_alpha=0.08,
        high_alpha_change_threshold=8,
    )
    first = torch.zeros((64, 13), dtype=torch.float32)
    first[:, 0] = 1.0
    first[4, 12] = 6.0
    first[60, 6] = 6.0

    second = first.clone()
    second[12, 1] = 4.0
    second[20, 0] = 4.0

    smoother.update(first)
    smoothed_second = smoother.update(second)

    expected = 0.08 * second + 0.92 * first
    assert torch.allclose(smoothed_second, expected)


def test_adaptive_board_logits_exponential_smoother_uses_low_alpha_for_large_changes() -> None:
    smoother = AdaptiveBoardLogitsExponentialSmoother(
        low_alpha=0.03,
        high_alpha=0.08,
        high_alpha_change_threshold=4,
    )
    first = torch.zeros((64, 13), dtype=torch.float32)
    first[:, 0] = 1.0
    first[4, 12] = 6.0
    first[60, 6] = 6.0

    second = first.clone()
    for index in range(8, 18):
        second[index, 3] = 4.0

    smoother.update(first)
    smoothed_second = smoother.update(second)

    expected = 0.03 * second + 0.97 * first
    assert torch.allclose(smoothed_second, expected)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_board_logits_exponential_smoother_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        BoardLogitsExponentialSmoother(alpha=alpha)


@pytest.mark.parametrize(
    ("low_alpha", "high_alpha", "high_alpha_change_threshold"),
    [(0.0, 0.1, 4), (0.1, 0.0, 4), (0.1, 0.2, 0)],
)
def test_adaptive_board_logits_exponential_smoother_rejects_invalid_params(
    low_alpha: float,
    high_alpha: float,
    high_alpha_change_threshold: int,
) -> None:
    with pytest.raises(ValueError):
        AdaptiveBoardLogitsExponentialSmoother(
            low_alpha=low_alpha,
            high_alpha=high_alpha,
            high_alpha_change_threshold=high_alpha_change_threshold,
        )
