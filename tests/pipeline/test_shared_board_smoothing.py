from __future__ import annotations

import pytest
import torch
from pipeline.shared.board_smoothing import BoardLogitsExponentialSmoother


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


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_board_logits_exponential_smoother_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        BoardLogitsExponentialSmoother(alpha=alpha)
