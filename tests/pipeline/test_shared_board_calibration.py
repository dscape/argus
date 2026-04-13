from __future__ import annotations

import pytest
import torch
from pipeline.shared.board_calibration import apply_board_logit_bias


def test_apply_board_logit_bias_adds_per_class_bias() -> None:
    logits = torch.zeros((64, 3), dtype=torch.float32)
    biased = apply_board_logit_bias(logits, [0.1, -0.2, 0.3])

    assert torch.allclose(biased[0], torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
    assert torch.allclose(biased[-1], torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))


def test_apply_board_logit_bias_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        apply_board_logit_bias(torch.zeros((64, 3), dtype=torch.float32), [0.1, 0.2])
