"""Shared helpers for calibrating per-square board logits."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def apply_board_logit_bias(
    square_logits: torch.Tensor,
    class_logit_bias: Sequence[float] | torch.Tensor | None,
) -> torch.Tensor:
    if class_logit_bias is None:
        return square_logits
    if square_logits.ndim != 2 or square_logits.shape[0] != 64:
        raise ValueError(f"Expected board logits shaped (64, C), got {tuple(square_logits.shape)}")

    if isinstance(class_logit_bias, torch.Tensor):
        bias = class_logit_bias.to(dtype=square_logits.dtype, device=square_logits.device)
    else:
        bias = torch.tensor(list(class_logit_bias), dtype=square_logits.dtype, device=square_logits.device)
    if bias.ndim != 1 or bias.shape[0] != square_logits.shape[1]:
        raise ValueError(
            "class_logit_bias must have shape (C,), got "
            f"{tuple(bias.shape)} for logits with {square_logits.shape[1]} classes"
        )
    return square_logits + bias.unsqueeze(0)
