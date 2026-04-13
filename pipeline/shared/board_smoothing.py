"""Shared temporal smoothing helpers for board-state logits."""

from __future__ import annotations

import torch


class BoardLogitsExponentialSmoother:
    """Causal EMA smoother over `(64, C)` per-square board logits."""

    def __init__(self, *, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._ema_logits: torch.Tensor | None = None

    def reset(self) -> None:
        self._ema_logits = None

    def update(self, square_logits: torch.Tensor) -> torch.Tensor:
        if square_logits.ndim != 2 or square_logits.shape[0] != 64:
            raise ValueError(
                f"Expected board logits shaped (64, C), got {tuple(square_logits.shape)}"
            )

        current_logits = square_logits.detach().clone()
        if self._ema_logits is None:
            self._ema_logits = current_logits
        else:
            self._ema_logits = (
                self.alpha * current_logits + (1.0 - self.alpha) * self._ema_logits
            )
        return self._ema_logits.clone()
