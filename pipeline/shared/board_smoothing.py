"""Shared temporal smoothing helpers for board-state logits."""

from __future__ import annotations

import torch

from pipeline.shared.board_constraints import constrained_board_class_ids


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
        current_logits = _validated_board_logits(square_logits)
        if self._ema_logits is None:
            self._ema_logits = current_logits
        else:
            self._ema_logits = (
                self.alpha * current_logits + (1.0 - self.alpha) * self._ema_logits
            )
        return self._ema_logits.clone()


class AdaptiveBoardLogitsExponentialSmoother:
    """Causal EMA smoother with a higher update rate for move-like changes."""

    def __init__(
        self,
        *,
        low_alpha: float,
        high_alpha: float,
        high_alpha_change_threshold: int,
    ) -> None:
        if not 0.0 < low_alpha <= 1.0:
            raise ValueError(f"low_alpha must be in (0, 1], got {low_alpha}")
        if not 0.0 < high_alpha <= 1.0:
            raise ValueError(f"high_alpha must be in (0, 1], got {high_alpha}")
        if high_alpha_change_threshold <= 0:
            raise ValueError(
                "high_alpha_change_threshold must be > 0, got "
                f"{high_alpha_change_threshold}"
            )
        self.low_alpha = float(low_alpha)
        self.high_alpha = float(high_alpha)
        self.high_alpha_change_threshold = int(high_alpha_change_threshold)
        self._ema_logits: torch.Tensor | None = None

    def reset(self) -> None:
        self._ema_logits = None

    def update(self, square_logits: torch.Tensor) -> torch.Tensor:
        current_logits = _validated_board_logits(square_logits)
        if self._ema_logits is None:
            self._ema_logits = current_logits
            return self._ema_logits.clone()

        previous_class_ids = constrained_board_class_ids(self._ema_logits)
        current_class_ids = constrained_board_class_ids(current_logits)
        change_count = int((current_class_ids != previous_class_ids).sum().item())
        alpha = self.low_alpha
        if 0 < change_count <= self.high_alpha_change_threshold:
            alpha = self.high_alpha
        self._ema_logits = alpha * current_logits + (1.0 - alpha) * self._ema_logits
        return self._ema_logits.clone()


def _validated_board_logits(square_logits: torch.Tensor) -> torch.Tensor:
    if square_logits.ndim != 2 or square_logits.shape[0] != 64:
        raise ValueError(f"Expected board logits shaped (64, C), got {tuple(square_logits.shape)}")
    return square_logits.detach().clone()
