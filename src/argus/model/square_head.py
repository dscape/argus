"""Dense per-square board-state classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


class SquareHead(nn.Module):
    """Predict piece classes for each of 64 board squares."""

    def __init__(self, embed_dim: int, num_classes: int = 13) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(square_tokens)
        return logits
