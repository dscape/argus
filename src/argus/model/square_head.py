"""Dense per-square board-state classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

_DEFAULT_HEAD_TYPE = "simple_mlp"
_DEFAULT_HIDDEN_DIM = 512
_DEFAULT_TRANSFORMER_LAYERS = 2
_DEFAULT_TRANSFORMER_HEADS = 8
_DEFAULT_TRANSFORMER_FF_DIM = 1024
_DEFAULT_DROPOUT = 0.1
_SQUARE_COUNT = 64


class SquareHead(nn.Module):
    """Predict piece classes for each of 64 board squares."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 13,
        *,
        head_type: str = _DEFAULT_HEAD_TYPE,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        transformer_layers: int = _DEFAULT_TRANSFORMER_LAYERS,
        transformer_heads: int = _DEFAULT_TRANSFORMER_HEADS,
        transformer_ff_dim: int = _DEFAULT_TRANSFORMER_FF_DIM,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.dropout = dropout
        self.classifier: nn.Module

        if head_type == "simple_mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_classes),
            )
            return

        if head_type == "linear":
            self.classifier = nn.Linear(embed_dim, num_classes)
            return

        self.position_embedding = nn.Parameter(torch.zeros(_SQUARE_COUNT, embed_dim))
        nn.init.normal_(self.position_embedding, std=0.02)

        if head_type == "pos_mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            return

        if head_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
            self.output_norm = nn.LayerNorm(embed_dim)
            self.classifier = nn.Linear(embed_dim, num_classes)
            return

        raise ValueError(f"Unsupported square head_type: {head_type}")

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        if self.head_type == "simple_mlp":
            logits: torch.Tensor = self.classifier(square_tokens)
            return logits
        if self.head_type == "linear":
            logits = self.classifier(square_tokens)
            return logits
        positioned_tokens = square_tokens + self.position_embedding.unsqueeze(0)
        if self.head_type == "pos_mlp":
            logits = self.classifier(positioned_tokens)
            return logits
        contextual_tokens = self.context_encoder(positioned_tokens)
        logits = self.classifier(self.output_norm(contextual_tokens))
        return logits

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "head_type": self.head_type,
            "hidden_dim": self.hidden_dim,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "transformer_ff_dim": self.transformer_ff_dim,
            "dropout": self.dropout,
        }
