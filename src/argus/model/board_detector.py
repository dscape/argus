"""DETR-style board detection and identity tracking."""

from __future__ import annotations

import torch
import torch.nn as nn


class BoardDetector(nn.Module):
    """DETR-style board detection with identity embeddings."""

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        identity_dim: int = 128,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if input_dim is not None and input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4), nn.Sigmoid(),
        )
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.identity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, identity_dim),
        )

    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = patch_features.shape[0]
        memory = self.input_proj(patch_features)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        decoder_out = self.decoder(queries, memory)
        bboxes = self.bbox_head(decoder_out)
        confidences = self.confidence_head(decoder_out).squeeze(-1)
        identities = self.identity_head(decoder_out)
        return bboxes, confidences, identities
