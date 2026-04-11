"""Patch-token pooling modules for board representations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchPoolingHead(nn.Module):
    """Pool patch tokens into a single board embedding.

    Supported modes:
    - ``mean``: global mean pooling over spatial patch tokens.
    - ``square_attention``: pool the patch grid to an 8x8 square grid,
      add learned square embeddings, then attention-pool the square tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        pooling_type: str = "mean",
        square_size: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.square_size = square_size

        if pooling_type == "mean":
            return
        if pooling_type != "square_attention":
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        num_squares = square_size * square_size
        self.square_pos_embed = nn.Parameter(torch.zeros(num_squares, embed_dim))
        self.square_norm = nn.LayerNorm(embed_dim)
        self.square_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.square_pos_embed, std=0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        spatial_tokens = self._extract_spatial_tokens(patch_tokens)
        if self.pooling_type == "mean":
            return spatial_tokens.mean(dim=1)
        return self._forward_square_attention(spatial_tokens)

    def _forward_square_attention(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        square_tokens = self.to_square_tokens(spatial_tokens)
        square_tokens = square_tokens + self.square_pos_embed.unsqueeze(0)
        square_tokens = square_tokens + self.square_mlp(self.square_norm(square_tokens))
        attn_logits = self.attn_proj(square_tokens).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(square_tokens * attn_weights.unsqueeze(-1), dim=1)
        result: torch.Tensor = self.output_norm(pooled)
        return result

    def to_square_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        spatial_tokens = self._extract_spatial_tokens(patch_tokens)
        batch_size, num_tokens, embed_dim = spatial_tokens.shape
        grid_size = self._infer_grid_size(num_tokens)
        grid = spatial_tokens.transpose(1, 2).reshape(batch_size, embed_dim, grid_size, grid_size)
        square_grid = F.adaptive_avg_pool2d(grid, (self.square_size, self.square_size))
        result: torch.Tensor = square_grid.flatten(2).transpose(1, 2)
        return result

    @staticmethod
    def _extract_spatial_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
        num_tokens = patch_tokens.shape[1]
        if PatchPoolingHead._is_perfect_square(num_tokens):
            return patch_tokens
        spatial_tokens = patch_tokens[:, 1:, :]
        if not PatchPoolingHead._is_perfect_square(spatial_tokens.shape[1]):
            raise ValueError(
                f"Expected a square number of spatial tokens, got {spatial_tokens.shape[1]}"
            )
        return spatial_tokens

    @staticmethod
    def _infer_grid_size(num_tokens: int) -> int:
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected square patch grid, got {num_tokens} tokens")
        return grid_size

    @staticmethod
    def _is_perfect_square(value: int) -> bool:
        root = math.isqrt(value)
        return root * root == value
