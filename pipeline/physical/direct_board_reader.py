"""Full-image direct board readers predicting 64 square labels jointly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from argus.model.oblique_square_decoder import (
    _build_canonical_square_coordinates,
    _build_patch_positions,
    _extract_spatial_tokens,
    _infer_grid_size,
)
from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.board_probe import PhysicalBoardStateProbe


@dataclass(frozen=True)
class DirectBoardReaderConfig:
    input_size: int = 224
    num_classes: int = 13
    num_heads: int = 8
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    head_type: str = "pos_mlp"
    hidden_dim: int = 512
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024


class DirectSquareQueryDecoder(nn.Module):
    """Decode 64 square tokens jointly from one full image without board corners."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(embed_dim * mlp_ratio), embed_dim)
        self.square_queries = nn.Parameter(torch.zeros(64, embed_dim))
        self.square_coord_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.patch_position_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.register_buffer(
            "canonical_square_coords",
            _build_canonical_square_coordinates(),
            persistent=False,
        )
        nn.init.normal_(self.square_queries, std=0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        spatial_tokens = _extract_spatial_tokens(patch_tokens)
        batch_size = spatial_tokens.shape[0]
        grid_size = _infer_grid_size(spatial_tokens.shape[1])
        patch_positions = _build_patch_positions(
            grid_size,
            batch_size=batch_size,
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        patch_features = spatial_tokens + self.patch_position_proj(patch_positions)
        square_coords = self.canonical_square_coords.to(
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        queries = self.square_queries.unsqueeze(0) + self.square_coord_proj(
            square_coords.unsqueeze(0).expand(batch_size, -1, -1)
        )
        attended, _ = self.cross_attn(
            self.query_norm(queries),
            self.patch_norm(patch_features),
            self.patch_norm(patch_features),
            need_weights=False,
        )
        square_tokens = queries + attended
        square_tokens = square_tokens + self.ffn(self.ffn_norm(square_tokens))
        result: torch.Tensor = self.output_norm(square_tokens)
        return result


class DirectPhysicalBoardReader(nn.Module):
    """Whole-image board reader using a pretrained dense encoder and 64 square queries."""

    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        config: DirectBoardReaderConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        embed_dim = int(vision_encoder.embed_dim)
        self.square_decoder = DirectSquareQueryDecoder(
            embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
        )
        self.square_head = PhysicalBoardStateProbe(
            embed_dim=embed_dim,
            num_classes=config.num_classes,
            head_type=config.head_type,
            hidden_dim=config.hidden_dim,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
            transformer_ff_dim=config.transformer_ff_dim,
            dropout=config.dropout,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.vision_encoder.forward_patches(images)
        square_tokens = self.square_decoder(patch_tokens)
        logits: torch.Tensor = self.square_head(square_tokens)
        return logits

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "reader_config": {
                "input_size": self.config.input_size,
                "num_classes": self.config.num_classes,
                "num_heads": self.config.num_heads,
                "dropout": self.config.dropout,
                "mlp_ratio": self.config.mlp_ratio,
                "head_type": self.config.head_type,
                "hidden_dim": self.config.hidden_dim,
                "transformer_layers": self.config.transformer_layers,
                "transformer_heads": self.config.transformer_heads,
                "transformer_ff_dim": self.config.transformer_ff_dim,
            }
        }
