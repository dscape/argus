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
    previous_board_conditioning: str = "none"
    use_previous_side_to_move: bool = True


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


class PreviousBoardConditioner(nn.Module):
    """Inject previous board-state context into the 64 decoded square tokens."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_classes: int,
        conditioning_mode: str,
        use_previous_side_to_move: bool,
    ) -> None:
        super().__init__()
        if conditioning_mode not in {"add", "gated"}:
            raise ValueError(
                "PreviousBoardConditioner requires conditioning mode 'add' or 'gated', got "
                f"{conditioning_mode!r}"
            )
        self.conditioning_mode = conditioning_mode
        self.use_previous_side_to_move = use_previous_side_to_move
        self.previous_label_embed = nn.Embedding(num_classes, embed_dim)
        self.previous_square_coord_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.previous_norm = nn.LayerNorm(embed_dim)
        self.missing_previous_tokens = nn.Parameter(torch.zeros(64, embed_dim))
        self.register_buffer(
            "canonical_square_coords",
            _build_canonical_square_coordinates(),
            persistent=False,
        )
        if use_previous_side_to_move:
            self.previous_side_to_move_embed = nn.Embedding(3, embed_dim)
        else:
            self.previous_side_to_move_embed = None
        if conditioning_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.gate = None
        nn.init.normal_(self.missing_previous_tokens, std=0.02)

    def forward(
        self,
        square_tokens: torch.Tensor,
        *,
        previous_labels: torch.Tensor,
        previous_board_available: torch.Tensor,
        previous_side_to_move: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = square_tokens.shape[0]
        square_coords = self.canonical_square_coords.to(
            device=square_tokens.device,
            dtype=square_tokens.dtype,
        )
        previous_tokens = self.previous_label_embed(previous_labels.to(square_tokens.device))
        previous_tokens = previous_tokens + self.previous_square_coord_proj(
            square_coords.unsqueeze(0).expand(batch_size, -1, -1)
        )
        if self.previous_side_to_move_embed is not None and previous_side_to_move is not None:
            previous_tokens = previous_tokens + self.previous_side_to_move_embed(
                previous_side_to_move.to(square_tokens.device)
            ).unsqueeze(1)
        available = previous_board_available.to(
            device=square_tokens.device,
            dtype=square_tokens.dtype,
        ).view(batch_size, 1, 1)
        missing_tokens = self.missing_previous_tokens.unsqueeze(0).to(
            device=square_tokens.device,
            dtype=square_tokens.dtype,
        )
        previous_tokens = available * previous_tokens + (1.0 - available) * missing_tokens
        previous_tokens = self.previous_norm(previous_tokens)
        if self.conditioning_mode == "add":
            return square_tokens + previous_tokens
        assert self.gate is not None
        gate = torch.sigmoid(self.gate(torch.cat([square_tokens, previous_tokens], dim=-1)))
        return square_tokens + gate * previous_tokens


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
        if config.previous_board_conditioning == "none":
            self.previous_board_conditioner: PreviousBoardConditioner | None = None
        else:
            self.previous_board_conditioner = PreviousBoardConditioner(
                embed_dim=embed_dim,
                num_classes=config.num_classes,
                conditioning_mode=config.previous_board_conditioning,
                use_previous_side_to_move=config.use_previous_side_to_move,
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

    def forward(
        self,
        images: torch.Tensor,
        previous_labels: torch.Tensor | None = None,
        previous_board_available: torch.Tensor | None = None,
        previous_side_to_move: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patch_tokens = self.vision_encoder.forward_patches(images)
        square_tokens = self.square_decoder(patch_tokens)
        if self.previous_board_conditioner is not None:
            batch_size = square_tokens.shape[0]
            if previous_labels is None:
                previous_labels = torch.zeros(
                    (batch_size, 64),
                    dtype=torch.long,
                    device=square_tokens.device,
                )
            if previous_board_available is None:
                previous_board_available = torch.zeros(
                    (batch_size,),
                    dtype=torch.bool,
                    device=square_tokens.device,
                )
            if previous_side_to_move is None:
                previous_side_to_move = torch.zeros(
                    (batch_size,),
                    dtype=torch.long,
                    device=square_tokens.device,
                )
            square_tokens = self.previous_board_conditioner(
                square_tokens,
                previous_labels=previous_labels,
                previous_board_available=previous_board_available,
                previous_side_to_move=previous_side_to_move,
            )
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
                "previous_board_conditioning": self.config.previous_board_conditioning,
                "use_previous_side_to_move": self.config.use_previous_side_to_move,
            }
        }
