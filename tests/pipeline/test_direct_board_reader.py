from __future__ import annotations

import torch
import torch.nn as nn
from pipeline.physical.direct_board_reader import (
    DirectBoardReaderConfig,
    DirectPhysicalBoardReader,
    DirectSquareQueryDecoder,
)


class _FakeVisionEncoder(nn.Module):
    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward_patches(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.randn(batch_size, 17, self.embed_dim)


def test_direct_square_query_decoder_outputs_64_square_tokens() -> None:
    decoder = DirectSquareQueryDecoder(embed_dim=32, num_heads=4, dropout=0.0, mlp_ratio=2.0)
    patch_tokens = torch.randn(2, 17, 32)

    square_tokens = decoder(patch_tokens)

    assert square_tokens.shape == (2, 64, 32)


def test_direct_physical_board_reader_supports_previous_board_conditioning() -> None:
    model = DirectPhysicalBoardReader(
        vision_encoder=_FakeVisionEncoder(embed_dim=32),
        config=DirectBoardReaderConfig(
            num_classes=13,
            num_heads=4,
            dropout=0.0,
            mlp_ratio=2.0,
            hidden_dim=64,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ff_dim=128,
            previous_board_conditioning="gated",
            use_previous_side_to_move=True,
        ),
    )

    logits = model(
        torch.randn(2, 3, 224, 224),
        previous_labels=torch.zeros((2, 64), dtype=torch.long),
        previous_board_available=torch.tensor([True, False]),
        previous_side_to_move=torch.tensor([1, 0], dtype=torch.long),
    )

    assert logits.shape == (2, 64, 13)
