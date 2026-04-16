from __future__ import annotations

import torch
from pipeline.physical.direct_board_reader import DirectSquareQueryDecoder


def test_direct_square_query_decoder_outputs_64_square_tokens() -> None:
    decoder = DirectSquareQueryDecoder(embed_dim=32, num_heads=4, dropout=0.0, mlp_ratio=2.0)
    patch_tokens = torch.randn(2, 17, 32)

    square_tokens = decoder(patch_tokens)

    assert square_tokens.shape == (2, 64, 32)
