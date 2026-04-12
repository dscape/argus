from __future__ import annotations

import torch
from pipeline.physical.board_probe import dino_patches_to_square_tokens


def test_dino_patches_to_square_tokens_pools_16x16_grid_into_64_tokens() -> None:
    patch_tokens = torch.arange((1 + 16 * 16) * 2, dtype=torch.float32).reshape(1, 1 + 16 * 16, 2)

    square_tokens = dino_patches_to_square_tokens(patch_tokens)

    assert square_tokens.shape == (1, 64, 2)
