from __future__ import annotations

import torch
from pipeline.physical.board_probe import (
    PhysicalBoardStateProbe,
    board_probe_config_from_checkpoint,
    dino_patches_to_square_tokens,
)


def test_dino_patches_to_square_tokens_pools_16x16_grid_into_64_tokens() -> None:
    patch_tokens = torch.arange((1 + 16 * 16) * 2, dtype=torch.float32).reshape(1, 1 + 16 * 16, 2)

    square_tokens = dino_patches_to_square_tokens(patch_tokens)

    assert square_tokens.shape == (1, 64, 2)


def test_dino_patches_to_square_tokens_accepts_plain_16x16_grid() -> None:
    patch_tokens = torch.arange(16 * 16 * 2, dtype=torch.float32).reshape(1, 16 * 16, 2)

    square_tokens = dino_patches_to_square_tokens(patch_tokens)

    assert square_tokens.shape == (1, 64, 2)


def test_board_probe_config_from_checkpoint_defaults_to_linear_head() -> None:
    assert board_probe_config_from_checkpoint({}) == {
        "head_type": "linear",
        "hidden_dim": 512,
        "transformer_layers": 2,
        "transformer_heads": 8,
        "transformer_ff_dim": 1024,
        "dropout": 0.1,
    }


def test_physical_board_state_probe_transformer_outputs_square_logits() -> None:
    probe = PhysicalBoardStateProbe(16, head_type="transformer", transformer_layers=1)

    logits = probe(torch.randn(2, 64, 16))

    assert logits.shape == (2, 64, 13)
