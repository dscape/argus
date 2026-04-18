from __future__ import annotations

import pytest
import torch
from pipeline.physical.board_probe.probe import (
    PhysicalBoardStateProbe,
    board_probe_config_from_checkpoint,
    dino_patches_to_square_tokens,
    extract_square_token_features,
    sample_projected_square_tokens_from_patch_tokens,
    selection_score_for_metrics,
)
from pipeline.physical.board_probe.square_probe import ProbeMetrics
from torch.utils.data import Dataset


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


def test_selection_score_for_metrics_prefers_non_empty_plus_macro() -> None:
    metrics = ProbeMetrics(
        accuracy=0.9,
        non_empty_accuracy=0.4,
        macro_f1=0.2,
        board_exact_match=0.0,
        mean_confidence=0.5,
        class_accuracy={},
    )

    assert selection_score_for_metrics(metrics, "accuracy") == 0.9
    assert selection_score_for_metrics(metrics, "non_empty_accuracy") == 0.4
    assert selection_score_for_metrics(metrics, "macro_f1") == 0.2
    assert selection_score_for_metrics(metrics, "non_empty_plus_macro") == pytest.approx(0.3)


def test_sample_projected_square_tokens_from_constant_patch_grid_stays_constant() -> None:
    patch_tokens = torch.ones((1, 64, 3), dtype=torch.float32)
    corners = torch.tensor(
        [[[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0]]],
        dtype=torch.float32,
    )

    square_tokens = sample_projected_square_tokens_from_patch_tokens(
        patch_tokens,
        corners=corners,
        image_size=16,
    )

    assert square_tokens.shape == (1, 64, 3)
    assert torch.allclose(square_tokens, torch.ones_like(square_tokens))


def test_extract_square_token_features_accepts_pre_cropped_square_batches() -> None:
    class DummyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            images = torch.full((64, 3, 8, 8), float(index), dtype=torch.float32)
            labels = torch.full((64,), index, dtype=torch.long)
            return images, labels

    class DummyEncoder:
        def eval(self) -> None:
            return None

        def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            return torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4)

    tokens, labels = extract_square_token_features(
        DummyDataset(),
        encoder=DummyEncoder(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert tokens.shape == (2, 64, 4)
    assert labels.shape == (2, 64)


def test_extract_square_token_features_accepts_whole_board_oblique_batches() -> None:
    class DummyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            image = torch.full((3, 16, 16), float(index + 1), dtype=torch.float32)
            labels = torch.full((64,), index, dtype=torch.long)
            corners = torch.tensor(
                [[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0]],
                dtype=torch.float32,
            )
            return image, labels, corners

    class DummyEncoder:
        def eval(self) -> None:
            return None

        def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones((x.shape[0], 64, 4), dtype=torch.float32)

    tokens, labels = extract_square_token_features(
        DummyDataset(),
        encoder=DummyEncoder(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert tokens.shape == (2, 64, 4)
    assert labels.shape == (2, 64)
    assert torch.allclose(tokens, torch.ones_like(tokens))
