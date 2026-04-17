from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pipeline.physical.two_stage_board_reader import read_board

_CORNERS = ((10.0, 20.0), (90.0, 20.0), (90.0, 100.0), (10.0, 100.0))


class _FixedLogitsModel(nn.Module):
    """Stub classifier that returns a fixed logit vector per sample, plus a config shim."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

        class _Config:
            num_classes = logits.shape[-1]

        self.config = _Config()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return self._logits.unsqueeze(0).expand(batch, -1).to(images.device)


def _blank_image() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


def test_read_board_labels_all_squares_empty_when_occupancy_predicts_empty() -> None:
    occupancy = _FixedLogitsModel(torch.tensor([5.0, -5.0]))  # strongly empty
    piece = _FixedLogitsModel(torch.zeros(12))

    result = read_board(
        _blank_image(),
        _CORNERS,
        occupancy_model=occupancy,
        piece_model=piece,
        occupancy_crop_size=32,
        piece_crop_size=64,
    )

    assert result.class_ids == (0,) * 64
    assert all(prob < 0.5 for prob in result.occupancy_probs)


def test_read_board_routes_occupied_squares_to_piece_classifier() -> None:
    occupancy = _FixedLogitsModel(torch.tensor([-5.0, 5.0]))  # strongly occupied
    piece_logits = torch.full((12,), -3.0)
    piece_logits[5] = 3.0  # predict class 5 -> white king -> square class 6
    piece = _FixedLogitsModel(piece_logits)

    result = read_board(
        _blank_image(),
        _CORNERS,
        occupancy_model=occupancy,
        piece_model=piece,
        occupancy_crop_size=32,
        piece_crop_size=64,
    )

    assert result.class_ids == (6,) * 64
    assert all(prob > 0.5 for prob in result.occupancy_probs)
    # argmax probability for the predicted class should dominate
    for row in result.piece_probs:
        assert row[5] > 0.5


def test_read_board_preserves_classifier_train_mode_state() -> None:
    occupancy = _FixedLogitsModel(torch.tensor([-5.0, 5.0]))
    piece_logits = torch.full((12,), 0.0)
    piece = _FixedLogitsModel(piece_logits)
    occupancy.train()
    piece.train()

    read_board(
        _blank_image(),
        _CORNERS,
        occupancy_model=occupancy,
        piece_model=piece,
        occupancy_crop_size=32,
        piece_crop_size=64,
    )

    assert occupancy.training
    assert piece.training
