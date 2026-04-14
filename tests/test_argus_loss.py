from __future__ import annotations

import pytest
import torch

from argus.model.losses import ArgusLoss


def test_move_loss_ignores_illegal_logits_when_legal_masks_are_provided() -> None:
    loss_fn = ArgusLoss(w_move=1.0, w_detect=0.0)
    move_targets = torch.tensor([[1]], dtype=torch.long)
    detect_targets = torch.zeros(1, 1, dtype=torch.float32)
    move_mask = torch.tensor([[True]], dtype=torch.bool)
    legal_masks = torch.tensor([[[False, True, True, False]]], dtype=torch.bool)

    losses_a = loss_fn(
        move_logits=torch.tensor([[[100.0, 0.0, 0.0, -100.0]]]),
        detect_logits=torch.zeros(1, 1),
        move_targets=move_targets,
        detect_targets=detect_targets,
        move_mask=move_mask,
        legal_masks=legal_masks,
    )
    losses_b = loss_fn(
        move_logits=torch.tensor([[[-100.0, 0.0, 0.0, 100.0]]]),
        detect_logits=torch.zeros(1, 1),
        move_targets=move_targets,
        detect_targets=detect_targets,
        move_mask=move_mask,
        legal_masks=legal_masks,
    )

    assert torch.allclose(losses_a["move"], losses_b["move"])


def test_move_loss_is_zero_when_only_target_move_is_legal() -> None:
    loss_fn = ArgusLoss(w_move=1.0, w_detect=0.0)
    losses = loss_fn(
        move_logits=torch.zeros(1, 1, 4),
        detect_logits=torch.zeros(1, 1),
        move_targets=torch.tensor([[2]], dtype=torch.long),
        detect_targets=torch.zeros(1, 1, dtype=torch.float32),
        move_mask=torch.tensor([[True]], dtype=torch.bool),
        legal_masks=torch.tensor([[[False, False, True, False]]], dtype=torch.bool),
    )

    assert losses["move"].item() == pytest.approx(0.0)


def test_move_loss_supports_weighted_all_frame_supervision() -> None:
    loss_fn = ArgusLoss(w_move=1.0, w_detect=0.0)
    move_logits = torch.tensor([[[0.0, 4.0, -4.0], [4.0, 0.0, -4.0]]])
    move_targets = torch.tensor([[1, 0]], dtype=torch.long)
    detect_targets = torch.zeros(1, 2, dtype=torch.float32)
    move_mask = torch.tensor([[True, False]], dtype=torch.bool)
    move_loss_mask = torch.tensor([[True, True]], dtype=torch.bool)
    move_loss_weights = torch.tensor([[1.0, 0.1]], dtype=torch.float32)
    legal_masks = torch.tensor([[[True, True, False], [True, True, False]]], dtype=torch.bool)

    losses = loss_fn(
        move_logits=move_logits,
        detect_logits=torch.zeros(1, 2),
        move_targets=move_targets,
        detect_targets=detect_targets,
        move_mask=move_mask,
        move_loss_mask=move_loss_mask,
        move_loss_weights=move_loss_weights,
        legal_masks=legal_masks,
    )

    assert losses["move"].item() < 0.05
