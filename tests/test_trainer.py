from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from argus.chess.constraint_mask import apply_constraint_mask
from argus.data.collate import argus_collate_fn
from argus.training.trainer import Trainer, compute_num_optimizer_steps, compute_warmup_steps
from argus.types import ModelOutput


class TinyArgusModel(nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.model_config = {"vocab_size": vocab_size, "tiny": True}
        self.move_bias = nn.Parameter(torch.zeros(vocab_size))
        self.detect_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        crops: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> ModelOutput:
        assert crops is not None
        batch_size, seq_len = crops.shape[:2]
        move_logits = self.move_bias.view(1, 1, 1, -1).expand(batch_size, seq_len, 1, -1)
        if legal_masks is not None:
            masked_logits = apply_constraint_mask(move_logits.squeeze(2), legal_masks)
            move_probs = masked_logits.softmax(dim=-1).unsqueeze(2)
        else:
            move_probs = move_logits.softmax(dim=-1)
        detect_logits = self.detect_bias.view(1, 1, 1).expand(batch_size, seq_len, 1)
        return ModelOutput(
            move_logits=move_logits,
            move_probs=move_probs,
            detect_logits=detect_logits,
        )


def _make_batch_item() -> dict[str, torch.Tensor]:
    return {
        "frames": torch.zeros(1, 3, 2, 2, dtype=torch.float32),
        "move_targets": torch.tensor([1], dtype=torch.long),
        "detect_targets": torch.tensor([1.0], dtype=torch.float32),
        "legal_masks": torch.tensor([[False, True, True, False]], dtype=torch.bool),
        "move_mask": torch.tensor([True], dtype=torch.bool),
    }


def test_compute_num_optimizer_steps_uses_ceiling_division() -> None:
    assert compute_num_optimizer_steps(0, 4) == 0
    assert compute_num_optimizer_steps(1, 4) == 1
    assert compute_num_optimizer_steps(19, 4) == 5


def test_compute_warmup_steps_scales_down_short_runs() -> None:
    assert compute_warmup_steps(0, 50) == 0
    assert compute_warmup_steps(100, 1) == 0
    assert compute_warmup_steps(25, 200) == 25
    assert compute_warmup_steps(1000, 50) == 5


def test_train_epoch_flushes_final_partial_gradient_accumulation(tmp_path) -> None:
    train_loader = DataLoader([_make_batch_item()], batch_size=1, collate_fn=argus_collate_fn)
    model = TinyArgusModel()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=0.1,
        weight_decay=0.0,
        warmup_steps=0,
        total_steps=1,
        clip_grad_norm=1.0,
        gradient_accumulation=4,
        precision="fp32",
        loss_weights={"move": 1.0, "detect": 0.0},
        output_dir=str(tmp_path),
        save_every=10,
        use_wandb=False,
        device="cpu",
    )

    initial_move_bias = model.move_bias.detach().clone()
    metrics = trainer.train_epoch(epoch=1)

    assert trainer.global_step == 1
    assert metrics["move"] > 0.0
    assert not torch.allclose(model.move_bias.detach(), initial_move_bias)


def test_save_checkpoint_includes_model_config(tmp_path) -> None:
    train_loader = DataLoader([_make_batch_item()], batch_size=1, collate_fn=argus_collate_fn)
    model = TinyArgusModel(vocab_size=7)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=0.1,
        weight_decay=0.0,
        warmup_steps=0,
        total_steps=1,
        clip_grad_norm=1.0,
        gradient_accumulation=1,
        precision="fp32",
        loss_weights={"move": 1.0, "detect": 0.0},
        output_dir=str(tmp_path),
        save_every=10,
        use_wandb=False,
        device="cpu",
    )

    ckpt_path = trainer.save_checkpoint(epoch=1)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    assert checkpoint["model_config"] == {"vocab_size": 7, "tiny": True}
