"""Chess-constrained move prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.chess.constraint_mask import apply_constraint_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, VOCAB_SIZE


class MoveHead(nn.Module):
    """Predicts chess moves with legality constraints."""

    def __init__(self, hidden_dim: int = 512, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.move_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.detect_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        move_logits = self.move_proj(features)
        detect_logits = self.detect_proj(features).squeeze(-1)
        if legal_masks is not None:
            masked_logits = apply_constraint_mask(move_logits, legal_masks)
            move_probs = F.softmax(masked_logits, dim=-1)
        else:
            move_probs = F.softmax(move_logits, dim=-1)
        return move_logits, move_probs, detect_logits

    def predict(
        self,
        features: torch.Tensor,
        legal_masks: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, move_probs, detect_logits = self.forward(features, legal_masks)
        predicted_indices = move_probs.argmax(dim=-1)
        confidences = move_probs.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)
        move_detected = torch.sigmoid(detect_logits) > threshold
        return predicted_indices, confidences, move_detected
