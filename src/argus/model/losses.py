"""Loss functions for Argus training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in move detection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class ArgusLoss(nn.Module):
    """Combined loss for Argus training."""

    def __init__(
        self,
        w_move: float = 1.0,
        w_detect: float = 0.5,
        w_bbox: float = 0.0,
        w_identity: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_move = w_move
        self.w_detect = w_detect
        self.w_bbox = w_bbox
        self.w_identity = w_identity
        self.move_loss = nn.CrossEntropyLoss(reduction="mean")
        self.detect_loss = FocalLoss()

    def forward(
        self,
        move_logits: torch.Tensor,
        detect_logits: torch.Tensor,
        move_targets: torch.Tensor,
        detect_targets: torch.Tensor,
        move_mask: torch.Tensor | None = None,
        bbox_pred: torch.Tensor | None = None,
        bbox_target: torch.Tensor | None = None,
        identity_pred: torch.Tensor | None = None,
        identity_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        device = move_logits.device

        if move_mask is not None and move_mask.any():
            active_logits = move_logits[move_mask]
            active_targets = move_targets[move_mask]
            losses["move"] = self.move_loss(active_logits, active_targets)
        else:
            losses["move"] = torch.tensor(0.0, device=device)

        losses["detect"] = self.detect_loss(detect_logits, detect_targets.float())

        if self.w_bbox > 0 and bbox_pred is not None and bbox_target is not None:
            losses["bbox"] = self._bbox_loss(bbox_pred, bbox_target)
        else:
            losses["bbox"] = torch.tensor(0.0, device=device)

        if self.w_identity > 0 and identity_pred is not None and identity_target is not None:
            losses["identity"] = self._identity_loss(identity_pred, identity_target)
        else:
            losses["identity"] = torch.tensor(0.0, device=device)

        losses["total"] = (
            self.w_move * losses["move"]
            + self.w_detect * losses["detect"]
            + self.w_bbox * losses["bbox"]
            + self.w_identity * losses["identity"]
        )
        return losses

    def _bbox_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        giou = self._generalized_iou_loss(pred, target)
        return l1 + giou

    def _generalized_iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1, 4)
        target = target.reshape(-1, 4)
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        giou = iou - (enc_area - union_area) / (enc_area + 1e-6)
        return (1 - giou).mean()

    def _identity_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = embeddings.reshape(-1, embeddings.shape[-1])
        lab = labels.reshape(-1)
        emb = F.normalize(emb, dim=-1)
        sim = emb @ emb.T
        label_match = lab.unsqueeze(0) == lab.unsqueeze(1)
        eye = torch.eye(len(lab), device=emb.device, dtype=torch.bool)
        label_match = label_match & ~eye
        temperature = 0.07
        sim = sim / temperature
        sim = sim.masked_fill(eye, float("-inf"))
        log_probs = F.log_softmax(sim, dim=-1)
        if label_match.any():
            pos_log_probs = log_probs.masked_fill(~label_match, 0.0)
            num_pos = label_match.float().sum(dim=-1).clamp(min=1)
            loss = -(pos_log_probs.sum(dim=-1) / num_pos).mean()
            return loss
        return torch.tensor(0.0, device=emb.device)
