"""Trainable joint oblique board reader on frozen dense patch tokens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from argus.model.oblique_square_decoder import ObliqueSquareQueryDecoder
from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.board_probe import PhysicalBoardStateProbe
from pipeline.physical.square_probe import ProbeMetrics, evaluate_probe


@dataclass(frozen=True)
class JointBoardReaderConfig:
    input_size: int = 224
    num_classes: int = 13
    num_heads: int = 8
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    head_type: str = "pos_mlp"
    hidden_dim: int = 512
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024


class PhysicalJointBoardReader(nn.Module):
    """Decode all 64 squares jointly from one oblique board crop."""

    def __init__(self, embed_dim: int, *, config: JointBoardReaderConfig) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config
        self.square_decoder = ObliqueSquareQueryDecoder(
            embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
        )
        self.square_head = PhysicalBoardStateProbe(
            embed_dim=embed_dim,
            num_classes=config.num_classes,
            head_type=config.head_type,
            hidden_dim=config.hidden_dim,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
            transformer_ff_dim=config.transformer_ff_dim,
            dropout=config.dropout,
        )

    def forward(self, patch_tokens: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        square_tokens = self.square_decoder(
            patch_tokens,
            corners=corners,
            image_size=self.config.input_size,
        )
        return self.square_head(square_tokens)

    def checkpoint_config(self) -> dict[str, float | int]:
        return {
            "input_size": self.config.input_size,
            "num_classes": self.config.num_classes,
            "num_heads": self.config.num_heads,
            "dropout": self.config.dropout,
            "mlp_ratio": self.config.mlp_ratio,
            "head_type": self.config.head_type,
            "hidden_dim": self.config.hidden_dim,
            "transformer_layers": self.config.transformer_layers,
            "transformer_heads": self.config.transformer_heads,
            "transformer_ff_dim": self.config.transformer_ff_dim,
        }


def extract_patch_token_features(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    encoder: VisionEncoder,
    device: torch.device,
    batch_size: int,
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache dense patch tokens + corners for oblique whole-board training."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    patch_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    corner_batches: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError(
                    "Expected oblique board dataset batches shaped (images, labels, corners)"
                )
            images, labels, corners = batch
            patch_tokens = encoder.forward_patches(images.to(device)).cpu().to(storage_dtype)
            patch_batches.append(patch_tokens)
            label_batches.append(labels.cpu())
            corner_batches.append(corners.cpu().to(torch.float32))
    return (
        torch.cat(patch_batches, dim=0),
        torch.cat(label_batches, dim=0),
        torch.cat(corner_batches, dim=0),
    )


def train_joint_board_reader(
    train_patch_tokens: torch.Tensor,
    train_labels: torch.Tensor,
    train_corners: torch.Tensor,
    val_patch_tokens: torch.Tensor,
    val_labels: torch.Tensor,
    val_corners: torch.Tensor,
    *,
    config: JointBoardReaderConfig,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    board_weights: torch.Tensor | None = None,
    train_teacher_logits: torch.Tensor | None = None,
    train_distill_mask: torch.Tensor | None = None,
    distillation_weight: float = 0.0,
    selection_patch_tokens: torch.Tensor | None = None,
    selection_labels: torch.Tensor | None = None,
    selection_corners: torch.Tensor | None = None,
    selection_metric: str = "non_empty_plus_macro",
) -> tuple[PhysicalJointBoardReader, float]:
    model = PhysicalJointBoardReader(train_patch_tokens.shape[-1], config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device),
        reduction="none",
    )

    if board_weights is None:
        weight_tensor = torch.ones(train_patch_tokens.shape[0], dtype=torch.float32)
    else:
        weight_tensor = board_weights.to(torch.float32)
    if train_teacher_logits is None:
        train_teacher_logits = torch.zeros(
            train_labels.shape[0],
            train_labels.shape[1],
            config.num_classes,
            dtype=torch.float32,
        )
    if train_distill_mask is None:
        train_distill_mask = torch.zeros(train_labels.shape[0], dtype=torch.float32)
    train_dataset = TensorDataset(
        train_patch_tokens,
        train_labels,
        train_corners,
        weight_tensor,
        train_teacher_logits,
        train_distill_mask,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    eval_patch_tokens = (
        val_patch_tokens if selection_patch_tokens is None else selection_patch_tokens
    )
    eval_labels = val_labels if selection_labels is None else selection_labels
    eval_corners = val_corners if selection_corners is None else selection_corners

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    for _epoch in range(epochs):
        model.train()
        for (
            patch_tokens,
            labels,
            corners,
            batch_weights,
            teacher_logits,
            distill_mask,
        ) in train_loader:
            patch_tokens = patch_tokens.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)
            corners = corners.to(device=device, dtype=torch.float32)
            logits = model(patch_tokens, corners)
            per_square_loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            per_board_loss = per_square_loss.reshape(labels.shape[0], labels.shape[1]).mean(dim=1)
            if distillation_weight > 0.0:
                teacher_logits = teacher_logits.to(device=device, dtype=torch.float32)
                distill_mask = distill_mask.to(device=device, dtype=torch.float32)
                distill_loss = (
                    F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1),
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean(dim=1)
                )
                per_board_loss = per_board_loss + distillation_weight * distill_mask * distill_loss
            batch_weights = batch_weights.to(device=device, dtype=torch.float32)
            loss = (per_board_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate_joint_board_reader(
            model,
            eval_patch_tokens,
            eval_labels,
            eval_corners,
            device=device,
        )
        score = selection_score_for_metrics(metrics, selection_metric)
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    best_model = PhysicalJointBoardReader(train_patch_tokens.shape[-1], config=config)
    best_model.load_state_dict(best_state)
    return best_model.to(device), best_score


def evaluate_joint_board_reader(
    model: PhysicalJointBoardReader,
    patch_tokens: torch.Tensor,
    labels: torch.Tensor,
    corners: torch.Tensor,
    *,
    device: torch.device,
    annotation_ids: Sequence[str] | None = None,
) -> ProbeMetrics:
    model.eval()
    dataset = TensorDataset(patch_tokens, labels, corners)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    logit_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_patch_tokens, _batch_labels, batch_corners in loader:
            logits = model(
                batch_patch_tokens.to(device=device, dtype=torch.float32),
                batch_corners.to(device=device, dtype=torch.float32),
            )
            logit_batches.append(logits.cpu())
    logits = torch.cat(logit_batches, dim=0)
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    flattened_labels = labels.reshape(-1)
    board_annotation_ids = None
    if annotation_ids is not None:
        board_annotation_ids = [
            annotation_id for annotation_id in annotation_ids for _ in range(64)
        ]
    return evaluate_probe(
        _IdentityProbe(),
        flattened_logits,
        flattened_labels,
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )


def selection_score_for_metrics(metrics: ProbeMetrics, selection_metric: str) -> float:
    if selection_metric == "accuracy":
        return metrics.accuracy
    if selection_metric == "non_empty_accuracy":
        return metrics.non_empty_accuracy
    if selection_metric == "macro_f1":
        return metrics.macro_f1
    if selection_metric == "non_empty_plus_macro":
        return (metrics.non_empty_accuracy + metrics.macro_f1) / 2.0
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def argus_overrides_from_joint_board_reader_checkpoint(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    encoder_config = checkpoint.get("encoder_config")
    reader_config = checkpoint.get("reader_config")
    if not isinstance(encoder_config, dict) or not isinstance(reader_config, dict):
        raise ValueError("Joint board reader checkpoint is missing encoder/reader config")
    return {
        "vision_encoder_type": str(encoder_config["encoder_type"]),
        "vision_encoder_name": str(encoder_config["model_name"]),
        "vision_feature_layer_indices": encoder_config.get("feature_layer_indices"),
        "square_token_mode": "oblique_square_queries",
        "square_query_num_heads": int(reader_config["num_heads"]),
        "square_query_dropout": float(reader_config["dropout"]),
        "square_query_mlp_ratio": float(reader_config["mlp_ratio"]),
        "square_head_type": str(reader_config["head_type"]),
        "square_head_hidden_dim": int(reader_config["hidden_dim"]),
        "square_head_transformer_layers": int(reader_config["transformer_layers"]),
        "square_head_transformer_heads": int(reader_config["transformer_heads"]),
        "square_head_transformer_ff_dim": int(reader_config["transformer_ff_dim"]),
        "square_head_dropout": float(reader_config["dropout"]),
    }


def argus_square_reader_state_dict_from_joint_board_reader_checkpoint(
    checkpoint: dict[str, Any],
) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Joint board reader checkpoint is missing state_dict")
    translated: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("square_decoder."):
            translated[f"square_tokenizer.{key.removeprefix('square_decoder.')}"] = value
        elif key.startswith("square_head."):
            translated[key] = value
    if not translated:
        raise ValueError("Joint board reader checkpoint had no square reader weights")
    return translated


class _IdentityProbe(nn.Module):
    def forward(self, logits_as_features: torch.Tensor) -> torch.Tensor:
        return logits_as_features
