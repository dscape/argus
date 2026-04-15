"""End-to-end trainable oblique whole-board reader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from argus.model.oblique_square_decoder import ObliqueSquareQueryDecoder
from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.board_probe import PhysicalBoardStateProbe
from pipeline.physical.square_probe import ProbeMetrics, evaluate_probe


@dataclass(frozen=True)
class EndToEndJointBoardReaderConfig:
    input_size: int = 224
    num_classes: int = 13
    square_query_num_heads: int = 8
    square_query_dropout: float = 0.0
    square_query_mlp_ratio: float = 4.0
    head_type: str = "pos_mlp"
    hidden_dim: int = 512
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024
    dropout: float = 0.1


class EndToEndPhysicalJointBoardReader(nn.Module):
    """Train the dense encoder and oblique square decoder together."""

    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        config: EndToEndJointBoardReaderConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        embed_dim = int(vision_encoder.embed_dim)
        self.square_decoder = ObliqueSquareQueryDecoder(
            embed_dim,
            num_heads=config.square_query_num_heads,
            dropout=config.square_query_dropout,
            mlp_ratio=config.square_query_mlp_ratio,
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

    def forward(self, images: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.vision_encoder.forward_patches(images)
        square_tokens = self.square_decoder(
            patch_tokens,
            corners=corners,
            image_size=images.shape[-1],
        )
        logits: torch.Tensor = self.square_head(square_tokens)
        return logits

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "reader_config": {
                "input_size": self.config.input_size,
                "num_classes": self.config.num_classes,
                "square_query_num_heads": self.config.square_query_num_heads,
                "square_query_dropout": self.config.square_query_dropout,
                "square_query_mlp_ratio": self.config.square_query_mlp_ratio,
                "head_type": self.config.head_type,
                "hidden_dim": self.config.hidden_dim,
                "transformer_layers": self.config.transformer_layers,
                "transformer_heads": self.config.transformer_heads,
                "transformer_ff_dim": self.config.transformer_ff_dim,
                "dropout": self.config.dropout,
            }
        }


def train_end_to_end_joint_board_reader(
    model: EndToEndPhysicalJointBoardReader,
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    selection_dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    encoder_lr_scale: float,
    weight_decay: float,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    selection_metric: str = "non_empty_plus_macro",
) -> tuple[EndToEndPhysicalJointBoardReader, float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if param.requires_grad and not name.startswith("vision_encoder")
                ],
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if param.requires_grad and name.startswith("vision_encoder")
                ],
                "lr": lr * encoder_lr_scale,
                "weight_decay": weight_decay,
            },
        ]
    )
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device),
        reduction="mean",
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    for _epoch in range(epochs):
        model.train()
        for images, labels, corners in train_loader:
            logits = model(
                images.to(device),
                corners.to(device=device, dtype=torch.float32),
            )
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.to(device).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate_end_to_end_joint_board_reader(
            model,
            selection_dataset,
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

    best_model = EndToEndPhysicalJointBoardReader(
        vision_encoder=model.vision_encoder,
        config=model.config,
    )
    best_model.load_state_dict(best_state)
    return best_model.to(device), best_score


def evaluate_end_to_end_joint_board_reader(
    model: EndToEndPhysicalJointBoardReader,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    annotation_ids: list[str] | None = None,
    batch_size: int = 32,
) -> ProbeMetrics:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels, corners in loader:
            logits = model(
                images.to(device),
                corners.to(device=device, dtype=torch.float32),
            )
            logits_batches.append(logits.cpu())
            label_batches.append(labels.cpu())
    logits = torch.cat(logits_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    flattened_labels = labels.reshape(-1)
    board_annotation_ids = None
    if annotation_ids is not None:
        board_annotation_ids = [annotation_id for annotation_id in annotation_ids for _ in range(64)]
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


class _IdentityProbe(nn.Module):
    def forward(self, logits_as_features: torch.Tensor) -> torch.Tensor:
        return logits_as_features
