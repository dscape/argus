"""Frozen DINO board-context probe for physical per-square state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.square_probe import ProbeMetrics, evaluate_probe
from pipeline.shared import NUM_SQUARE_CLASSES


class PhysicalBoardStateProbe(nn.Module):
    """Shared linear readout applied to 64 square tokens."""

    def __init__(self, embed_dim: int, num_classes: int = NUM_SQUARE_CLASSES) -> None:
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(square_tokens)


def dino_patches_to_square_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Pool patch tokens into 8x8 square tokens.

    Accepts either DINO-style tokens with a leading CLS token or plain square
    grids such as the YOLO-derived frontend.
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected (B, N, D) patch tokens, got {tuple(patch_tokens.shape)}")

    token_count = patch_tokens.shape[1]
    cls_grid_size = int((token_count - 1) ** 0.5)
    plain_grid_size = int(token_count**0.5)

    if cls_grid_size * cls_grid_size == token_count - 1:
        tokens = patch_tokens[:, 1:, :]
        grid_size = cls_grid_size
    elif plain_grid_size * plain_grid_size == token_count:
        tokens = patch_tokens
        grid_size = plain_grid_size
    else:
        raise ValueError(
            f"Token count {token_count} is neither a square grid nor a square grid plus CLS"
        )

    if grid_size % 8 != 0:
        raise ValueError(
            f"Patch grid {grid_size}x{grid_size} cannot map cleanly to 8x8 squares"
        )

    batch_size, patch_count, embed_dim = tokens.shape
    patches_per_square = grid_size // 8
    reshaped = tokens.reshape(batch_size, grid_size, grid_size, embed_dim)
    square_tokens = reshaped.reshape(
        batch_size,
        8,
        patches_per_square,
        8,
        patches_per_square,
        embed_dim,
    ).mean(dim=(2, 4))
    return square_tokens.reshape(batch_size, 64, embed_dim)


def extract_square_token_features(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    encoder: VisionEncoder,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for images, labels in loader:
            patch_tokens = encoder.forward_patches(images.to(device)).cpu()
            square_tokens = dino_patches_to_square_tokens(patch_tokens)
            feature_batches.append(square_tokens)
            label_batches.append(labels.cpu())
    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)


def train_board_probe(
    train_square_tokens: torch.Tensor,
    train_labels: torch.Tensor,
    val_square_tokens: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> tuple[PhysicalBoardStateProbe, float]:
    probe = PhysicalBoardStateProbe(train_square_tokens.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device)
    )

    train_features = train_square_tokens.to(device)
    train_targets = train_labels.to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0

    for _epoch in range(epochs):
        probe.train()
        logits = probe(train_features)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), train_targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = evaluate_board_probe(probe, val_square_tokens, val_labels, device=device)
        if metrics.accuracy > best_val_accuracy:
            best_val_accuracy = metrics.accuracy
            best_state = {
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    best_probe = PhysicalBoardStateProbe(train_square_tokens.shape[-1])
    best_probe.load_state_dict(best_state)
    return best_probe.to(device), best_val_accuracy


def evaluate_board_probe(
    probe: PhysicalBoardStateProbe,
    square_tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    annotation_ids: list[str] | None = None,
) -> ProbeMetrics:
    probe.eval()
    with torch.no_grad():
        logits = probe(square_tokens.to(device)).cpu()
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    flattened_labels = labels.reshape(-1)
    board_annotation_ids = None
    if annotation_ids is not None:
        board_annotation_ids = [
            annotation_id for annotation_id in annotation_ids for _ in range(64)
        ]
    return evaluate_probe(
        _ProbeAdapter(),
        flattened_logits,
        flattened_labels,
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )


class _ProbeAdapter(nn.Module):
    def forward(self, logits_as_features: torch.Tensor) -> torch.Tensor:
        # ``evaluate_probe`` expects embeddings -> logits; here features are already logits.
        return logits_as_features


def save_board_probe_checkpoint(
    path: str | Path,
    *,
    probe: PhysicalBoardStateProbe,
    model_name: str,
    input_size: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": probe.state_dict(),
        "model_name": model_name,
        "input_size": input_size,
        "num_classes": NUM_SQUARE_CLASSES,
        "metadata": metadata or {},
        "architecture": "board_probe",
    }
    torch.save(payload, Path(path))
