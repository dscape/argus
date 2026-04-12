"""Frozen-feature linear probe utilities for physical square classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.square_data import CLASS_NAMES, NUM_CLASSES


class PhysicalSquareLinearProbe(nn.Module):
    """Linear probe on top of frozen DINOv2 square embeddings."""

    def __init__(self, embed_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


@dataclass(frozen=True)
class ProbeMetrics:
    accuracy: float
    non_empty_accuracy: float
    macro_f1: float
    board_exact_match: float | None
    mean_confidence: float
    class_accuracy: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "non_empty_accuracy": self.non_empty_accuracy,
            "macro_f1": self.macro_f1,
            "board_exact_match": self.board_exact_match,
            "mean_confidence": self.mean_confidence,
            "class_accuracy": self.class_accuracy,
        }


def extract_features(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    encoder: VisionEncoder,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen encoder once and cache square embeddings in memory."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for images, labels in loader:
            embeddings = encoder.forward_pooled(images.to(device)).cpu()
            feature_batches.append(embeddings)
            label_batches.append(labels.cpu())
    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[PhysicalSquareLinearProbe, float]:
    probe = PhysicalSquareLinearProbe(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    for _epoch in range(epochs):
        probe.train()
        logits = probe(train_features)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            val_predictions = val_logits.argmax(dim=1)
            val_accuracy = float((val_predictions == val_labels).float().mean().item())
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    best_probe = PhysicalSquareLinearProbe(train_features.shape[1])
    best_probe.load_state_dict(best_state)
    return best_probe.to(device), best_val_accuracy


def evaluate_probe(
    probe: PhysicalSquareLinearProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    board_annotation_ids: list[str] | None = None,
) -> ProbeMetrics:
    probe.eval()
    with torch.no_grad():
        logits = probe(features.to(device)).cpu()
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)

    labels_np = labels.numpy()
    predictions_np = predictions.numpy()
    confidences_np = probabilities.max(dim=1).values.numpy()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, predicted in zip(labels_np.tolist(), predictions_np.tolist()):
        confusion[target, predicted] += 1

    class_accuracy: dict[str, float] = {}
    f1_scores: list[float] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = float(confusion[class_index, class_index])
        support = float(confusion[class_index].sum())
        predicted_count = float(confusion[:, class_index].sum())
        class_accuracy[class_name] = true_positive / support if support > 0 else 0.0
        precision = true_positive / predicted_count if predicted_count > 0 else 0.0
        recall = true_positive / support if support > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))

    non_empty_mask = labels_np != 0
    non_empty_accuracy = float(
        (predictions_np[non_empty_mask] == labels_np[non_empty_mask]).mean()
    ) if np.any(non_empty_mask) else 0.0

    board_exact_match = None
    if board_annotation_ids is not None and len(board_annotation_ids) == len(labels_np):
        board_predictions: dict[str, list[int]] = {}
        board_labels: dict[str, list[int]] = {}
        for annotation_id, label, prediction in zip(
            board_annotation_ids,
            labels_np.tolist(),
            predictions_np.tolist(),
        ):
            board_predictions.setdefault(annotation_id, []).append(prediction)
            board_labels.setdefault(annotation_id, []).append(label)
        board_matches = [
            int(board_predictions[annotation_id] == board_labels[annotation_id])
            for annotation_id in board_labels
        ]
        board_exact_match = float(np.mean(board_matches)) if board_matches else 0.0

    return ProbeMetrics(
        accuracy=float((predictions_np == labels_np).mean()),
        non_empty_accuracy=non_empty_accuracy,
        macro_f1=float(np.mean(f1_scores)),
        board_exact_match=board_exact_match,
        mean_confidence=float(confidences_np.mean()),
        class_accuracy=class_accuracy,
    )


def save_probe_checkpoint(
    path: str | Path,
    *,
    probe: PhysicalSquareLinearProbe,
    model_name: str,
    input_size: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": probe.state_dict(),
        "model_name": model_name,
        "input_size": input_size,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "metadata": metadata or {},
    }
    torch.save(payload, Path(path))


def load_probe_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid probe checkpoint: {path}")
    return payload
