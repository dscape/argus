"""Frozen DINO board-context probe for physical per-square state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.board_probe.square_probe import ProbeMetrics, evaluate_probe
from pipeline.physical.piece_projection import DEFAULT_PIECE_HEIGHT, project_piece_bboxes
from pipeline.shared import NUM_SQUARE_CLASSES

_DEFAULT_HEAD_TYPE = "linear"
_DEFAULT_HIDDEN_DIM = 512
_DEFAULT_TRANSFORMER_LAYERS = 2
_DEFAULT_TRANSFORMER_HEADS = 8
_DEFAULT_TRANSFORMER_FF_DIM = 1024
_DEFAULT_DROPOUT = 0.1
_SQUARE_COUNT = 64


class PhysicalBoardStateProbe(nn.Module):
    """Trainable frozen-feature readout applied to 64 square tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = NUM_SQUARE_CLASSES,
        *,
        head_type: str = _DEFAULT_HEAD_TYPE,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        transformer_layers: int = _DEFAULT_TRANSFORMER_LAYERS,
        transformer_heads: int = _DEFAULT_TRANSFORMER_HEADS,
        transformer_ff_dim: int = _DEFAULT_TRANSFORMER_FF_DIM,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.dropout = dropout

        if head_type == "linear":
            self.classifier = nn.Linear(embed_dim, num_classes)
            return

        self.position_embedding = nn.Parameter(torch.zeros(_SQUARE_COUNT, embed_dim))
        nn.init.normal_(self.position_embedding, std=0.02)

        if head_type == "pos_mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            return

        if head_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
            self.output_norm = nn.LayerNorm(embed_dim)
            self.classifier = nn.Linear(embed_dim, num_classes)
            return

        raise ValueError(f"Unsupported board-probe head_type: {head_type}")

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        if self.head_type == "linear":
            return self.classifier(square_tokens)
        positioned_tokens = square_tokens + self.position_embedding.unsqueeze(0)
        if self.head_type == "pos_mlp":
            return self.classifier(positioned_tokens)
        contextual_tokens = self.context_encoder(positioned_tokens)
        return self.classifier(self.output_norm(contextual_tokens))

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "head_type": self.head_type,
            "hidden_dim": self.hidden_dim,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "transformer_ff_dim": self.transformer_ff_dim,
            "dropout": self.dropout,
        }


class PhysicalBoardStateEnsembleProbe(nn.Module):
    """Logit-space ensemble over compatible board-state probes."""

    def __init__(
        self,
        probes: list[PhysicalBoardStateProbe],
        *,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if not probes:
            raise ValueError("PhysicalBoardStateEnsembleProbe requires at least one member")
        if weights is None:
            weights = [1.0 / len(probes)] * len(probes)
        if len(weights) != len(probes):
            raise ValueError(f"Expected {len(probes)} ensemble weights, got {len(weights)}")
        self.probes = nn.ModuleList(probes)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, square_tokens: torch.Tensor) -> torch.Tensor:
        member_logits = torch.stack([probe(square_tokens) for probe in self.probes], dim=0)
        weights = self.weights.to(device=square_tokens.device, dtype=member_logits.dtype)
        return (member_logits * weights[:, None, None, None]).sum(dim=0) / weights.sum()


def board_probe_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    probe_config = checkpoint.get("probe_config")
    if not isinstance(probe_config, dict):
        metadata = checkpoint.get("metadata")
        probe_config = metadata if isinstance(metadata, dict) else {}

    return {
        "head_type": str(probe_config.get("head_type", _DEFAULT_HEAD_TYPE)),
        "hidden_dim": int(probe_config.get("hidden_dim", _DEFAULT_HIDDEN_DIM)),
        "transformer_layers": int(
            probe_config.get("transformer_layers", _DEFAULT_TRANSFORMER_LAYERS)
        ),
        "transformer_heads": int(probe_config.get("transformer_heads", _DEFAULT_TRANSFORMER_HEADS)),
        "transformer_ff_dim": int(
            probe_config.get("transformer_ff_dim", _DEFAULT_TRANSFORMER_FF_DIM)
        ),
        "dropout": float(probe_config.get("dropout", _DEFAULT_DROPOUT)),
    }


def build_board_state_probe(
    embed_dim: int,
    *,
    num_classes: int = NUM_SQUARE_CLASSES,
    probe_config: dict[str, Any] | None = None,
) -> PhysicalBoardStateProbe:
    config = board_probe_config_from_checkpoint({"probe_config": probe_config or {}})
    return PhysicalBoardStateProbe(embed_dim, num_classes=num_classes, **config)


def dino_patches_to_square_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Pool patch tokens into 8x8 square tokens.

    Accepts either DINO-style tokens with a leading CLS token or plain square
    grids such as the YOLO-derived frontend.
    """
    tokens, grid_size = _tokens_and_grid_size_from_patch_tokens(patch_tokens)
    if grid_size % 8 != 0:
        raise ValueError(f"Patch grid {grid_size}x{grid_size} cannot map cleanly to 8x8 squares")

    batch_size, _patch_count, embed_dim = tokens.shape
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


def _tokens_and_grid_size_from_patch_tokens(
    patch_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected (B, N, D) patch tokens, got {tuple(patch_tokens.shape)}")

    token_count = patch_tokens.shape[1]
    cls_grid_size = int((token_count - 1) ** 0.5)
    plain_grid_size = int(token_count**0.5)

    if cls_grid_size * cls_grid_size == token_count - 1:
        return patch_tokens[:, 1:, :], cls_grid_size
    if plain_grid_size * plain_grid_size == token_count:
        return patch_tokens, plain_grid_size
    raise ValueError(
        f"Token count {token_count} is neither a square grid nor a square grid plus CLS"
    )


def _project_piece_bboxes_from_corners(
    corners: torch.Tensor,
    *,
    image_size: int,
    piece_height: float,
) -> torch.Tensor:
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError(f"Expected corners shaped (B, 4, 2), got {tuple(corners.shape)}")

    bboxes = np.stack(
        [
            project_piece_bboxes(
                sample_corners.tolist(),
                frame_shape=(image_size, image_size),
                piece_height=piece_height,
            )
            for sample_corners in corners.detach().cpu().numpy()
        ],
        axis=0,
    )
    return torch.as_tensor(bboxes, device=corners.device, dtype=torch.float32)


def _patch_centers(
    *,
    grid_size: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    patch_step = float(image_size) / float(grid_size)
    axis = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) * patch_step
    center_x, center_y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack([center_x.reshape(-1), center_y.reshape(-1)], dim=-1)


def sample_projected_square_tokens_from_patch_tokens(
    patch_tokens: torch.Tensor,
    *,
    geometry: torch.Tensor,
    image_size: int,
    piece_height: float = DEFAULT_PIECE_HEIGHT,
) -> torch.Tensor:
    """Pool patch tokens inside projected piece-box bboxes."""
    tokens, grid_size = _tokens_and_grid_size_from_patch_tokens(patch_tokens)
    batch_size, patch_count, _embed_dim = tokens.shape
    if geometry.shape == (batch_size, 4, 2):
        bboxes = _project_piece_bboxes_from_corners(
            geometry.to(device=tokens.device, dtype=torch.float32),
            image_size=image_size,
            piece_height=piece_height,
        ).to(device=tokens.device, dtype=tokens.dtype)
    elif geometry.shape == (batch_size, 64, 4):
        bboxes = geometry.to(device=tokens.device, dtype=tokens.dtype)
    else:
        raise ValueError(
            "Expected geometry with shape "
            f"({batch_size}, 4, 2) or ({batch_size}, 64, 4), got {tuple(geometry.shape)}"
        )
    patch_centers = _patch_centers(
        grid_size=grid_size,
        image_size=image_size,
        device=tokens.device,
    )
    patch_x = patch_centers[:, 0].view(1, 1, patch_count)
    patch_y = patch_centers[:, 1].view(1, 1, patch_count)

    xmin = bboxes[..., 0].unsqueeze(-1)
    ymin = bboxes[..., 1].unsqueeze(-1)
    xmax = bboxes[..., 2].unsqueeze(-1)
    ymax = bboxes[..., 3].unsqueeze(-1)
    region_mask = (patch_x >= xmin) & (patch_x <= xmax) & (patch_y >= ymin) & (patch_y <= ymax)

    weights = region_mask.to(dtype=tokens.dtype)
    empty_regions = weights.sum(dim=-1) == 0
    if empty_regions.any():
        centers = torch.stack(
            [
                (bboxes[..., 0] + bboxes[..., 2]) / 2.0,
                (bboxes[..., 1] + bboxes[..., 3]) / 2.0,
            ],
            dim=-1,
        )
        distances = torch.sum(
            (patch_centers.view(1, 1, patch_count, 2) - centers.unsqueeze(2)) ** 2,
            dim=-1,
        )
        nearest_patch_index = distances.argmin(dim=-1)
        fallback_weights = torch.zeros_like(weights)
        fallback_weights.scatter_(-1, nearest_patch_index.unsqueeze(-1), 1.0)
        weights = torch.where(empty_regions.unsqueeze(-1), fallback_weights, weights)

    normalized_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("bsp,bpd->bsd", normalized_weights, tokens)


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
        for batch in loader:
            geometry: torch.Tensor | None = None
            if not isinstance(batch, (tuple, list)):
                raise ValueError("Expected dataloader batch to be a tuple or list")
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, geometry = batch
            else:
                raise ValueError(f"Unsupported dataset batch shape: {len(batch)} items")

            if images.ndim == 5:
                batch_size_actual, square_count, channels, height, width = images.shape
                square_images = images.reshape(
                    batch_size_actual * square_count,
                    channels,
                    height,
                    width,
                )
                square_embeddings = encoder.forward_pooled(square_images.to(device))
                square_tokens = square_embeddings.reshape(batch_size_actual, square_count, -1).cpu()
            else:
                patch_tokens = encoder.forward_patches(images.to(device))
                if geometry is None:
                    square_tokens = dino_patches_to_square_tokens(patch_tokens)
                else:
                    square_tokens = sample_projected_square_tokens_from_patch_tokens(
                        patch_tokens,
                        geometry=geometry.to(device),
                        image_size=images.shape[-1],
                    )
                square_tokens = square_tokens.cpu()
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
    board_weights: torch.Tensor | None = None,
    head_type: str = _DEFAULT_HEAD_TYPE,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    transformer_layers: int = _DEFAULT_TRANSFORMER_LAYERS,
    transformer_heads: int = _DEFAULT_TRANSFORMER_HEADS,
    transformer_ff_dim: int = _DEFAULT_TRANSFORMER_FF_DIM,
    dropout: float = _DEFAULT_DROPOUT,
    selection_square_tokens: torch.Tensor | None = None,
    selection_labels: torch.Tensor | None = None,
    selection_metric: str = "accuracy",
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[PhysicalBoardStateProbe, float]:
    probe = PhysicalBoardStateProbe(
        train_square_tokens.shape[-1],
        head_type=head_type,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ff_dim=transformer_ff_dim,
        dropout=dropout,
    ).to(device)
    if initial_state_dict is not None:
        probe.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device),
        reduction="none",
    )

    train_features = train_square_tokens.to(device)
    train_targets = train_labels.to(device)
    train_board_weights = None if board_weights is None else board_weights.to(device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_selection_score = float("-inf")
    eval_square_tokens = (
        val_square_tokens if selection_square_tokens is None else selection_square_tokens
    )
    eval_labels = val_labels if selection_labels is None else selection_labels

    if initial_state_dict is not None:
        metrics = evaluate_board_probe(probe, eval_square_tokens, eval_labels, device=device)
        best_selection_score = selection_score_for_metrics(metrics, selection_metric)
        best_state = {
            key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
        }

    for _epoch in range(epochs):
        probe.train()
        logits = probe(train_features)
        per_square_loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            train_targets.reshape(-1),
        )
        per_board_loss = per_square_loss.reshape(
            train_targets.shape[0],
            train_targets.shape[1],
        ).mean(dim=1)
        if train_board_weights is None:
            loss = per_board_loss.mean()
        else:
            loss = (per_board_loss * train_board_weights).sum() / train_board_weights.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = evaluate_board_probe(probe, eval_square_tokens, eval_labels, device=device)
        selection_score = selection_score_for_metrics(metrics, selection_metric)
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_state = {
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    best_probe = PhysicalBoardStateProbe(
        train_square_tokens.shape[-1],
        head_type=head_type,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ff_dim=transformer_ff_dim,
        dropout=dropout,
    )
    best_probe.load_state_dict(best_state)
    return best_probe.to(device), best_selection_score


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
        "probe_config": probe.checkpoint_config(),
        "architecture": "board_probe",
    }
    torch.save(payload, Path(path))
