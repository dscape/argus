from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from data import (
    DETECTION_CLASS_COUNT,
    NO_PIECE_TYPE_INDEX,
    NO_SQUARE_INDEX,
    SQUARE_OUTPUT_NAMES,
    TYPE_CLASS_NAMES,
    board_index_to_square_name,
    detection_class_to_board_label,
    square_output_index_to_board_index,
)
from scipy.optimize import linear_sum_assignment
from transformers import RTDetrConfig, RTDetrForObjectDetection
from transformers.image_transforms import center_to_corners_format

from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

Architecture = Literal["minimal_detr", "rt_detr"]

MINIMAL_ARCHITECTURE: Architecture = "minimal_detr"
RTDETR_ARCHITECTURE: Architecture = "rt_detr"


@dataclass(frozen=True)
class MinimalDetrConfig:
    num_queries: int = 32
    decoder_layers: int = 3
    dropout: float = 0.1
    num_type_classes: int = len(TYPE_CLASS_NAMES)
    num_square_classes: int = len(SQUARE_OUTPUT_NAMES)


class MinimalDetr(nn.Module):
    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        config: MinimalDetrConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        embed_dim = int(vision_encoder.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.query_embed = nn.Embedding(config.num_queries, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.type_head = nn.Linear(embed_dim, config.num_type_classes)
        self.square_head = nn.Linear(embed_dim, config.num_square_classes)
        self.presence_head = nn.Linear(embed_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.vision_encoder.forward_patches(images)
        patch_tokens = tokens[:, 1:, :]
        batch_size, num_tokens, embed_dim = patch_tokens.shape
        grid_size = int(round(math.sqrt(num_tokens)))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected square patch grid, got {num_tokens} tokens")
        pos = build_2d_sincos_position_embedding(
            height=grid_size,
            width=grid_size,
            embed_dim=embed_dim,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        memory = patch_tokens + pos.unsqueeze(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.query_norm(self.decoder(tgt=queries, memory=memory))
        return {
            "type_logits": self.type_head(decoded),
            "square_logits": self.square_head(decoded),
            "presence_logits": self.presence_head(decoded).squeeze(-1),
        }

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "model_config": {
                "num_queries": self.config.num_queries,
                "decoder_layers": self.config.decoder_layers,
                "dropout": self.config.dropout,
                "num_type_classes": self.config.num_type_classes,
                "num_square_classes": self.config.num_square_classes,
            }
        }


class RtDetrStudyModel(nn.Module):
    def __init__(
        self,
        *,
        detector: RTDetrForObjectDetection,
        pretrained_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.pretrained_model_name = pretrained_model_name

    def forward(
        self,
        images: torch.Tensor,
        *,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ):
        return self.detector(pixel_values=images, labels=labels)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "model_config": self.detector.config.to_dict(),
            "pretrained_model_name": self.pretrained_model_name,
        }


def normalize_architecture(value: object) -> Architecture:
    normalized = str(value or MINIMAL_ARCHITECTURE).strip().lower().replace("-", "_")
    if normalized in {MINIMAL_ARCHITECTURE, "study_detr_minimal"}:
        return MINIMAL_ARCHITECTURE
    if normalized in {RTDETR_ARCHITECTURE, "rtdetr", "study_rtdetr"}:
        return RTDETR_ARCHITECTURE
    raise ValueError(f"Unsupported architecture: {value!r}")


def build_model(
    *,
    architecture: str = MINIMAL_ARCHITECTURE,
    encoder_type: str = "dinov2",
    model_name: str | None = None,
    rtdetr_model_name: str | None = None,
    freeze_encoder: bool = True,
    num_queries: int = 32,
    decoder_layers: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    resolved_architecture = normalize_architecture(architecture)
    if resolved_architecture == MINIMAL_ARCHITECTURE:
        resolved_model_name = model_name or default_model_name_for_encoder_type(encoder_type)
        encoder = VisionEncoder(
            encoder_type=encoder_type,
            model_name=resolved_model_name,
            frozen=freeze_encoder,
        )
        return MinimalDetr(
            vision_encoder=encoder,
            config=MinimalDetrConfig(
                num_queries=num_queries,
                decoder_layers=decoder_layers,
                dropout=dropout,
            ),
        )
    return build_rtdetr_model(
        model_name=rtdetr_model_name,
        num_queries=num_queries,
        decoder_layers=decoder_layers,
    )


def build_rtdetr_model(
    *,
    model_name: str | None = None,
    num_queries: int = 300,
    decoder_layers: int = 6,
) -> RtDetrStudyModel:
    id2label = {index: name for index, name in enumerate(TYPE_CLASS_NAMES[1:])}
    label2id = {name: index for index, name in id2label.items()}
    if model_name is None:
        config = RTDetrConfig(
            num_labels=DETECTION_CLASS_COUNT,
            num_queries=num_queries,
            decoder_layers=decoder_layers,
            id2label=id2label,
            label2id=label2id,
        )
        detector = RTDetrForObjectDetection(config)
        return RtDetrStudyModel(detector=detector, pretrained_model_name=None)

    config = RTDetrConfig.from_pretrained(model_name)
    config.num_labels = DETECTION_CLASS_COUNT
    config.id2label = id2label
    config.label2id = label2id
    config.num_queries = num_queries
    config.decoder_layers = decoder_layers
    detector = RTDetrForObjectDetection.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return RtDetrStudyModel(detector=detector, pretrained_model_name=model_name)


def load_checkpoint(checkpoint_path: str | Path, *, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = normalize_architecture(payload.get("architecture"))
    if architecture == MINIMAL_ARCHITECTURE:
        encoder_cfg = payload["encoder_config"]
        model_cfg = payload["model_config"]
        model = build_model(
            architecture=architecture,
            encoder_type=str(encoder_cfg.get("encoder_type", "dinov2")),
            model_name=encoder_cfg.get("model_name"),
            freeze_encoder=bool(encoder_cfg.get("frozen", True)),
            num_queries=int(model_cfg.get("num_queries", 32)),
            decoder_layers=int(model_cfg.get("decoder_layers", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    else:
        config = RTDetrConfig.from_dict(payload["model_config"])
        detector = RTDetrForObjectDetection(config)
        model = RtDetrStudyModel(
            detector=detector,
            pretrained_model_name=payload.get("pretrained_model_name"),
        )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def forward_with_loss(
    model: nn.Module,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    architecture: str,
    lambda_square: float,
    lambda_presence: float,
) -> tuple[Any, torch.Tensor, dict[str, float]]:
    resolved_architecture = normalize_architecture(architecture)
    if resolved_architecture == MINIMAL_ARCHITECTURE:
        outputs = model(images)
        prepared_targets = [
            {
                "piece_types": target["piece_types"].to(images.device),
                "square_indices": target["square_indices"].to(images.device),
            }
            for target in targets
        ]
        loss, loss_terms = compute_losses(
            outputs,
            prepared_targets,
            lambda_square=lambda_square,
            lambda_presence=lambda_presence,
        )
        return outputs, loss, loss_terms

    prepared_labels = [
        {
            "class_labels": target["class_labels"].to(images.device),
            "boxes": target["boxes"].to(images.device),
        }
        for target in targets
    ]
    outputs = model(images, labels=prepared_labels)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("RT-DETR forward pass did not return a loss")
    loss_terms = {"loss_total": float(loss.item())}
    if outputs.loss_dict is not None:
        for key, value in outputs.loss_dict.items():
            loss_terms[str(key)] = float(value.item() if hasattr(value, "item") else value)
    return outputs, loss, loss_terms


def hungarian_match(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    lambda_square: float,
    lambda_presence: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    type_probs = outputs["type_logits"].softmax(dim=-1)
    square_probs = outputs["square_logits"].softmax(dim=-1)
    presence_probs = torch.sigmoid(outputs["presence_logits"])
    assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index, target in enumerate(targets):
        gt_types = target["piece_types"]
        gt_squares = target["square_indices"]
        if gt_types.numel() == 0:
            assignments.append(
                (
                    torch.empty((0,), dtype=torch.long),
                    torch.empty((0,), dtype=torch.long),
                )
            )
            continue
        type_cost = -torch.log(type_probs[batch_index][:, gt_types].clamp_min(1e-8))
        square_cost = -torch.log(square_probs[batch_index][:, gt_squares].clamp_min(1e-8))
        presence_cost = 1.0 - presence_probs[batch_index].unsqueeze(-1)
        cost_matrix = type_cost + (lambda_square * square_cost) + (lambda_presence * presence_cost)
        row_indices, col_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        assignments.append(
            (
                torch.tensor(row_indices, dtype=torch.long),
                torch.tensor(col_indices, dtype=torch.long),
            )
        )
    return assignments


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    lambda_square: float,
    lambda_presence: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    assignments = hungarian_match(
        outputs,
        targets,
        lambda_square=lambda_square,
        lambda_presence=lambda_presence,
    )
    type_losses: list[torch.Tensor] = []
    square_losses: list[torch.Tensor] = []
    matched_presence_losses: list[torch.Tensor] = []
    unmatched_presence_losses: list[torch.Tensor] = []
    presence_logits = outputs["presence_logits"]
    type_logits = outputs["type_logits"]
    square_logits = outputs["square_logits"]
    bce = nn.BCEWithLogitsLoss()

    for batch_index, (prediction_indices, target_indices) in enumerate(assignments):
        gt_types = targets[batch_index]["piece_types"]
        gt_squares = targets[batch_index]["square_indices"]
        matched_mask = torch.zeros(
            (presence_logits.shape[1],),
            dtype=torch.bool,
            device=presence_logits.device,
        )
        if prediction_indices.numel() > 0:
            prediction_indices = prediction_indices.to(presence_logits.device)
            target_indices = target_indices.to(presence_logits.device)
            matched_mask[prediction_indices] = True
            matched_type_logits = type_logits[batch_index, prediction_indices]
            matched_square_logits = square_logits[batch_index, prediction_indices]
            matched_presence_logits = presence_logits[batch_index, prediction_indices]
            type_losses.append(
                nn.functional.cross_entropy(
                    matched_type_logits, gt_types[target_indices].to(matched_type_logits.device)
                )
            )
            square_losses.append(
                nn.functional.cross_entropy(
                    matched_square_logits,
                    gt_squares[target_indices].to(matched_square_logits.device),
                )
            )
            matched_presence_losses.append(
                bce(
                    matched_presence_logits,
                    torch.ones_like(matched_presence_logits),
                )
            )
        unmatched_logits = presence_logits[batch_index, ~matched_mask]
        if unmatched_logits.numel() > 0:
            unmatched_presence_losses.append(
                bce(unmatched_logits, torch.zeros_like(unmatched_logits))
            )

    loss_type = mean_or_zero(type_losses, device=presence_logits.device)
    loss_square = mean_or_zero(square_losses, device=presence_logits.device)
    loss_matched_presence = mean_or_zero(matched_presence_losses, device=presence_logits.device)
    loss_unmatched_presence = mean_or_zero(unmatched_presence_losses, device=presence_logits.device)
    total = (
        loss_type
        + (lambda_square * loss_square)
        + (lambda_presence * (loss_matched_presence + loss_unmatched_presence))
    )
    return total, {
        "loss_type": float(loss_type.item()),
        "loss_square": float(loss_square.item()),
        "loss_matched_presence": float(loss_matched_presence.item()),
        "loss_unmatched_presence": float(loss_unmatched_presence.item()),
        "loss_total": float(total.item()),
    }


def decode_predictions(
    outputs: Any,
    *,
    architecture: str = MINIMAL_ARCHITECTURE,
    presence_threshold: float = 0.5,
    square_boxes: torch.Tensor | list[torch.Tensor] | None = None,
    image_size: int | None = None,
) -> list[dict[str, object]]:
    resolved_architecture = normalize_architecture(architecture)
    if resolved_architecture == MINIMAL_ARCHITECTURE:
        return decode_minimal_detr_predictions(outputs, presence_threshold=presence_threshold)
    if square_boxes is None:
        raise ValueError("square_boxes are required to decode RT-DETR predictions")
    if image_size is None:
        raise ValueError("image_size is required to decode RT-DETR predictions")
    return decode_rtdetr_predictions(
        outputs,
        score_threshold=presence_threshold,
        square_boxes=square_boxes,
        image_size=image_size,
    )


def decode_minimal_detr_predictions(
    outputs: dict[str, torch.Tensor],
    *,
    presence_threshold: float = 0.5,
) -> list[dict[str, object]]:
    decoded: list[dict[str, object]] = []
    type_probs = outputs["type_logits"].softmax(dim=-1)
    square_probs = outputs["square_logits"].softmax(dim=-1)
    presence_probs = torch.sigmoid(outputs["presence_logits"])
    batch_size = outputs["type_logits"].shape[0]
    for batch_index in range(batch_size):
        board_labels = [0] * 64
        piece_entries: list[tuple[float, str, str | None]] = []
        occupied_squares: set[int] = set()
        scores = presence_probs[batch_index]
        order = torch.argsort(scores, descending=True)
        for query_index in order.tolist():
            presence_score = float(presence_probs[batch_index, query_index].item())
            if presence_score < presence_threshold:
                continue
            type_index = int(type_probs[batch_index, query_index].argmax().item())
            square_index = int(square_probs[batch_index, query_index].argmax().item())
            if type_index == NO_PIECE_TYPE_INDEX:
                continue
            piece_name = TYPE_CLASS_NAMES[type_index]
            type_score = float(type_probs[batch_index, query_index, type_index].item())
            square_score = float(square_probs[batch_index, query_index, square_index].item())
            combined_score = presence_score * type_score * square_score
            if square_index == NO_SQUARE_INDEX:
                piece_entries.append((combined_score, piece_name, None))
                continue
            board_index = square_output_index_to_board_index(square_index)
            if board_index is None or board_index in occupied_squares:
                continue
            occupied_squares.add(board_index)
            board_labels[board_index] = type_index
            piece_entries.append((combined_score, piece_name, SQUARE_OUTPUT_NAMES[square_index]))
        piece_entries.sort(key=lambda item: (-item[0], item[1], item[2] or ""))
        decoded.append(
            {
                "board_labels": tuple(board_labels),
                "pieces": tuple(
                    (piece_name, square_name) for _score, piece_name, square_name in piece_entries
                ),
            }
        )
    return decoded


def decode_rtdetr_predictions(
    outputs: Any,
    *,
    score_threshold: float,
    square_boxes: torch.Tensor | list[torch.Tensor],
    image_size: int,
) -> list[dict[str, object]]:
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    pred_boxes = outputs["pred_boxes"] if isinstance(outputs, dict) else outputs.pred_boxes
    if logits is None or pred_boxes is None:
        raise ValueError("RT-DETR outputs must include logits and pred_boxes")

    scale = torch.tensor(
        [image_size, image_size, image_size, image_size],
        device=pred_boxes.device,
        dtype=pred_boxes.dtype,
    )
    absolute_boxes = center_to_corners_format(pred_boxes) * scale
    square_box_batch = normalize_square_boxes_batch(
        square_boxes,
        batch_size=absolute_boxes.shape[0],
        device=absolute_boxes.device,
        dtype=absolute_boxes.dtype,
    )

    decoded: list[dict[str, object]] = []
    batch_size = logits.shape[0]
    for batch_index in range(batch_size):
        scores, class_indices = torch.sigmoid(logits[batch_index]).max(dim=-1)
        order = torch.argsort(scores, descending=True)
        board_labels = [0] * 64
        piece_entries: list[tuple[float, str, str | None]] = []
        occupied_squares: set[int] = set()
        board_square_boxes = square_box_batch[batch_index]
        for query_index in order.tolist():
            score = float(scores[query_index].item())
            if score < score_threshold:
                continue
            class_index = int(class_indices[query_index].item())
            if class_index < 0 or class_index >= DETECTION_CLASS_COUNT:
                continue
            ious = box_iou(
                absolute_boxes[batch_index, query_index].unsqueeze(0), board_square_boxes
            )
            square_index = int(ious.squeeze(0).argmax().item())
            if float(ious[0, square_index].item()) <= 0.0 or square_index in occupied_squares:
                continue
            occupied_squares.add(square_index)
            board_label = detection_class_to_board_label(class_index)
            board_labels[square_index] = board_label
            piece_entries.append(
                (
                    score,
                    TYPE_CLASS_NAMES[board_label],
                    board_index_to_square_name(square_index),
                )
            )
        piece_entries.sort(key=lambda item: (-item[0], item[1], item[2] or ""))
        decoded.append(
            {
                "board_labels": tuple(board_labels),
                "pieces": tuple(
                    (piece_name, square_name) for _score, piece_name, square_name in piece_entries
                ),
            }
        )
    return decoded


def normalize_square_boxes_batch(
    square_boxes: torch.Tensor | list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(square_boxes, torch.Tensor):
        batch = square_boxes.unsqueeze(0) if square_boxes.ndim == 2 else square_boxes
        if batch.shape != (batch_size, 64, 4):
            raise ValueError(
                f"Expected square_boxes with shape {(batch_size, 64, 4)}, got {tuple(batch.shape)}"
            )
        return batch.to(device=device, dtype=dtype)
    if len(square_boxes) != batch_size:
        raise ValueError(f"Expected {batch_size} square-box tensors, got {len(square_boxes)}")
    tensors = [
        (
            value.to(device=device, dtype=dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=device, dtype=dtype)
        )
        for value in square_boxes
    ]
    batch = torch.stack(tensors, dim=0)
    if batch.shape != (batch_size, 64, 4):
        raise ValueError(
            f"Expected square_boxes with shape {(batch_size, 64, 4)}, got {tuple(batch.shape)}"
        )
    return batch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1 = boxes1.to(torch.float32)
    boxes2 = boxes2.to(torch.float32)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(
        0.0
    )
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(
        0.0
    )
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-6)


def build_2d_sincos_position_embedding(
    *,
    height: int,
    width: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    half_dim = embed_dim // 2
    y_dim = x_dim = half_dim // 2
    y_positions = torch.arange(height, device=device, dtype=dtype)
    x_positions = torch.arange(width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_positions, x_positions, indexing="ij")
    y_embedding = _build_1d_sincos(y_grid.reshape(-1), y_dim)
    x_embedding = _build_1d_sincos(x_grid.reshape(-1), x_dim)
    return torch.cat([y_embedding, x_embedding], dim=-1)


def _build_1d_sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    omega = torch.arange(dim, device=positions.device, dtype=positions.dtype)
    omega = 1.0 / (10000 ** (omega / max(dim, 1)))
    out = positions.unsqueeze(-1) * omega.unsqueeze(0)
    return torch.cat([out.sin(), out.cos()], dim=-1)


def mean_or_zero(values: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(values).mean()


__all__ = [
    "MINIMAL_ARCHITECTURE",
    "RTDETR_ARCHITECTURE",
    "MinimalDetr",
    "MinimalDetrConfig",
    "RtDetrStudyModel",
    "box_iou",
    "build_model",
    "compute_losses",
    "decode_predictions",
    "decode_minimal_detr_predictions",
    "decode_rtdetr_predictions",
    "forward_with_loss",
    "hungarian_match",
    "load_checkpoint",
    "normalize_architecture",
]
