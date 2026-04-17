"""DETR-style piece detector built on a pretrained dense encoder (DINOv2).

This module defines the architecture and inference helper only. Training uses
Hungarian matching + GIoU loss, which is not implemented here yet (bounding-box
annotations do not exist in the project today; see the companion dev-tool).
The detector predicts, per query:

- class logits over (``num_classes`` + 1) slots, where the final slot is
  "no-object".
- a normalized bounding box in ``(cx, cy, w, h)`` ∈ [0, 1] on the rectified
  512×512 board image.

``predict_detections`` turns raw forward output into a flat list of
``PieceDetection`` ready for ``piece_detection_assignment.assign_detections_to_squares``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from argus.model.oblique_square_decoder import _extract_spatial_tokens, _infer_grid_size
from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.piece_detection_assignment import PieceDetection
from pipeline.physical.square_classifiers import PIECE_NUM_CLASSES


@dataclass(frozen=True)
class PieceDetectorConfig:
    num_classes: int = PIECE_NUM_CLASSES
    num_queries: int = 64
    num_decoder_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    image_size: int = 512


class PieceDetector(nn.Module):
    """DETR-style head (learnable queries + transformer decoder) on a dense encoder."""

    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        config: PieceDetectorConfig = PieceDetectorConfig(),
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        embed_dim = int(vision_encoder.embed_dim)

        self.queries = nn.Parameter(torch.zeros(config.num_queries, embed_dim))
        nn.init.normal_(self.queries, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)

        self.class_head = nn.Linear(embed_dim, config.num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_tokens = self.vision_encoder.forward_patches(images)
        memory = _extract_spatial_tokens(patch_tokens)
        # _infer_grid_size exists only for side-effects/validation here; ignore result.
        _infer_grid_size(memory.shape[1])

        batch_size = memory.shape[0]
        queries = self.query_norm(
            self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        )
        memory = self.memory_norm(memory)
        features = self.decoder(queries, memory)
        class_logits = self.class_head(features)  # (batch, num_queries, num_classes + 1)
        bbox_normalized = self.bbox_head(features)  # (batch, num_queries, 4)
        return {"class_logits": class_logits, "bbox_normalized": bbox_normalized}

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "detector_config": {
                "num_classes": self.config.num_classes,
                "num_queries": self.config.num_queries,
                "num_decoder_layers": self.config.num_decoder_layers,
                "num_heads": self.config.num_heads,
                "mlp_ratio": self.config.mlp_ratio,
                "dropout": self.config.dropout,
                "image_size": self.config.image_size,
            }
        }


@torch.no_grad()
def predict_detections(
    detector: PieceDetector,
    images: torch.Tensor,
    *,
    score_threshold: float = 0.25,
) -> list[list[PieceDetection]]:
    """Run inference and return per-image lists of ``PieceDetection`` in pixel space."""
    detector.eval()
    outputs = detector(images)
    class_logits = outputs["class_logits"]  # (batch, Q, C+1)
    bbox_normalized = outputs["bbox_normalized"]  # (batch, Q, 4)
    probs = torch.softmax(class_logits, dim=-1)
    # All classes except the last (no-object) slot.
    piece_probs = probs[..., :-1]
    scores, piece_labels = piece_probs.max(dim=-1)  # (batch, Q)

    image_size = float(detector.config.image_size)
    batch_results: list[list[PieceDetection]] = []
    for batch_index in range(scores.shape[0]):
        detections: list[PieceDetection] = []
        for query_index in range(scores.shape[1]):
            score = float(scores[batch_index, query_index].item())
            if score < score_threshold:
                continue
            cx, cy, w, h = bbox_normalized[batch_index, query_index].tolist()
            xmin = max(0.0, (cx - w / 2.0) * image_size)
            ymin = max(0.0, (cy - h / 2.0) * image_size)
            xmax = min(image_size, (cx + w / 2.0) * image_size)
            ymax = min(image_size, (cy + h / 2.0) * image_size)
            detections.append(
                PieceDetection(
                    piece_label=int(piece_labels[batch_index, query_index].item()),
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    score=score,
                )
            )
        batch_results.append(detections)
    return batch_results
