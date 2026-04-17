"""Two-stage per-square classifiers (occupancy + piece).

Both stages share the same head architecture — a linear head on pooled
vision-encoder features — and only differ in the number of output classes and
the crop geometry they consume.

- ``OCCUPANCY_NUM_CLASSES`` (2): classes are ``{empty, occupied}``.
- ``PIECE_NUM_CLASSES`` (12): classes map to ``SQUARE_CLASS_NAMES[1:]``, i.e.
  the 6 white + 6 black piece classes, excluding the empty class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from argus.model.vision_encoder import VisionEncoder
from pipeline.shared import SQUARE_CLASS_NAMES

OCCUPANCY_NUM_CLASSES = 2
PIECE_NUM_CLASSES = 12

OCCUPANCY_CLASS_NAMES: tuple[str, ...] = ("empty", "occupied")
PIECE_CLASS_NAMES: tuple[str, ...] = tuple(SQUARE_CLASS_NAMES[1:])


def square_class_to_occupancy_label(class_id: int) -> int:
    """0 for empty, 1 for any piece."""
    return 0 if class_id == 0 else 1


def square_class_to_piece_label(class_id: int) -> int:
    """Map a 13-class SQUARE_CLASS id (>0) to a 12-class piece label."""
    if class_id <= 0 or class_id >= len(SQUARE_CLASS_NAMES):
        raise ValueError(f"square class {class_id} is not a piece class")
    return class_id - 1


def piece_label_to_square_class(piece_label: int) -> int:
    """Inverse of ``square_class_to_piece_label``."""
    if piece_label < 0 or piece_label >= PIECE_NUM_CLASSES:
        raise ValueError(f"piece label out of range: {piece_label}")
    return piece_label + 1


@dataclass(frozen=True)
class SquareClassifierConfig:
    num_classes: int
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")


class SquareClassifier(nn.Module):
    """Linear head on pooled vision-encoder features for one crop → one class."""

    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        config: SquareClassifierConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        embed_dim = int(vision_encoder.embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(config.dropout),
            nn.Linear(embed_dim, config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder.forward_pooled(images)
        return self.head(features)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "classifier_config": {
                "num_classes": self.config.num_classes,
                "dropout": self.config.dropout,
            }
        }


def build_occupancy_classifier(
    vision_encoder: VisionEncoder,
    *,
    dropout: float = 0.1,
) -> SquareClassifier:
    return SquareClassifier(
        vision_encoder=vision_encoder,
        config=SquareClassifierConfig(num_classes=OCCUPANCY_NUM_CLASSES, dropout=dropout),
    )


def build_piece_classifier(
    vision_encoder: VisionEncoder,
    *,
    dropout: float = 0.1,
) -> SquareClassifier:
    return SquareClassifier(
        vision_encoder=vision_encoder,
        config=SquareClassifierConfig(num_classes=PIECE_NUM_CLASSES, dropout=dropout),
    )
