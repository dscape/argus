from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type


@dataclass(frozen=True)
class BaseHeadConfig:
    num_type_classes: int
    dropout: float = 0.1


class BaseHeadModel(nn.Module):
    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        config: BaseHeadConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        embed_dim = int(vision_encoder.embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.type_head = nn.Linear(embed_dim, config.num_type_classes)
        self.base_head = nn.Linear(embed_dim, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.vision_encoder.forward_patches(images)
        cls_token = tokens[:, 0, :]
        features = self.dropout(self.norm(cls_token))
        return self.type_head(features), self.base_head(features)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "model_config": {
                "num_type_classes": self.config.num_type_classes,
                "dropout": self.config.dropout,
            }
        }


def build_model(
    *,
    encoder_type: str = "dinov2",
    model_name: str | None = None,
    freeze_encoder: bool = True,
    dropout: float = 0.1,
    num_type_classes: int,
) -> BaseHeadModel:
    resolved_model_name = model_name or default_model_name_for_encoder_type(encoder_type)
    encoder = VisionEncoder(
        encoder_type=encoder_type,
        model_name=resolved_model_name,
        frozen=freeze_encoder,
    )
    return BaseHeadModel(
        vision_encoder=encoder,
        config=BaseHeadConfig(num_type_classes=num_type_classes, dropout=dropout),
    )


def load_checkpoint(checkpoint_path: str | Path, *, device: torch.device) -> BaseHeadModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_cfg = payload["encoder_config"]
    model_cfg = payload["model_config"]
    model = build_model(
        encoder_type=str(encoder_cfg.get("encoder_type", "dinov2")),
        model_name=encoder_cfg.get("model_name"),
        freeze_encoder=bool(encoder_cfg.get("frozen", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_type_classes=int(model_cfg["num_type_classes"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


__all__ = ["BaseHeadConfig", "BaseHeadModel", "build_model", "load_checkpoint"]
