"""DINOv2 Vision Encoder for board and piece feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Dinov2Model


class VisionEncoder(nn.Module):
    """DINOv2 ViT wrapper for extracting visual features."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        frozen: bool = True,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        self.embed_dim = embed_dim
        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze()
        for layer in self.model.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.forward_patches(x)
        return patches[:, 1:, :].mean(dim=1)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        if return_patches:
            return self.forward_patches(x)
        return self.forward_pooled(x)
