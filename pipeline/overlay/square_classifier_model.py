"""Tiny CNN for ONNX-exported per-square chess piece classification."""

from __future__ import annotations

import torch
import torch.nn as nn

MODEL_CODE_VERSION = "v2"
NUM_CLASSES = 13
INPUT_SIZE = 64


class ConvBnAct(nn.Sequential):
    """Conv → BatchNorm → SiLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    """Depthwise separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            ConvBnAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels),
            ConvBnAct(in_channels, out_channels, kernel_size=1, padding=0),
        )


class TinySquareClassifier(nn.Module):
    """Small CNN sized for fast batched CPU inference across 64 squares."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(3, 24, kernel_size=3, stride=2),
            DepthwiseSeparableBlock(24, 32),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 96, stride=2),
            DepthwiseSeparableBlock(96, 96),
            DepthwiseSeparableBlock(96, 160, stride=2),
            ConvBnAct(160, 192, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(192, NUM_CLASSES),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
