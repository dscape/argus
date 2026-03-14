"""Augmentation transforms for Argus training data."""

from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


class TemporalAugmentation:
    """Augmentations applied consistently across temporal frames.

    Ensures the same spatial transform is applied to all frames in a clip
    for temporal consistency.
    """

    def __init__(
        self,
        color_jitter: bool = True,
        random_erasing: bool = True,
        normalize: bool = True,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing
        self.normalize = normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a clip of frames.

        Args:
            frames: (T, C, H, W) float tensor in [0, 1].

        Returns:
            Augmented (T, C, H, W) float tensor.
        """
        T_len = frames.shape[0]

        if self.color_jitter and random.random() < 0.5:
            # Apply same color jitter params to all frames
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.05, 0.05)
            for t in range(T_len):
                frames[t] = TF.adjust_brightness(frames[t], brightness)
                frames[t] = TF.adjust_contrast(frames[t], contrast)
                frames[t] = TF.adjust_saturation(frames[t], saturation)
                frames[t] = TF.adjust_hue(frames[t], hue)

        if self.random_erasing and random.random() < 0.3:
            # Apply same erasing region to all frames
            _, h, w = frames.shape[1:]
            eh = random.randint(h // 8, h // 3)
            ew = random.randint(w // 8, w // 3)
            ei = random.randint(0, h - eh)
            ej = random.randint(0, w - ew)
            fill_value = random.random()
            for t in range(T_len):
                frames[t, :, ei : ei + eh, ej : ej + ew] = fill_value

        if self.normalize:
            mean = torch.tensor(self.image_mean, dtype=frames.dtype).view(1, 3, 1, 1)
            std = torch.tensor(self.image_std, dtype=frames.dtype).view(1, 3, 1, 1)
            frames = (frames - mean) / std

        return frames


class ValidationTransform:
    """Minimal transform for validation/evaluation: just normalize."""

    def __init__(
        self,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize frames.

        Args:
            frames: (T, C, H, W) float tensor in [0, 1].

        Returns:
            Normalized (T, C, H, W) float tensor.
        """
        mean = torch.tensor(self.image_mean, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, dtype=frames.dtype).view(1, 3, 1, 1)
        return (frames - mean) / std


class TemporalSubsample:
    """Subsample or repeat frames to achieve a target clip length."""

    def __init__(self, target_length: int = 16) -> None:
        self.target_length = target_length

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Subsample or repeat frames.

        Args:
            frames: (T, C, H, W) tensor.

        Returns:
            (target_length, C, H, W) tensor.
        """
        T_len = frames.shape[0]
        if T_len == self.target_length:
            return frames
        indices = torch.linspace(0, T_len - 1, self.target_length).long()
        return frames[indices]


class ResizeFrames:
    """Resize all frames in a clip to a target size."""

    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize frames.

        Args:
            frames: (T, C, H, W) tensor.

        Returns:
            (T, C, size, size) tensor.
        """
        return TF.resize(frames, [self.size, self.size], antialias=True)


class ComposeTemporalTransforms:
    """Compose multiple temporal transforms."""

    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            frames = transform(frames)
        return frames
