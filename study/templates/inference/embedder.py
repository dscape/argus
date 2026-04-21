from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pipeline.physical.two_stage.classifier_data import preprocess_square_crop

from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

DEFAULT_ENCODER_TYPE = "dinov3"
DEFAULT_INPUT_SIZE = 224


@dataclass(frozen=True)
class EmbedderConfig:
    encoder_type: str = DEFAULT_ENCODER_TYPE
    model_name: str | None = None
    input_size: int = DEFAULT_INPUT_SIZE
    device: str = "cpu"


class FrozenBackboneEmbedder:
    def __init__(self, *, config: EmbedderConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model_name = self._resolve_model_name(config)
        self.encoder = VisionEncoder(
            encoder_type=config.encoder_type,
            model_name=self.model_name,
            frozen=True,
        )
        self.encoder.to(self.device)
        self.encoder.eval()
        self.embedding_dim = int(self.encoder.embed_dim)

    def embed(self, crop: Any) -> torch.Tensor:
        return self.embed_many([crop])[0]

    def embed_many(self, crops: list[Any]) -> torch.Tensor:
        if not crops:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        batch = torch.stack([self._preprocess_crop(crop) for crop in crops], dim=0).to(self.device)
        with torch.no_grad():
            tokens = self.encoder.forward_patches(batch)
            vectors = tokens[:, 0, :]
        return vectors.detach().cpu()

    def _preprocess_crop(self, crop: Any) -> torch.Tensor:
        crop_bgr = _load_crop_bgr(crop)
        return preprocess_square_crop(crop_bgr, size=self.config.input_size, augment=False)

    @staticmethod
    def _resolve_model_name(config: EmbedderConfig) -> str:
        configured_name = config.model_name or default_model_name_for_encoder_type(
            config.encoder_type
        )
        path_candidate = Path(configured_name)
        if path_candidate.exists():
            return str(path_candidate)
        cache_path = _resolve_hf_cache_model_path(configured_name)
        if cache_path is not None:
            return cache_path
        return configured_name


_EMBEDDER_CACHE: dict[tuple[str, str, int, str], FrozenBackboneEmbedder] = {}


def get_embedder(
    *,
    encoder_type: str = DEFAULT_ENCODER_TYPE,
    model_name: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    device: str = "cpu",
) -> FrozenBackboneEmbedder:
    resolved_model_name = model_name or default_model_name_for_encoder_type(encoder_type)
    cache_key = (encoder_type, resolved_model_name, int(input_size), str(device))
    cached = _EMBEDDER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    embedder = FrozenBackboneEmbedder(
        config=EmbedderConfig(
            encoder_type=encoder_type,
            model_name=resolved_model_name,
            input_size=input_size,
            device=device,
        )
    )
    _EMBEDDER_CACHE[cache_key] = embedder
    return embedder


def embed(
    crop: Any,
    *,
    encoder_type: str = DEFAULT_ENCODER_TYPE,
    model_name: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    device: str = "cpu",
) -> torch.Tensor:
    return get_embedder(
        encoder_type=encoder_type,
        model_name=model_name,
        input_size=input_size,
        device=device,
    ).embed(crop)


def _load_crop_bgr(crop: Any) -> np.ndarray:
    if isinstance(crop, np.ndarray):
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 crop array, got {crop.shape}")
        if crop.dtype == np.uint8:
            return np.ascontiguousarray(crop)
        return np.clip(crop, 0.0, 255.0).astype(np.uint8)

    if isinstance(crop, (str, Path)):
        image = cv2.imread(str(crop), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read crop image: {crop}")
        return image

    raise TypeError(f"Unsupported crop type: {type(crop)!r}")


def _resolve_hf_cache_model_path(model_name: str) -> str | None:
    if "/" not in model_name:
        return None
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_root = hub_root / f"models--{model_name.replace('/', '--')}"
    if not model_root.exists():
        return None
    refs_main = model_root / "refs" / "main"
    if refs_main.exists():
        snapshot = model_root / "snapshots" / refs_main.read_text().strip()
        if snapshot.exists():
            return str(snapshot)
    snapshots_dir = model_root / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


__all__ = [
    "DEFAULT_ENCODER_TYPE",
    "DEFAULT_INPUT_SIZE",
    "EmbedderConfig",
    "FrozenBackboneEmbedder",
    "embed",
    "get_embedder",
]
