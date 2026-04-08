"""Pre-compute and cache DINOv2 features for piece classifier training.

Since the DINOv2 encoder is frozen, its output for any given image is
deterministic.  Pre-computing the 768-D feature vectors once converts
training from minutes of ViT forward passes per epoch to seconds of
MLP-only computation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline.overlay.piece_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    MODEL_CODE_VERSION,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_CACHE_DIR = _PROJECT_ROOT / "data" / "overlay" / "dataset" / "torch"

_NORM_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)


def _preprocess_images(images: np.ndarray, batch_size: int = 64) -> list[torch.Tensor]:
    """Convert (N, H, W, 3) uint8 BGR → list of (B, 3, 224, 224) normalised batches."""
    batches: list[torch.Tensor] = []
    n = len(images)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        resized = []
        for img in images[start:end]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized.append(cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)))
        arr = np.stack(resized)
        tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
        tensor = (tensor - _NORM_MEAN) / _NORM_STD
        batches.append(tensor)
    return batches


def extract_and_cache_features(
    images: np.ndarray,
    labels: np.ndarray,
    source_tag: str,
    cache_dir: str | Path = FEATURE_CACHE_DIR,
    device: str = "cpu",
    batch_size: int = 64,
) -> Path:
    """Pre-compute DINOv2 768-D features for all images and save to disk.

    Saves ``{cache_dir}/{source_tag}_features.pt`` containing:
    - ``features``: (N, 768) float32 tensor
    - ``labels``: (N,) long tensor
    - ``source``: str
    - ``model_version``: str
    - ``count``: int
    - ``cached_at``: str (ISO timestamp)

    Returns the cache file path.
    """
    from argus.model.vision_encoder import VisionEncoder

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source_tag}_features.pt"

    encoder = VisionEncoder(frozen=True)
    encoder = encoder.to(device)
    encoder.eval()

    all_features: list[torch.Tensor] = []
    batches = _preprocess_images(images, batch_size)

    logger.info(
        "Extracting features for %d images (%s) on %s…",
        len(images), source_tag, device,
    )

    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch = batch.to(device)
            feats = encoder.forward_pooled(batch)  # (B, 768)
            all_features.append(feats.cpu())
            if (i + 1) % 20 == 0:
                done = min((i + 1) * batch_size, len(images))
                logger.info("  %d/%d images processed", done, len(images))

    features = torch.cat(all_features, dim=0)  # (N, 768)
    labels_tensor = torch.from_numpy(labels).long()

    torch.save(
        {
            "features": features,
            "labels": labels_tensor,
            "source": source_tag,
            "model_version": MODEL_CODE_VERSION,
            "count": len(features),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        },
        cache_path,
    )
    logger.info("Cached %d features → %s", len(features), cache_path)
    return cache_path


def load_cached_features(
    source_tag: str,
    cache_dir: str | Path = FEATURE_CACHE_DIR,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load pre-computed features for a single source.  Returns None if not cached."""
    cache_path = Path(cache_dir) / f"{source_tag}_features.pt"
    if not cache_path.exists():
        return None

    data = torch.load(cache_path, map_location="cpu", weights_only=True)

    # Validate model version
    cached_version = data.get("model_version", "")
    if cached_version != MODEL_CODE_VERSION:
        logger.warning(
            "Cache %s was built with %s, current is %s — ignoring",
            cache_path.name, cached_version, MODEL_CODE_VERSION,
        )
        return None

    logger.info(
        "Loaded %d cached features from %s (%s)",
        data["count"], cache_path.name, data.get("cached_at", "?"),
    )
    return data["features"], data["labels"]


def load_all_cached_features(
    cache_dir: str | Path = FEATURE_CACHE_DIR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate features from all cached sources."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"No feature cache at {cache_dir}")

    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for pt_file in sorted(cache_dir.glob("*_features.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=True)
        cached_version = data.get("model_version", "")
        if cached_version != MODEL_CODE_VERSION:
            logger.warning("Skipping stale cache %s (version %s)", pt_file.name, cached_version)
            continue
        all_features.append(data["features"])
        all_labels.append(data["labels"])
        logger.info("  %s: %d features", pt_file.name, len(data["features"]))

    if not all_features:
        raise FileNotFoundError(f"No valid cached features in {cache_dir}")

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
