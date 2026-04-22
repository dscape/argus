"""Stage-2 foreground isolation for templates-v2.

Three patch-grid masks per proposal crop, all aligned to the same 14×14 grid:

- ``pca_mask``  : sign of the first principal component of DINOv3 patch tokens
                  at a mid-late layer. Separates the piece from background when
                  it is the dominant source of feature variance inside the crop.
- ``geometric_mask`` : the visibility mask from Stage 1 (0..1), i.e. which
                        patches lie inside the occluder-subtracted piece cuboid.
- ``intersect_mask`` : per-patch product of the two — a patch is foreground
                        only if DINOv3 agrees it is piece-like AND geometry
                        says it is visible.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_GEOMETRY_DIR = _THIS_DIR.parent / "geometry"
for _path in (_PROJECT_ROOT, _GEOMETRY_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import numpy as np
import torch

from pipeline.physical.two_stage.classifier_data import preprocess_square_crop
from study.templates.inference.embedder import FrozenBackboneEmbedder, get_embedder

from visibility import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_PATCH_SIZE,
    FrameVisibility,
    PieceVisibility,
)

# ViT-B has 12 transformer blocks. The plan called out layer 18 for ViT-L
# (≈75% of its 24 layers); 9/12 for ViT-B hits the same relative depth.
DEFAULT_PCA_LAYER_INDEX = 9


@dataclass(frozen=True)
class ForegroundMasks:
    pca: np.ndarray
    geometric: np.ndarray
    intersect: np.ndarray


def _patch_grid_shape(input_size: int, patch_size: int) -> tuple[int, int]:
    if input_size % patch_size != 0:
        raise ValueError(
            f"input_size ({input_size}) must be a multiple of patch_size ({patch_size})"
        )
    count = input_size // patch_size
    return count, count


def _extract_patch_tokens(
    embedder: FrozenBackboneEmbedder,
    crop_bgr: np.ndarray,
    *,
    layer_index: int,
    input_size: int,
) -> np.ndarray:
    """Run DINOv3 on a single crop and return the patch tokens at ``layer_index``.

    The returned array has shape ``(num_patches, hidden_size)``. The CLS
    token at position 0 and any register tokens are skipped. ``layer_index``
    is 0-based over the transformer blocks (0 .. num_hidden_layers-1).
    """
    tensor = preprocess_square_crop(crop_bgr, size=input_size, augment=False)
    batch = tensor.unsqueeze(0).to(embedder.device)
    backbone = embedder.encoder.backend
    num_hidden_layers = int(backbone.model.config.num_hidden_layers)
    if layer_index < 0 or layer_index >= num_hidden_layers:
        raise ValueError(
            f"layer_index must be in [0, {num_hidden_layers}), got {layer_index}"
        )
    with torch.no_grad():
        outputs = backbone.model(pixel_values=batch, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("DINOv3 did not return hidden states")
    # ``hidden_states`` has length ``num_hidden_layers + 1``; index 0 is
    # the token embeddings pre-transformer, index i+1 is the output of
    # transformer block i. We want the output of block ``layer_index``.
    selected = hidden_states[layer_index + 1][0]
    skip = 1 + int(backbone.num_register_tokens)
    patch_tokens = selected[skip:, :].detach().cpu().numpy()
    return patch_tokens.astype(np.float32)


def _first_pc_signed_projection(patch_tokens: np.ndarray) -> np.ndarray:
    """Project ``patch_tokens`` onto the first principal component and return
    the signed projection per patch."""
    if patch_tokens.ndim != 2:
        raise ValueError(f"Expected 2D patch tokens, got shape {patch_tokens.shape}")
    centered = patch_tokens - patch_tokens.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    return centered @ pc1


def pca_mask(
    crop_bgr: np.ndarray,
    embedder: FrozenBackboneEmbedder,
    *,
    layer_index: int = DEFAULT_PCA_LAYER_INDEX,
    input_size: int = DEFAULT_INPUT_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    disambiguation_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Binary foreground mask on the patch grid from DINOv3 first-PC sign.

    The sign of the first principal component separates patches into two
    clusters but the sign itself is arbitrary. ``disambiguation_hint`` — if
    provided — is an approximate foreground mask (same shape as the patch
    grid); the sign whose ``+`` cluster overlaps the hint more is kept as
    foreground. Without a hint we fall back to "bottom-left region =
    foreground" since that matches where pieces sit after the extractor's
    bottom-left aspect-preserving paste.
    """
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError(f"crop_bgr must be HxWx3, got {crop_bgr.shape}")
    patch_grid_shape = _patch_grid_shape(input_size, patch_size)
    patch_tokens = _extract_patch_tokens(
        embedder, crop_bgr, layer_index=layer_index, input_size=input_size
    )
    if patch_tokens.shape[0] != patch_grid_shape[0] * patch_grid_shape[1]:
        raise RuntimeError(
            f"Expected {patch_grid_shape[0] * patch_grid_shape[1]} patch tokens, "
            f"got {patch_tokens.shape[0]}"
        )
    projection = _first_pc_signed_projection(patch_tokens)
    grid = projection.reshape(patch_grid_shape)

    if disambiguation_hint is not None:
        if disambiguation_hint.shape != grid.shape:
            raise ValueError(
                f"disambiguation_hint shape {disambiguation_hint.shape} does not match "
                f"patch grid {grid.shape}"
            )
        hint_binary = disambiguation_hint > 0.3
        if hint_binary.any():
            pos_overlap = int(((grid > 0) & hint_binary).sum())
            neg_overlap = int(((grid < 0) & hint_binary).sum())
            if neg_overlap > pos_overlap:
                grid = -grid
        else:
            # Hint is empty — fall back to geometric heuristic below.
            disambiguation_hint = None

    if disambiguation_hint is None:
        h, w = grid.shape
        bottom_left = grid[h // 2 :, : max(1, w // 2)]
        if bottom_left.mean() < 0:
            grid = -grid
    return (grid > 0).astype(np.float32)


def geometric_mask(piece: PieceVisibility) -> np.ndarray:
    """The Stage 1 patch-visibility mask for a single piece."""
    return np.asarray(piece.patch_visibility, dtype=np.float32)


def intersect_mask(
    pca: np.ndarray,
    geometric: np.ndarray,
    *,
    geometric_threshold: float = 0.3,
) -> np.ndarray:
    """Per-patch foreground = PCA foreground AND geometric visibility.

    The geometric mask is continuous (0..1) while PCA is binary; we treat a
    geometric patch as "present" once it crosses ``geometric_threshold``.
    """
    if pca.shape != geometric.shape:
        raise ValueError(
            f"pca shape {pca.shape} does not match geometric shape {geometric.shape}"
        )
    pca_binary = (pca > 0.5).astype(np.float32)
    geom_binary = (geometric > float(geometric_threshold)).astype(np.float32)
    return (pca_binary * geom_binary).astype(np.float32)


def compute_foreground_masks(
    crop_bgr: np.ndarray,
    piece: PieceVisibility,
    embedder: FrozenBackboneEmbedder,
    *,
    layer_index: int = DEFAULT_PCA_LAYER_INDEX,
    input_size: int = DEFAULT_INPUT_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    geometric_threshold: float = 0.3,
) -> ForegroundMasks:
    geometric = geometric_mask(piece)
    pca = pca_mask(
        crop_bgr,
        embedder,
        layer_index=layer_index,
        input_size=input_size,
        patch_size=patch_size,
        disambiguation_hint=geometric,
    )
    intersect = intersect_mask(pca, geometric, geometric_threshold=geometric_threshold)
    return ForegroundMasks(pca=pca, geometric=geometric, intersect=intersect)


def get_default_embedder(device: str = "cpu") -> FrozenBackboneEmbedder:
    """Cached DINOv3-ViT-B embedder; shared across foreground-mask calls."""
    return get_embedder(encoder_type="dinov3", device=device)


__all__ = [
    "DEFAULT_PCA_LAYER_INDEX",
    "ForegroundMasks",
    "compute_foreground_masks",
    "geometric_mask",
    "get_default_embedder",
    "intersect_mask",
    "pca_mask",
]
