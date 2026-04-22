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


def _padding_patch_mask(
    crop_bgr: np.ndarray,
    patch_grid_shape: tuple[int, int],
    patch_size: int,
    *,
    luminance_threshold: float = 4.0,
) -> np.ndarray:
    """Identify patches that are pure zero-padding from ``_place_on_canvas``.

    Returns a bool array shaped ``patch_grid_shape``; ``True`` where the
    patch is entirely dark (padding), ``False`` where there is any image
    content. Padding patches are excluded from PCA and forced to 0 in the
    resulting mask so the PCA ~50/50 split doesn't bleed onto black pixels.
    """
    height = patch_grid_shape[0] * patch_size
    width = patch_grid_shape[1] * patch_size
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.shape != (height, width):
        raise ValueError(
            f"crop shape {gray.shape} does not match patch grid * size {(height, width)}"
        )
    patch_means = gray.reshape(
        patch_grid_shape[0], patch_size, patch_grid_shape[1], patch_size
    ).mean(axis=(1, 3))
    return patch_means < luminance_threshold


def pca_mask(
    crop_bgr: np.ndarray,
    embedder: FrozenBackboneEmbedder,
    *,
    layer_index: int = DEFAULT_PCA_LAYER_INDEX,
    input_size: int = DEFAULT_INPUT_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    piece_symbol: str | None = None,
    disambiguation_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Binary foreground mask on the patch grid from DINOv3 first-PC sign.

    The crop canvas has zero-padding on its top and right (see
    ``_place_on_canvas``) because the extractor preserves aspect ratio and
    bottom-left-pastes the piece. Pure-padding patches are detected by
    luminance and excluded from PCA — otherwise the first PC would spend
    most of its variance separating padding from content, leaving only the
    second PC to separate piece from board.

    Among the remaining content patches, the first principal component
    splits them into two clusters; the sign is arbitrary. When
    ``disambiguation_hint`` is provided (typically the geometric visibility
    mask), the sign whose ``+`` cluster overlaps the hint more is kept as
    foreground. Padding patches are always forced to ``0`` in the output.
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

    padding_mask = _padding_patch_mask(crop_bgr, patch_grid_shape, patch_size)
    content_mask = ~padding_mask
    content_flat = content_mask.ravel()
    if int(content_flat.sum()) < 4:
        return np.zeros(patch_grid_shape, dtype=np.float32)

    content_tokens = patch_tokens[content_flat]
    centered = content_tokens - content_tokens.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    content_projection = centered @ pc1

    grid = np.zeros(patch_grid_shape, dtype=np.float32)
    grid_flat = grid.ravel()
    grid_flat[content_flat] = content_projection.astype(np.float32)
    grid = grid_flat.reshape(patch_grid_shape)

    resolved = False
    if piece_symbol is not None:
        # Primary disambiguator: the piece's known color from the FEN.
        # White pieces are brighter than the board; black pieces darker.
        # Pick the sign whose ``+`` cluster's mean luminance is in the
        # right direction. This is robust to a misaligned geometric mask.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch_luminance = gray.reshape(
            patch_grid_shape[0], patch_size, patch_grid_shape[1], patch_size
        ).mean(axis=(1, 3))
        pos_patches = (grid > 0) & content_mask
        neg_patches = (grid < 0) & content_mask
        if pos_patches.any() and neg_patches.any():
            pos_luminance = float(patch_luminance[pos_patches].mean())
            neg_luminance = float(patch_luminance[neg_patches].mean())
            if piece_symbol.isupper():
                if pos_luminance < neg_luminance:
                    grid = -grid
            else:
                if pos_luminance > neg_luminance:
                    grid = -grid
            resolved = True

    if not resolved and disambiguation_hint is not None:
        if disambiguation_hint.shape != grid.shape:
            raise ValueError(
                f"disambiguation_hint shape {disambiguation_hint.shape} does not match "
                f"patch grid {grid.shape}"
            )
        hint_binary = (disambiguation_hint > 0.3) & content_mask
        if hint_binary.any():
            pos_overlap = int(((grid > 0) & hint_binary).sum())
            neg_overlap = int(((grid < 0) & hint_binary).sum())
            if neg_overlap > pos_overlap:
                grid = -grid
            resolved = True

    if not resolved:
        h, w = grid.shape
        bottom_left = grid[h // 2 :, : max(1, w // 2)]
        if bottom_left.mean() < 0:
            grid = -grid

    mask = (grid > 0).astype(np.float32)
    mask[padding_mask] = 0.0
    return mask


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
        piece_symbol=piece.symbol,
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
