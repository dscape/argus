"""Realistic chess piece styles with 3D material effects.

Real tournament pieces are 3D objects made of plastic (most common), wood,
or metal (rare). They cast shadows, have bevel edges, material-specific
surface textures, and realistic colors (ivory/cream for white, dark
brown/charcoal for black — never pure white/black).

This module provides a registry of piece styles and a post-processing
pipeline that transforms flat SVG piece layers into 3D-looking pieces
using only PIL and numpy operations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

from argus.datagen.board_themes import _perturb_color


@dataclass
class PieceStyle:
    """A realistic chess piece material and appearance."""

    name: str
    material: str  # "plastic", "wood", or "metal"

    # Realistic base colors (not pure white/black)
    white_base: tuple[int, int, int]
    black_base: tuple[int, int, int]

    # Shadow parameters
    shadow_offset: tuple[int, int]  # (dx, dy) scaled by square size
    shadow_blur: float
    shadow_opacity: float  # 0.0–1.0

    # Bevel / emboss
    bevel_strength: float
    bevel_blur: float

    # Lighting
    gradient_strength: float  # top-to-bottom brightness gradient
    specular_intensity: float  # brightness of specular highlight
    specular_size: float  # Gaussian sigma as fraction of square size

    # Surface texture
    surface_noise: float  # per-pixel noise std
    grain_intensity: float  # wood grain overlay (only for wood material)

    weight: float = 1.0

    def with_perturbation(self, rng: random.Random) -> PieceStyle:
        """Return a copy with small random color and parameter variation."""
        wb = _perturb_color(self.white_base, rng)
        bb = _perturb_color(self.black_base, rng)
        jitter = lambda v: v * rng.uniform(0.85, 1.15)  # noqa: E731
        return PieceStyle(
            name=self.name,
            material=self.material,
            white_base=wb,
            black_base=bb,
            shadow_offset=self.shadow_offset,
            shadow_blur=max(0.5, jitter(self.shadow_blur)),
            shadow_opacity=min(1.0, max(0.1, jitter(self.shadow_opacity))),
            bevel_strength=jitter(self.bevel_strength),
            bevel_blur=max(0.3, jitter(self.bevel_blur)),
            gradient_strength=jitter(self.gradient_strength),
            specular_intensity=jitter(self.specular_intensity),
            specular_size=jitter(self.specular_size),
            surface_noise=jitter(self.surface_noise),
            grain_intensity=jitter(self.grain_intensity),
            weight=self.weight,
        )


# ---------------------------------------------------------------------------
# Piece style registry
# ---------------------------------------------------------------------------

PIECE_STYLES: list[PieceStyle] = [
    PieceStyle(
        name="tournament_plastic",
        material="plastic",
        white_base=(235, 225, 210),
        black_base=(55, 45, 35),
        shadow_offset=(2, 3),
        shadow_blur=2.5,
        shadow_opacity=0.35,
        bevel_strength=0.5,
        bevel_blur=1.5,
        gradient_strength=0.10,
        specular_intensity=25.0,
        specular_size=0.18,
        surface_noise=3.0,
        grain_intensity=0.0,
        weight=0.30,
    ),
    PieceStyle(
        name="club_plastic",
        material="plastic",
        white_base=(245, 240, 230),
        black_base=(40, 35, 30),
        shadow_offset=(2, 3),
        shadow_blur=2.0,
        shadow_opacity=0.30,
        bevel_strength=0.55,
        bevel_blur=1.3,
        gradient_strength=0.12,
        specular_intensity=30.0,
        specular_size=0.15,
        surface_noise=2.5,
        grain_intensity=0.0,
        weight=0.15,
    ),
    PieceStyle(
        name="basic_plastic",
        material="plastic",
        white_base=(250, 250, 245),
        black_base=(35, 35, 35),
        shadow_offset=(2, 2),
        shadow_blur=2.0,
        shadow_opacity=0.30,
        bevel_strength=0.45,
        bevel_blur=1.5,
        gradient_strength=0.08,
        specular_intensity=20.0,
        specular_size=0.20,
        surface_noise=2.0,
        grain_intensity=0.0,
        weight=0.15,
    ),
    PieceStyle(
        name="weighted_plastic",
        material="plastic",
        white_base=(230, 215, 195),
        black_base=(60, 50, 40),
        shadow_offset=(2, 3),
        shadow_blur=3.0,
        shadow_opacity=0.40,
        bevel_strength=0.50,
        bevel_blur=1.8,
        gradient_strength=0.10,
        specular_intensity=22.0,
        specular_size=0.18,
        surface_noise=3.5,
        grain_intensity=0.0,
        weight=0.10,
    ),
    PieceStyle(
        name="boxwood_ebony",
        material="wood",
        white_base=(220, 195, 150),
        black_base=(70, 45, 25),
        shadow_offset=(2, 3),
        shadow_blur=2.5,
        shadow_opacity=0.35,
        bevel_strength=0.35,
        bevel_blur=2.0,
        gradient_strength=0.08,
        specular_intensity=12.0,
        specular_size=0.25,
        surface_noise=4.0,
        grain_intensity=0.6,
        weight=0.12,
    ),
    PieceStyle(
        name="sheesham_boxwood",
        material="wood",
        white_base=(225, 200, 160),
        black_base=(100, 60, 30),
        shadow_offset=(2, 3),
        shadow_blur=2.5,
        shadow_opacity=0.35,
        bevel_strength=0.30,
        bevel_blur=2.0,
        gradient_strength=0.08,
        specular_intensity=10.0,
        specular_size=0.25,
        surface_noise=4.0,
        grain_intensity=0.5,
        weight=0.08,
    ),
    PieceStyle(
        name="brushed_steel",
        material="metal",
        white_base=(190, 190, 195),
        black_base=(60, 60, 65),
        shadow_offset=(2, 3),
        shadow_blur=2.0,
        shadow_opacity=0.40,
        bevel_strength=0.70,
        bevel_blur=1.0,
        gradient_strength=0.12,
        specular_intensity=45.0,
        specular_size=0.12,
        surface_noise=2.0,
        grain_intensity=0.0,
        weight=0.05,
    ),
    PieceStyle(
        name="brass_pewter",
        material="metal",
        white_base=(200, 185, 140),
        black_base=(75, 70, 65),
        shadow_offset=(2, 3),
        shadow_blur=2.0,
        shadow_opacity=0.40,
        bevel_strength=0.65,
        bevel_blur=1.0,
        gradient_strength=0.12,
        specular_intensity=40.0,
        specular_size=0.12,
        surface_noise=2.5,
        grain_intensity=0.0,
        weight=0.05,
    ),
]


def select_random_piece_style(rng: random.Random) -> PieceStyle:
    """Select a piece style using weighted random sampling."""
    weights = [s.weight for s in PIECE_STYLES]
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for style in PIECE_STYLES:
        cumulative += style.weight
        if r <= cumulative:
            return style
    return PIECE_STYLES[-1]


# ---------------------------------------------------------------------------
# 3D effect pipeline
# ---------------------------------------------------------------------------


def _remap_piece_colors(
    piece_arr: np.ndarray,
    style: PieceStyle,
) -> np.ndarray:
    """Remap flat white/black piece colors to realistic material colors.

    Uses luminance to classify pixels as belonging to white or black pieces,
    then interpolates toward the material base color while preserving the
    SVG's internal shading and outline structure.
    """
    result = piece_arr.copy().astype(np.float32)
    alpha = result[:, :, 3]
    opaque = alpha > 0

    if not np.any(opaque):
        return piece_arr

    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # White piece pixels: high luminance body, with dark outlines
    white_mask = (lum > 128) & opaque
    # Black piece pixels: low luminance body, with light outlines
    black_mask = (lum <= 128) & opaque

    wb = np.array(style.white_base, dtype=np.float32)
    bb = np.array(style.black_base, dtype=np.float32)

    # For white pieces: t=1 at full white (body), t=0 at black (outline)
    # Map to: body -> white_base, outline -> darkened white_base
    if np.any(white_mask):
        t = lum[white_mask] / 255.0  # 1.0 = body, 0.0 = outline
        dark_wb = wb * 0.35  # darkened version for outlines
        for c in range(3):
            result[:, :, c][white_mask] = (
                dark_wb[c] + (wb[c] - dark_wb[c]) * t
            )

    # For black pieces: t=0 at black (body), t=1 at white (outline)
    # Map to: body -> black_base, outline -> lightened black_base
    if np.any(black_mask):
        t = lum[black_mask] / 255.0  # 0.0 = body, 1.0 = outline
        light_bb = bb + (255.0 - bb) * 0.35  # lightened version for outlines
        for c in range(3):
            result[:, :, c][black_mask] = (
                bb[c] + (light_bb[c] - bb[c]) * t
            )

    return np.clip(result, 0, 255).astype(np.uint8)


def _create_shadow_layer(
    alpha: np.ndarray,
    style: PieceStyle,
    sq_size: int,
) -> Image.Image:
    """Create a drop shadow RGBA layer from the piece alpha mask.

    The shadow is offset and blurred to simulate pieces casting shadows
    on the board from overhead lighting.
    """
    h, w = alpha.shape

    # Scale shadow offset by square size (relative to a reference of 28px)
    scale = sq_size / 28.0
    dx = int(round(style.shadow_offset[0] * scale))
    dy = int(round(style.shadow_offset[1] * scale))

    # Create shadow alpha from piece alpha
    shadow_alpha = (alpha > 128).astype(np.float32) * style.shadow_opacity * 255.0

    # Offset
    shifted = np.zeros_like(shadow_alpha)
    # Clamp ranges for the shift
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = shadow_alpha[src_y0:src_y1, src_x0:src_x1]

    # Blur
    blur_sigma = style.shadow_blur * scale
    shifted = gaussian_filter(shifted, sigma=blur_sigma)

    # Build RGBA shadow layer (black with variable alpha)
    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    shadow[:, :, 3] = np.clip(shifted, 0, 255).astype(np.uint8)

    return Image.fromarray(shadow, "RGBA")


def _apply_bevel(
    piece_arr: np.ndarray,
    style: PieceStyle,
    sq_size: int,
) -> np.ndarray:
    """Apply bevel/emboss effect to make pieces look raised and 3D.

    Uses edge detection on the alpha channel with directional lighting
    (light from top-left) to create highlight and shadow edges.
    """
    result = piece_arr.copy().astype(np.float32)
    alpha = result[:, :, 3]
    opaque = alpha > 0

    if not np.any(opaque):
        return piece_arr

    alpha_f = alpha.astype(np.float32) / 255.0

    # Edge detection via shifted differences (Sobel-like)
    edge_x = np.roll(alpha_f, -1, axis=1) - np.roll(alpha_f, 1, axis=1)
    edge_y = np.roll(alpha_f, -1, axis=0) - np.roll(alpha_f, 1, axis=0)

    # Light from top-left: highlight = -edge_x + -edge_y (top-left gets bright)
    bevel = (-edge_x - edge_y) * style.bevel_strength * 200.0

    # Smooth the bevel
    scale = sq_size / 28.0
    bevel = gaussian_filter(bevel, sigma=style.bevel_blur * scale)

    # Apply only within opaque pixels
    for c in range(3):
        channel = result[:, :, c]
        channel[opaque] = channel[opaque] + bevel[opaque]

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_gradient(
    piece_arr: np.ndarray,
    style: PieceStyle,
) -> np.ndarray:
    """Apply vertical brightness gradient simulating overhead lighting.

    Pieces are brighter on top and slightly darker on bottom.
    """
    result = piece_arr.copy().astype(np.float32)
    alpha = result[:, :, 3]
    opaque = alpha > 0

    if not np.any(opaque):
        return piece_arr

    h = result.shape[0]
    gs = style.gradient_strength
    grad = np.linspace(1.0 + gs, 1.0 - gs, h).reshape(-1, 1)

    for c in range(3):
        channel = result[:, :, c]
        channel[opaque] = channel[opaque] * grad[np.where(opaque)[0]].ravel()

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_specular(
    piece_arr: np.ndarray,
    style: PieceStyle,
    sq_size: int,
) -> np.ndarray:
    """Add per-square specular highlight to simulate light reflection.

    Places a Gaussian bright spot slightly above center of each occupied
    square to simulate overhead lighting on glossy/semi-glossy surfaces.
    """
    if style.specular_intensity < 1.0:
        return piece_arr

    result = piece_arr.copy().astype(np.float32)
    alpha = result[:, :, 3]
    h, w = alpha.shape

    sigma = style.specular_size * sq_size

    # Process each square on the 8x8 grid
    for row in range(8):
        for col in range(8):
            y0 = row * sq_size
            x0 = col * sq_size
            y1 = min(y0 + sq_size, h)
            x1 = min(x0 + sq_size, w)

            sq_alpha = alpha[y0:y1, x0:x1]
            if not np.any(sq_alpha > 0):
                continue

            # Specular center: slightly above center of square
            cy = sq_size * 0.35
            cx = sq_size * 0.5
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            spec = style.specular_intensity * np.exp(-dist_sq / (2 * sigma ** 2))

            sq_mask = sq_alpha[: y1 - y0, : x1 - x0] > 0
            for c in range(3):
                block = result[y0:y1, x0:x1, c]
                block[sq_mask] = block[sq_mask] + spec[sq_mask]

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_surface_texture(
    piece_arr: np.ndarray,
    style: PieceStyle,
    rng: random.Random,
) -> np.ndarray:
    """Apply material-specific surface texture to piece pixels.

    Plastic: subtle per-pixel noise.
    Wood: fine grain pattern overlaid on piece surfaces.
    Metal: directional brushing streaks.
    """
    result = piece_arr.copy().astype(np.float32)
    alpha = result[:, :, 3]
    opaque = alpha > 0
    h, w = alpha.shape

    if not np.any(opaque) or style.surface_noise < 0.5:
        return piece_arr

    rs = np.random.RandomState(rng.randint(0, 2**31))

    if style.material == "plastic":
        noise = rs.normal(0, style.surface_noise, (h, w))
        for c in range(3):
            result[:, :, c][opaque] += noise[opaque]

    elif style.material == "wood":
        # Fine grain pattern
        noise = rs.normal(0, 1, (h, w))
        grain = gaussian_filter(noise, sigma=(0.5, h * 0.15))
        grain = grain / (np.abs(grain).max() + 1e-8)
        grain *= style.grain_intensity * 15.0

        # Also add fine surface noise
        fine = rs.normal(0, style.surface_noise, (h, w))

        for c in range(3):
            result[:, :, c][opaque] += grain[opaque] + fine[opaque]

    elif style.material == "metal":
        # Directional brushing (horizontal streaks)
        noise = rs.normal(0, 1, (h, w))
        brushed = gaussian_filter(noise, sigma=(0.3, w * 0.1))
        brushed = brushed / (np.abs(brushed).max() + 1e-8)
        brushed *= style.surface_noise * 2.0

        for c in range(3):
            result[:, :, c][opaque] += brushed[opaque]

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_piece_style(
    piece_layer: Image.Image,
    style: PieceStyle,
    board_size: int,
    rng: random.Random,
    skip_3d_effects: bool = False,
) -> tuple[Image.Image, Image.Image]:
    """Transform piece layer with material effects.

    When skip_3d_effects is False (legacy SVG path): applies all 6 stages
    including color remapping, bevel, gradient, and specular.

    When skip_3d_effects is True (3D renderer path): only generates the
    shadow layer. The 3D renderer already handles shading, so bevel,
    gradient, and specular are redundant.

    Args:
        piece_layer: RGBA piece layer.
        style: PieceStyle defining shadow and texture params.
        board_size: Board image size in pixels.
        rng: Random number generator.
        skip_3d_effects: If True, only generate shadow (3D renderer
            handles shading intrinsically).

    Returns:
        Tuple of (styled_piece_layer, shadow_layer) — both RGBA Images.
        Shadow layer should be composited *under* pieces on the board.
    """
    piece_arr = np.array(piece_layer)
    sq_size = board_size // 8

    # Shadow is always generated (position-dependent, adds realism)
    shadow = _create_shadow_layer(piece_arr[:, :, 3], style, sq_size)

    if skip_3d_effects:
        # 3D renderer handles color, bevel, gradient, specular already
        return piece_layer, shadow

    # Full 2D post-processing pipeline (legacy SVG path)
    piece_arr = _remap_piece_colors(piece_arr, style)
    piece_arr = _apply_bevel(piece_arr, style, sq_size)
    piece_arr = _apply_gradient(piece_arr, style)
    piece_arr = _apply_specular(piece_arr, style, sq_size)
    piece_arr = _apply_surface_texture(piece_arr, style, rng)

    return Image.fromarray(piece_arr, "RGBA"), shadow
