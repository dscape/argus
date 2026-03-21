"""Realistic chess board themes and procedural texture generators.

Real tournament boards are green/white vinyl mats, wood with grain, or
DGT electronic boards — not the flat solid colors of online chess sites.
This module provides a registry of real-world board appearances and
procedural texture generators that produce per-square pixel variance
matching real boards (variance > 25, unlike rendered boards' ~0-5).
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class BoardTheme:
    """A realistic chess board appearance."""

    name: str
    light: str  # hex color for light squares
    dark: str  # hex color for dark squares
    texture_type: str  # "vinyl", "wood", or "plastic"
    square_noise_range: tuple[float, float] = (3.0, 8.0)
    has_coordinates: bool = False
    border_color: str | None = None
    weight: float = 1.0

    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    @property
    def light_rgb(self) -> tuple[int, int, int]:
        return self.hex_to_rgb(self.light)

    @property
    def dark_rgb(self) -> tuple[int, int, int]:
        return self.hex_to_rgb(self.dark)

    def with_perturbation(self, rng: random.Random) -> BoardTheme:
        """Return a copy with small random color perturbation."""
        light = _perturb_color(self.light_rgb, rng)
        dark = _perturb_color(self.dark_rgb, rng)
        return BoardTheme(
            name=self.name,
            light=f"#{light[0]:02x}{light[1]:02x}{light[2]:02x}",
            dark=f"#{dark[0]:02x}{dark[1]:02x}{dark[2]:02x}",
            texture_type=self.texture_type,
            square_noise_range=self.square_noise_range,
            has_coordinates=self.has_coordinates,
            border_color=self.border_color,
            weight=self.weight,
        )


def _perturb_color(
    rgb: tuple[int, int, int], rng: random.Random
) -> tuple[int, int, int]:
    """Apply small hue/saturation/brightness perturbation to a color."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + rng.uniform(-0.02, 0.02)) % 1.0
    s = max(0.0, min(1.0, s + rng.uniform(-0.08, 0.08)))
    v = max(0.0, min(1.0, v + rng.uniform(-0.06, 0.06)))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


# ---------------------------------------------------------------------------
# Theme registry — real-world board appearances
# ---------------------------------------------------------------------------

BOARD_THEMES: list[BoardTheme] = [
    BoardTheme(
        name="green_white_vinyl",
        light="#FFFEF0",
        dark="#4A7C59",
        texture_type="vinyl",
        square_noise_range=(3.0, 8.0),
        has_coordinates=True,
        weight=0.40,
    ),
    BoardTheme(
        name="green_buff_vinyl",
        light="#E8DCC8",
        dark="#6B8E4E",
        texture_type="vinyl",
        square_noise_range=(3.0, 8.0),
        has_coordinates=True,
        weight=0.15,
    ),
    BoardTheme(
        name="blue_white_vinyl",
        light="#ECF0F1",
        dark="#2E6B9E",
        texture_type="vinyl",
        square_noise_range=(3.0, 8.0),
        has_coordinates=False,
        weight=0.10,
    ),
    BoardTheme(
        name="mahogany_maple",
        light="#F0DEB4",
        dark="#8B5E3C",
        texture_type="wood",
        square_noise_range=(10.0, 25.0),
        has_coordinates=False,
        border_color="#5C3317",
        weight=0.15,
    ),
    BoardTheme(
        name="walnut_maple",
        light="#F5E6C8",
        dark="#6B4226",
        texture_type="wood",
        square_noise_range=(10.0, 25.0),
        has_coordinates=False,
        border_color="#4A2E14",
        weight=0.10,
    ),
    BoardTheme(
        name="dgt_electronic",
        light="#E8E8E8",
        dark="#333333",
        texture_type="plastic",
        square_noise_range=(5.0, 12.0),
        has_coordinates=True,
        weight=0.10,
    ),
]


def select_random_theme(rng: random.Random) -> BoardTheme:
    """Select a board theme using weighted random sampling."""
    weights = [t.weight for t in BOARD_THEMES]
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for theme in BOARD_THEMES:
        cumulative += theme.weight
        if r <= cumulative:
            return theme
    return BOARD_THEMES[-1]


# ---------------------------------------------------------------------------
# Procedural texture generators
# ---------------------------------------------------------------------------


def generate_wood_grain_texture(
    size: int,
    base_color: tuple[int, int, int],
    grain_intensity: float,
    rng: random.Random,
) -> np.ndarray:
    """Generate a square texture with wood grain pattern.

    Uses directional gaussian blur on noise to create parallel grain lines,
    plus subtle swirl patterns for knots.

    Returns:
        (size, size, 3) uint8 numpy array.
    """
    rs = np.random.RandomState(rng.randint(0, 2**31))

    # Base color array
    img = np.full((size, size, 3), base_color, dtype=np.float32)

    # Layer 1: directional grain lines
    noise = rs.normal(0, 1, (size, size))
    grain_angle = rng.uniform(0, np.pi)  # random grain direction
    # Apply anisotropic gaussian blur — high sigma along grain, low across
    sigma_along = size * 0.4
    sigma_across = size * 0.02
    cos_a, sin_a = np.cos(grain_angle), np.sin(grain_angle)

    # Rotate, blur directionally, rotate back
    # Simpler approach: blur with different sigmas on each axis
    grain = gaussian_filter(noise, sigma=(sigma_across, sigma_along))
    grain = grain / (np.abs(grain).max() + 1e-8)  # normalize to [-1, 1]

    # Layer 2: subtle larger-scale variation (simulates color variation in wood)
    broad_noise = rs.normal(0, 1, (size, size))
    broad = gaussian_filter(broad_noise, sigma=size * 0.15)
    broad = broad / (np.abs(broad).max() + 1e-8)

    # Combine
    intensity = grain_intensity * 20  # scale to pixel values
    for c in range(3):
        img[:, :, c] += grain * intensity * 0.7
        img[:, :, c] += broad * intensity * 0.3

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_vinyl_texture(
    size: int,
    base_color: tuple[int, int, int],
    noise_range: tuple[float, float],
    rng: random.Random,
) -> np.ndarray:
    """Generate a square texture simulating vinyl mat surface.

    Vinyl mats have slight mottling and micro-texture but are much
    smoother than wood.

    Returns:
        (size, size, 3) uint8 numpy array.
    """
    rs = np.random.RandomState(rng.randint(0, 2**31))

    img = np.full((size, size, 3), base_color, dtype=np.float32)

    # Subtle gaussian mottling
    noise = rs.normal(0, 1, (size, size))
    mottled = gaussian_filter(noise, sigma=rng.uniform(2.0, 5.0))
    mottled = mottled / (np.abs(mottled).max() + 1e-8)

    noise_std = rng.uniform(*noise_range)
    for c in range(3):
        img[:, :, c] += mottled * noise_std

    # Fine speckle noise
    speckle = rs.normal(0, noise_std * 0.3, (size, size, 3))
    img += speckle

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_plastic_texture(
    size: int,
    base_color: tuple[int, int, int],
    noise_range: tuple[float, float],
    rng: random.Random,
) -> np.ndarray:
    """Generate a square texture simulating plastic/electronic board surface.

    Smoother than vinyl with slight glossy variation.

    Returns:
        (size, size, 3) uint8 numpy array.
    """
    rs = np.random.RandomState(rng.randint(0, 2**31))

    img = np.full((size, size, 3), base_color, dtype=np.float32)

    # Very subtle smooth variation (simulates slight gloss differences)
    noise = rs.normal(0, 1, (size, size))
    smooth = gaussian_filter(noise, sigma=size * 0.2)
    smooth = smooth / (np.abs(smooth).max() + 1e-8)

    noise_std = rng.uniform(*noise_range)
    for c in range(3):
        img[:, :, c] += smooth * noise_std * 0.5

    # Minimal speckle
    speckle = rs.normal(0, noise_std * 0.15, (size, size, 3))
    img += speckle

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_square_texture(
    size: int,
    theme: BoardTheme,
    is_light: bool,
    rng: random.Random,
) -> np.ndarray:
    """Generate a textured square image for the given theme and square color.

    Dispatches to the appropriate texture generator based on theme type.

    Returns:
        (size, size, 3) uint8 numpy array.
    """
    base_color = theme.light_rgb if is_light else theme.dark_rgb
    noise_range = theme.square_noise_range

    if theme.texture_type == "wood":
        grain_intensity = rng.uniform(0.5, 1.5)
        return generate_wood_grain_texture(size, base_color, grain_intensity, rng)
    elif theme.texture_type == "vinyl":
        return generate_vinyl_texture(size, base_color, noise_range, rng)
    else:  # plastic
        return generate_plastic_texture(size, base_color, noise_range, rng)


def render_textured_board(
    size: int,
    theme: BoardTheme,
    flipped: bool,
    rng: random.Random,
) -> np.ndarray:
    """Render a full 8x8 textured board without pieces.

    Returns:
        (size, size, 3) uint8 numpy array.
    """
    sq_size = size // 8
    board_img = np.zeros((sq_size * 8, sq_size * 8, 3), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            square = generate_square_texture(sq_size, theme, is_light, rng)
            y0 = row * sq_size
            x0 = col * sq_size
            board_img[y0 : y0 + sq_size, x0 : x0 + sq_size] = square

    # Resize to exact target if needed (handles non-multiple-of-8 sizes)
    if board_img.shape[0] != size or board_img.shape[1] != size:
        import cv2

        board_img = cv2.resize(board_img, (size, size))

    return board_img
