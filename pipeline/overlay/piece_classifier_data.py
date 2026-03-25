"""Synthetic training data generator for the per-square piece classifier.

Renders chess squares with pieces from multiple sources:
- 3D pieces via ``piece_renderer.py`` (8 materials × 6 piece types × 2 colors)
- SVG pieces via ``chess.svg`` (python-chess CBurnett set)

On backgrounds from:
- ``board_themes.py`` (6 themes × texture generators)
- Random solid colours sampled from a broad palette

With augmentations:
- Brightness / contrast jitter
- Gaussian blur
- JPEG compression artefacts
- Random move-highlight tints (yellow, green, blue)
- Resize noise (render at high-res then down-sample)

Classes (13):
  0  = empty
  1-6  = white P, N, B, R, Q, K
  7-12 = black p, n, b, r, q, k
"""

from __future__ import annotations

import io
import logging
import random
from pathlib import Path

import chess
import chess.svg
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Class index mapping — matches PIECE_CLASSES in overlay_reader.py
CLASS_NAMES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


# ---------------------------------------------------------------------------
# SVG piece rendering (flat 2D via python-chess / cairosvg)
# ---------------------------------------------------------------------------


def _render_svg_square(
    piece: chess.Piece | None,
    is_light: bool,
    light_hex: str,
    dark_hex: str,
    size: int,
) -> np.ndarray:
    """Render a single square with an optional piece using python-chess SVG."""
    import cairosvg

    board = chess.Board(fen=None)
    sq_idx = chess.A8 if is_light else chess.B8  # A8 is light, B8 is dark
    if piece is not None:
        board.set_piece_at(sq_idx, piece)

    render_size = max(size * 4, 256)
    svg = chess.svg.board(
        board,
        size=int(render_size * 390 / 360),
        colors={"square light": light_hex, "square dark": dark_hex},
    )
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=render_size,
        output_height=render_size,
    )
    img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))

    # Crop margin and extract the target square
    h, w = img.shape[:2]
    margin = int(round(15 / 390 * w))
    board_area = img[margin : h - margin, margin : w - margin]
    sq_px = board_area.shape[0] // 8
    if is_light:
        square_crop = board_area[:sq_px, :sq_px]
    else:
        square_crop = board_area[:sq_px, sq_px : 2 * sq_px]

    return cv2.resize(cv2.cvtColor(square_crop, cv2.COLOR_RGB2BGR), (size, size))


# ---------------------------------------------------------------------------
# 3D piece rendering
# ---------------------------------------------------------------------------

_3D_AVAILABLE = True
try:
    from argus.datagen.board_themes import BOARD_THEMES as DATAGEN_THEMES
    from argus.datagen.board_themes import generate_square_texture
    from argus.datagen.piece_renderer import (
        PIECE_MATERIALS,
        PieceRenderCache,
        select_random_material,
    )
except ImportError:
    _3D_AVAILABLE = False
    DATAGEN_THEMES = []  # type: ignore[assignment]
    PIECE_MATERIALS = []  # type: ignore[assignment]


def _render_3d_square(
    piece_type: int | None,
    is_white: bool,
    is_light_square: bool,
    size: int,
    rng: random.Random,
    cache: PieceRenderCache | None = None,
) -> np.ndarray:
    """Render a square with a 3D piece on a textured background."""
    theme = random.choice(DATAGEN_THEMES).with_perturbation(rng)
    bg = generate_square_texture(size, theme, is_light_square, rng)

    if piece_type is None:
        return cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    material = select_random_material(rng).with_perturbation(rng)
    light_dir = np.array([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.7, -0.3),
        rng.uniform(0.5, 1.0),
    ])
    light_dir = light_dir / np.linalg.norm(light_dir)

    if cache is not None:
        sprite = cache.get_or_render(piece_type, is_white, material, size, light_dir, rng)
    else:
        from argus.datagen.piece_renderer import render_piece_sprite
        sprite = render_piece_sprite(piece_type, material, is_white, size, light_dir, rng)

    # Composite sprite onto background
    sprite_rgba = np.array(sprite.convert("RGBA"))
    alpha = sprite_rgba[:, :, 3:4].astype(float) / 255.0
    rgb = sprite_rgba[:, :, :3].astype(float)
    bg_f = bg.astype(float)
    composited = (rgb * alpha + bg_f * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply random augmentations to a square image."""
    # Brightness / contrast
    alpha = rng.uniform(0.8, 1.2)  # contrast
    beta = rng.uniform(-20, 20)    # brightness
    img = np.clip(img.astype(float) * alpha + beta, 0, 255).astype(np.uint8)

    # Random highlight tint (simulate move highlights)
    if rng.random() < 0.15:
        tint_color = rng.choice([
            (0, 200, 200),   # yellow-ish
            (0, 200, 0),     # green
            (200, 100, 0),   # blue-ish
            (0, 100, 200),   # orange
        ])
        tint = np.full_like(img, tint_color, dtype=np.uint8)
        alpha_tint = rng.uniform(0.15, 0.35)
        img = cv2.addWeighted(img, 1 - alpha_tint, tint, alpha_tint, 0)

    # Coordinate label overlay (simulate rank/file labels on squares)
    if rng.random() < 0.2:
        h, w = img.shape[:2]
        label = rng.choice(list("abcdefgh12345678"))
        pos = rng.choice([(5, h - 5), (w - 15, h - 5), (5, 15), (w - 15, 15)])
        color = rng.choice([(80, 60, 40), (160, 140, 120), (50, 50, 50)])
        cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Gaussian blur
    if rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # JPEG compression
    if rng.random() < 0.4:
        quality = rng.randint(50, 90)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------------------------
# Overlay-style themes (flat solid squares like lichess/chess.com)
# ---------------------------------------------------------------------------

OVERLAY_THEMES = [
    ("#F0D9B5", "#B58863"),  # lichess default
    ("#EEEED2", "#769656"),  # chess.com green
    ("#DEE3E6", "#8CA2AD"),  # chess.com blue
    ("#FFFFFF", "#DD3333"),  # red/white (like Ov8)
    ("#E8E8E8", "#555555"),  # grey
    ("#F5F5DC", "#8B4513"),  # beige/brown
    ("#FFFACD", "#2E8B57"),  # light yellow/green
    ("#E6D5AC", "#A0522D"),  # tan/sienna
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dataset(
    output_dir: str | Path,
    num_samples_per_class: int = 500,
    size: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced dataset of piece square crops.

    Returns (images, labels) where:
    - images: (N, size, size, 3) uint8 BGR
    - labels: (N,) int in [0..12]
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    images: list[np.ndarray] = []
    labels: list[int] = []

    use_3d = _3D_AVAILABLE and len(PIECE_MATERIALS) > 0
    cache = PieceRenderCache() if use_3d else None

    for class_idx in range(13):
        for i in range(num_samples_per_class):
            is_light = rng.random() < 0.5
            use_svg = not use_3d or rng.random() < 0.7  # bias toward SVG (overlay-style)

            if class_idx == 0:
                # Empty square
                piece = None
                piece_type = None
                is_white = True  # irrelevant
            else:
                is_white = class_idx <= 6
                pt_idx = (class_idx - 1) % 6
                piece_type = PIECE_TYPES[pt_idx]
                piece = chess.Piece(piece_type, chess.WHITE if is_white else chess.BLACK)

            if use_svg:
                theme = rng.choice(OVERLAY_THEMES)
                light_hex, dark_hex = theme
                img = _render_svg_square(piece, is_light, light_hex, dark_hex, size)
            else:
                img = _render_3d_square(piece_type, is_white, is_light, size, rng, cache)

            img = _augment(img, rng)
            images.append(img)
            labels.append(class_idx)

    images_arr = np.array(images, dtype=np.uint8)
    labels_arr = np.array(labels, dtype=np.int64)

    # Shuffle
    perm = np_rng.permutation(len(images_arr))
    return images_arr[perm], labels_arr[perm]
