"""Full 2D synthetic data generator for Argus training.

Renders chess board images with realistic textures (wood grain, vinyl mat)
matching real tournament boards, applies perspective transforms and
augmentations, and generates temporal clips with ground truth move
annotations.
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Any, Callable

import chess
import chess.svg
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.data.pgn_sampler import sample_random_game
from argus.datagen.board_themes import (
    BoardTheme,
    render_textured_board,
    select_random_theme,
)
from argus.datagen.piece_renderer import (
    PieceMaterial,
    PieceRenderCache,
    render_pieces_layer,
    select_random_material,
)
from argus.datagen.piece_styles import (
    PieceStyle,
    apply_piece_style,
    select_random_piece_style,
)


def _render_svg_pieces(
    board: chess.Board,
    size: int,
    flipped: bool,
) -> Image.Image | None:
    """Render only the pieces (no board squares) from chess.svg as RGBA.

    Renders the board with pieces on a known background, then renders an
    empty board with the same background, and diffs them to extract just
    the piece pixels as an alpha-compositable layer.

    Returns RGBA PIL Image, or None if cairosvg is unavailable.
    """
    try:
        import cairosvg
    except ImportError:
        return None

    # Use a distinctive key color for the board squares so we can mask them
    key = "#FF00FF"
    svg_with = chess.svg.board(
        board,
        size=size,
        flipped=flipped,
        colors={"square light": key, "square dark": key},
    )
    svg_empty = chess.svg.board(
        chess.Board(fen=None),
        size=size,
        flipped=flipped,
        colors={"square light": key, "square dark": key},
    )

    png_with = cairosvg.svg2png(
        bytestring=svg_with.encode("utf-8"), output_width=size, output_height=size
    )
    png_empty = cairosvg.svg2png(
        bytestring=svg_empty.encode("utf-8"), output_width=size, output_height=size
    )

    img_with = np.array(Image.open(io.BytesIO(png_with)).convert("RGBA"))
    img_empty = np.array(Image.open(io.BytesIO(png_empty)).convert("RGBA"))

    # Pixels that differ between the two renders are piece pixels
    diff = np.any(img_with[:, :, :3] != img_empty[:, :, :3], axis=2)

    # Create RGBA layer: piece pixels are opaque, background is transparent
    result = img_with.copy()
    result[~diff, 3] = 0  # make non-piece pixels transparent

    # Remove magenta color spill on anti-aliased piece edges.
    # Edge pixels are blended with the FF00FF key color by the SVG
    # rasterizer. Detect them by high magenta component and reduce alpha.
    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
    # Magenta = high R, low G, high B
    magenta_score = (
        r.astype(np.float32) - g.astype(np.float32)
        + b.astype(np.float32) - g.astype(np.float32)
    ) / 510.0  # normalize to [0, 1]
    magenta_mask = (magenta_score > 0.3) & diff
    # Fade out magenta-contaminated edge pixels
    result[magenta_mask, 3] = np.clip(
        result[magenta_mask, 3].astype(np.float32)
        * (1.0 - magenta_score[magenta_mask]),
        0, 255,
    ).astype(np.uint8)

    return Image.fromarray(result, "RGBA")


def _crop_svg_margin(img: Image.Image) -> Image.Image:
    """Crop the margin added by chess.svg.board().

    chess.svg uses a viewBox of 390 = 15 (margin) + 8*45 (squares) + 15.
    """
    w, h = img.size
    margin_x = int(round(15 / 390 * w))
    margin_y = int(round(15 / 390 * h))
    return img.crop((margin_x, margin_y, w - margin_x, h - margin_y))


def render_board_image(
    board: chess.Board,
    size: int = 224,
    flipped: bool = False,
    colors: dict[str, str] | None = None,
    theme: BoardTheme | None = None,
    piece_style: PieceStyle | None = None,
    piece_material: PieceMaterial | None = None,
    piece_cache: PieceRenderCache | None = None,
    rng: random.Random | None = None,
) -> Image.Image:
    """Render a chess board position to a PIL Image with realistic textures.

    Args:
        board: Chess board position.
        size: Output image size (square).
        flipped: Whether to flip the board (black perspective).
        colors: Legacy dict with 'light' and 'dark' hex colors.
            If provided without a theme, creates a flat-fill board.
        theme: BoardTheme for realistic textured rendering.
            Takes precedence over colors.
        piece_style: PieceStyle for 2D piece post-processing (legacy).
        piece_material: PieceMaterial for 3D piece rendering.
            Takes precedence over piece_style.
        piece_cache: PieceRenderCache for caching rendered sprites.
        rng: Random number generator for texture variation.

    Returns:
        PIL Image of the board.
    """
    if rng is None:
        rng = random.Random()

    # If theme is provided, use textured rendering
    if theme is not None:
        return _render_textured_with_pieces(
            board, size, flipped, theme, rng,
            piece_style=piece_style,
            piece_material=piece_material,
            piece_cache=piece_cache,
        )

    # Legacy path: flat colors (backward compat)
    default_colors = {"light": "#F0D9B5", "dark": "#B58863"}
    c = colors or default_colors

    svg_text = chess.svg.board(
        board,
        size=size,
        flipped=flipped,
        colors={"square light": c["light"], "square dark": c["dark"]},
    )
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"), output_width=size, output_height=size
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGB")
    except ImportError:
        img = _render_textured_with_pieces(
            board, size, flipped,
            BoardTheme(name="fallback", light=c["light"], dark=c["dark"],
                       texture_type="vinyl"),
            rng,
        )
    return img


def _render_textured_with_pieces(
    board: chess.Board,
    size: int,
    flipped: bool,
    theme: BoardTheme,
    rng: random.Random,
    piece_style: PieceStyle | None = None,
    piece_material: PieceMaterial | None = None,
    piece_cache: PieceRenderCache | None = None,
) -> Image.Image:
    """Render textured board and composite pieces on top."""
    # Render textured board (no pieces)
    board_arr = render_textured_board(size, theme, flipped, rng)
    board_img = Image.fromarray(board_arr, "RGB")

    if piece_material is not None:
        # 3D rendered pieces from geometry
        piece_layer = render_pieces_layer(
            board, size, flipped, piece_material,
            cache=piece_cache, rng=rng,
        )
        # Generate shadow from piece alpha
        if piece_style is not None:
            _, shadow_layer = apply_piece_style(
                piece_layer, piece_style, size, rng,
                skip_3d_effects=True,
            )
            board_img.paste(shadow_layer, (0, 0), shadow_layer)
        board_img.paste(piece_layer, (0, 0), piece_layer)
    else:
        # Legacy SVG path
        render_size = int(size * 390 / 360)
        piece_layer = _render_svg_pieces(board, render_size, flipped)

        if piece_layer is not None:
            piece_layer = _crop_svg_margin(piece_layer)
            piece_layer = piece_layer.resize((size, size), Image.LANCZOS)

            if piece_style is not None:
                piece_layer, shadow_layer = apply_piece_style(
                    piece_layer, piece_style, size, rng,
                )
                board_img.paste(shadow_layer, (0, 0), shadow_layer)

            board_img.paste(piece_layer, (0, 0), piece_layer)
        else:
            _draw_piece_symbols(board_img, board, size, flipped)

    return board_img


def _draw_piece_symbols(
    img: Image.Image,
    board: chess.Board,
    size: int,
    flipped: bool,
) -> None:
    """Draw piece letter symbols as fallback when SVG rendering unavailable."""
    draw = ImageDraw.Draw(img)
    sq_size = size // 8

    piece_symbols = {
        chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
        chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K",
    }

    for rank in range(8):
        for file in range(8):
            display_rank = 7 - rank if not flipped else rank
            display_file = file if not flipped else 7 - file
            sq = chess.square(display_file, display_rank)

            piece = board.piece_at(sq)
            if piece is not None:
                symbol = piece_symbols.get(piece.piece_type, "?")
                if piece.color == chess.BLACK:
                    symbol = symbol.lower()
                text_color = "white" if piece.color == chess.BLACK else "black"
                x0 = file * sq_size
                y0 = rank * sq_size
                cx = x0 + sq_size // 2
                cy = y0 + sq_size // 2
                draw.text((cx - 4, cy - 6), symbol, fill=text_color)


# -----------------------------------------------------------------------
# Camera projection (isometric + perspective)
# -----------------------------------------------------------------------

# Table surface colors (background behind projected board)
TABLE_COLORS = [
    (139, 90, 43),    # brown wood
    (160, 140, 120),  # light wood
    (180, 180, 175),  # grey laminate
    (200, 190, 170),  # beige
    (120, 100, 80),   # dark wood
]


def apply_projection(
    img: Image.Image,
    elevation_deg: float,
    azimuth_deg: float,
    rng: random.Random,
    mode: str = "isometric",
) -> Image.Image:
    """Apply camera projection to a top-down board image.

    Supports isometric (parallel) and perspective projection.
    Isometric is more realistic for tournament overhead cameras
    and simpler for future 3D piece sprite compositing.

    Args:
        img: Square board image (top-down view).
        elevation_deg: Camera elevation above table plane.
            90 = directly overhead, 30 = steep angle.
        azimuth_deg: Camera rotation around the board (0-360).
        rng: Random number generator.
        mode: "isometric" for parallel projection (no depth-based
            scaling — far edge same width as near edge) or
            "perspective" for vanishing-point projection.

    Returns:
        Projected PIL Image with table-colored background.
    """
    w, h = img.size
    arr = np.array(img)

    elev_rad = math.radians(max(elevation_deg, 10.0))
    azim_rad = math.radians(azimuth_deg)

    # Vertical compression from viewing angle
    # At 90 deg (overhead): scale_y = 1.0 (no compression)
    # At 30 deg: scale_y = sin(30) = 0.5 (board appears half as tall)
    scale_y = math.sin(elev_rad)

    cos_a = math.cos(azim_rad)
    sin_a = math.sin(azim_rad)
    cx, cy = w / 2.0, h / 2.0

    # Margin for output canvas
    margin = int(w * 0.2)
    out_w = w + 2 * margin
    out_h = h + 2 * margin

    # Source corners: TL, TR, BR, BL
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    dst_pts = []
    for sx, sy in src:
        dx, dy = sx - cx, sy - cy

        # Rotate into camera-aligned frame
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a

        if mode == "isometric":
            # Isometric: compress depth axis uniformly (no convergence)
            ry *= scale_y
        else:
            # Perspective: far edge shrinks, near edge stays
            depth_t = (ry / (h / 2.0) + 1.0) / 2.0  # 0 = near, 1 = far
            depth_scale = 1.0 - (1.0 - scale_y) * depth_t
            rx *= depth_scale
            ry *= scale_y

        # Rotate back to image frame
        fx = rx * cos_a + ry * sin_a
        fy = -rx * sin_a + ry * cos_a

        dst_pts.append([fx + cx + margin, fy + cy + margin])

    dst = np.float32(dst_pts)

    M = cv2.getPerspectiveTransform(src, dst)

    bg_color = rng.choice(TABLE_COLORS)

    result = cv2.warpPerspective(
        arr, M, (out_w, out_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color,
    )

    # Center crop back to original dimensions
    cx_out = out_w // 2
    cy_out = out_h // 2
    half = min(w, h) // 2
    result = result[
        cy_out - half : cy_out + half,
        cx_out - half : cx_out + half,
    ]
    result = cv2.resize(result, (w, h))

    return Image.fromarray(result)


# Keep backward-compatible alias
def apply_perspective(
    img: Image.Image,
    elevation_deg: float,
    azimuth_deg: float,
    rng: random.Random,
) -> Image.Image:
    """Apply perspective transform. See apply_projection()."""
    return apply_projection(img, elevation_deg, azimuth_deg, rng, mode="perspective")


class ClipAugmentParams:
    """Pre-sampled clip-level augmentation parameters for temporal coherence.

    Sample once per clip via ``sample_clip_augment_params()``, then pass to
    each frame's ``apply_augmentations()`` call.  Per-frame jitter is added
    automatically so frames look consistent but not identical.
    """

    __slots__ = (
        "brightness", "contrast", "noise_sigma", "blur_radius",
        "apply_blur", "apply_noise", "rotation",
    )

    def __init__(
        self, brightness: float, contrast: float, noise_sigma: float,
        blur_radius: float, apply_blur: bool, apply_noise: bool,
        rotation: float,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.noise_sigma = noise_sigma
        self.blur_radius = blur_radius
        self.apply_blur = apply_blur
        self.apply_noise = apply_noise
        self.rotation = rotation


def _scale_factor(image_size: int) -> float:
    """Return a [0, 1] factor that reduces augmentation intensity for small images."""
    return min(image_size / 224.0, 1.0)


def sample_clip_augment_params(
    rng: random.Random,
    image_size: int = 224,
) -> ClipAugmentParams:
    """Sample augmentation parameters once per clip for temporal coherence."""
    s = _scale_factor(image_size)
    return ClipAugmentParams(
        brightness=rng.uniform(1.0 - 0.3 * s, 1.0 + 0.3 * s),
        contrast=rng.uniform(1.0 - 0.2 * s, 1.0 + 0.2 * s),
        noise_sigma=rng.uniform(5 * s, 20 * s),
        blur_radius=rng.uniform(0.5 * s, 2.0 * s),
        apply_blur=rng.random() < 0.3,
        apply_noise=rng.random() < 0.4,
        rotation=rng.uniform(-5.0, 5.0),
    )


def apply_augmentations(
    img: Image.Image,
    rng: random.Random,
    clip_params: ClipAugmentParams | None = None,
    image_size: int = 224,
    blur_prob: float = 0.3,
    noise_prob: float = 0.4,
    brightness_range: tuple[float, float] = (0.7, 1.3),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    rotation_range: tuple[float, float] = (-5.0, 5.0),
) -> Image.Image:
    """Apply random augmentations to a board image.

    When *clip_params* is provided, base values come from there with
    small per-frame jitter — giving temporal coherence across a clip.
    Intensity is also scaled by *image_size* so that small images
    (e.g. 64x64) aren't destroyed by augmentations tuned for 224x224.

    Args:
        img: Input PIL image.
        rng: Random number generator for reproducibility.
        clip_params: Pre-sampled clip-level params (preferred).
        image_size: Image resolution — augmentations are scaled down for
            smaller images.
        blur_prob: Probability of applying Gaussian blur (ignored when
            *clip_params* is set).
        noise_prob: Probability of adding noise (ignored when
            *clip_params* is set).
        brightness_range: Min/max brightness multiplier (ignored when
            *clip_params* is set).
        contrast_range: Min/max contrast multiplier (ignored when
            *clip_params* is set).
        rotation_range: Min/max rotation in degrees (ignored when
            *clip_params* is set).

    Returns:
        Augmented PIL image.
    """
    s = _scale_factor(image_size)

    if clip_params is not None:
        # Use clip-level base with small per-frame jitter
        brightness = clip_params.brightness + rng.gauss(0, 0.02)
        contrast = clip_params.contrast + rng.gauss(0, 0.02)
        rotation = clip_params.rotation + rng.gauss(0, 0.5)
        do_blur = clip_params.apply_blur
        do_noise = clip_params.apply_noise
        blur_radius = max(0.1, clip_params.blur_radius + rng.gauss(0, 0.1))
        noise_sigma = max(1.0, clip_params.noise_sigma + rng.gauss(0, 1.0))
    else:
        # Legacy: sample fresh per frame, scaled by image size
        brightness = rng.uniform(1.0 - 0.3 * s, 1.0 + 0.3 * s)
        contrast = rng.uniform(1.0 - 0.2 * s, 1.0 + 0.2 * s)
        rotation = rng.uniform(*rotation_range)
        do_blur = rng.random() < blur_prob
        do_noise = rng.random() < noise_prob
        blur_radius = rng.uniform(0.5 * s, 2.0 * s)
        noise_sigma = rng.uniform(5 * s, 20 * s)

    # Random rotation
    img = img.rotate(rotation, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # Brightness
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Contrast
    mean = arr.mean()
    arr = np.array(img, dtype=np.float32)
    arr = np.clip((arr - mean) * contrast + mean, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Gaussian blur
    if do_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Gaussian noise
    if do_noise:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.RandomState(rng.randint(0, 2**31)).normal(0, noise_sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def add_occlusion(
    img: Image.Image,
    rng: random.Random,
    prob: float = 0.2,
    max_rects: int = 3,
    image_size: int = 224,
) -> Image.Image:
    """Add random rectangular occlusions to simulate hands/objects.

    Rectangle sizes are scaled relative to *image_size* so that
    small images aren't overwhelmed.

    Args:
        img: Input PIL image.
        rng: Random number generator.
        prob: Probability of adding any occlusion.
        max_rects: Maximum number of occlusion rectangles.
        image_size: Image resolution for scaling rect sizes.

    Returns:
        Image with potential occlusions.
    """
    if rng.random() > prob:
        return img

    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Scale max rectangle size: w/3 at 224, w/5 at 64
    s = _scale_factor(image_size)
    max_frac = 0.15 + 0.18 * s  # ~0.15 at 64px, ~0.33 at 224px
    min_frac = 0.05 + 0.05 * s  # ~0.05 at 64px, ~0.10 at 224px
    num_rects = rng.randint(1, max_rects)

    for _ in range(num_rects):
        rect_w = rng.randint(max(1, int(w * min_frac)), max(2, int(w * max_frac)))
        rect_h = rng.randint(max(1, int(h * min_frac)), max(2, int(h * max_frac)))
        x0 = rng.randint(0, w - rect_w)
        y0 = rng.randint(0, h - rect_h)
        # Skin-like or dark colors for hand simulation
        color = rng.choice([
            (rng.randint(180, 230), rng.randint(140, 190), rng.randint(100, 150)),
            (rng.randint(30, 80), rng.randint(30, 80), rng.randint(30, 80)),
        ])
        draw.rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=color)

    return img


def _sample_illegal_move(board: chess.Board, rng: random.Random) -> str | None:
    """Sample a random UCI string that is NOT legal in the current position."""
    legal_set = set(board.legal_moves)
    for _ in range(50):
        from_sq = rng.randint(0, 63)
        to_sq = rng.randint(0, 63)
        if from_sq == to_sq:
            continue
        move = chess.Move(from_sq, to_sq)
        if move not in legal_set:
            return move.uci()
    return None


def generate_clip(
    moves: list[str],
    clip_length: int = 16,
    start_move: int = 0,
    image_size: int = 224,
    frames_per_move: int = 4,
    augment: bool = True,
    occlusion_prob: float = 0.2,
    illegal_clip_prob: float = 0.2,
    render_3d_prob: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic training clip from a move sequence.

    Creates a temporal sequence of board images with ground truth
    move and detection labels.

    Args:
        moves: List of UCI move strings for the full game.
        clip_length: Number of frames in the output clip.
        start_move: Which move index to start the clip from.
        image_size: Size of each rendered board image.
        frames_per_move: Average number of frames between moves.
        augment: Whether to apply augmentations.
        occlusion_prob: Probability of occlusion per frame.
        illegal_clip_prob: Probability that this clip contains exactly
            one illegal move (0.2 = 20% of clips have one illegal move,
            80% have none).
        render_3d_prob: Probability of using 3D standing-piece rendering
            (1.0 = always 3D, 0.0 = always 2D flat composite).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            frames: (T, C, H, W) float32 tensor, normalized [0, 1]
            move_targets: (T,) int64 tensor of move vocabulary indices
            detect_targets: (T,) float32 tensor (1.0 if move occurs at frame)
            legal_masks: (T, VOCAB_SIZE) bool tensor
            move_mask: (T,) bool tensor (True at frames where a move occurs)
            fens: list of FEN strings at each frame
    """
    rng = random.Random(seed)
    vocab = get_vocabulary()
    board = chess.Board()

    # Play up to start_move
    for i, uci in enumerate(moves):
        if i >= start_move:
            break
        board.push(chess.Move.from_uci(uci))

    remaining_moves = moves[start_move:]
    frames_list: list[torch.Tensor] = []
    move_targets: list[int] = []
    detect_targets: list[float] = []
    legal_masks: list[torch.Tensor] = []
    move_mask_list: list[bool] = []
    fens: list[str] = []

    move_idx = 0
    frame_count = 0
    next_move_frame = rng.randint(1, frames_per_move)

    # Consistent augmentation params for the clip (board + piece style)
    board_theme = None
    piece_style = None
    piece_material = None
    piece_cache = None
    clip_aug_params = None
    if augment:
        board_theme = select_random_theme(rng).with_perturbation(rng)
        piece_style = select_random_piece_style(rng).with_perturbation(rng)
        piece_material = select_random_material(rng).with_perturbation(rng)
        piece_cache = PieceRenderCache()
        # Per-clip consistent camera angle for perspective
        elevation = rng.uniform(30.0, 75.0)
        azimuth = rng.uniform(0.0, 360.0)
        # Per-clip consistent augmentation intensity
        clip_aug_params = sample_clip_augment_params(rng, image_size)

    # Decide whether to use 3D standing-piece rendering for this clip
    use_3d = augment and rng.random() < render_3d_prob

    flipped = rng.random() < 0.5 if augment else False

    # Per-clip illegal move decision: 20% of clips get exactly one
    # illegal move injected at a randomly chosen move frame.
    clip_has_illegal = rng.random() < illegal_clip_prob
    # Pick which move occurrence (0-indexed) will be the illegal one.
    # Use expected_moves - 1 as max to avoid overshooting.
    expected_moves = max(clip_length // max(frames_per_move, 1) - 1, 1)
    illegal_at_move_occurrence = rng.randint(0, expected_moves - 1)
    move_occurrence_count = 0

    while frame_count < clip_length:
        is_move_frame = (
            frame_count == next_move_frame
            and move_idx < len(remaining_moves)
            and not board.is_game_over()
        )

        if is_move_frame:
            inject_illegal = (
                clip_has_illegal
                and move_occurrence_count == illegal_at_move_occurrence
            )
            move_occurrence_count += 1

            if inject_illegal:
                # Inject a random illegal move without advancing the game
                illegal_uci = _sample_illegal_move(board, rng)
                legal_masks.append(get_legal_mask(board))
                if illegal_uci is not None and vocab.contains(illegal_uci):
                    move_targets.append(vocab.uci_to_index(illegal_uci))
                else:
                    move_targets.append(NO_MOVE_IDX)
                detect_targets.append(1.0)
                move_mask_list.append(True)
                # Don't push to board or advance move_idx — the legal
                # move is still pending for a future frame.
            else:
                uci = remaining_moves[move_idx]
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    legal_mask = get_legal_mask(board)
                    board.push(move)
                    if vocab.contains(uci):
                        move_targets.append(vocab.uci_to_index(uci))
                    else:
                        move_targets.append(NO_MOVE_IDX)
                    detect_targets.append(1.0)
                    move_mask_list.append(True)
                    legal_masks.append(legal_mask)
                else:
                    # Move no longer legal (stale sequence) — skip without
                    # recording a move so subsequent moves stay in sync.
                    legal_masks.append(get_legal_mask(board))
                    move_targets.append(NO_MOVE_IDX)
                    detect_targets.append(0.0)
                    move_mask_list.append(False)
                move_idx += 1
            next_move_frame = frame_count + rng.randint(1, frames_per_move)
        else:
            legal_masks.append(get_legal_mask(board))
            move_targets.append(NO_MOVE_IDX)
            detect_targets.append(0.0)
            move_mask_list.append(False)

        fens.append(board.fen())

        # Render frame
        if use_3d:
            # 3D: warp board first, then composite standing pieces
            from argus.datagen.synth3d import render_3d_scene

            frame_elev = elevation + rng.gauss(0, 1.5)
            frame_azim = azimuth + rng.gauss(0, 2.0)
            img = render_3d_scene(
                board, image_size, flipped,
                frame_elev, frame_azim,
                board_theme, piece_material, piece_cache,
                rng, mode="isometric",
            )
            if augment:
                img = apply_augmentations(img, rng, clip_params=clip_aug_params, image_size=image_size)
                img = add_occlusion(img, rng, prob=occlusion_prob, image_size=image_size)
        else:
            # 2D: composite pieces flat, then warp everything together
            img = render_board_image(
                board, size=image_size, flipped=flipped,
                theme=board_theme, piece_style=piece_style,
                piece_material=piece_material, piece_cache=piece_cache,
                rng=rng,
            )
            if augment:
                frame_elev = elevation + rng.gauss(0, 1.5)
                frame_azim = azimuth + rng.gauss(0, 2.0)
                img = apply_projection(img, frame_elev, frame_azim, rng, mode="isometric")
                img = apply_augmentations(img, rng, clip_params=clip_aug_params, image_size=image_size)
                img = add_occlusion(img, rng, prob=occlusion_prob, image_size=image_size)

        # Convert to tensor (C, H, W), normalized [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        frames_list.append(tensor)
        frame_count += 1

    return {
        "frames": torch.stack(frames_list),
        "move_targets": torch.tensor(move_targets, dtype=torch.long),
        "detect_targets": torch.tensor(detect_targets, dtype=torch.float32),
        "legal_masks": torch.stack(legal_masks),
        "move_mask": torch.tensor(move_mask_list, dtype=torch.bool),
        "fens": fens,
    }


def generate_dataset(
    num_clips: int = 1000,
    clip_length: int = 16,
    image_size: int = 224,
    frames_per_move: int = 4,
    augment: bool = True,
    occlusion_prob: float = 0.2,
    illegal_clip_prob: float = 0.2,
    render_3d_prob: float = 1.0,
    min_moves: int = 10,
    max_moves: int = 80,
    output_dir: str | Path | None = None,
    seed: int = 42,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate a full synthetic dataset of training clips.

    Args:
        num_clips: Number of clips to generate.
        clip_length: Frames per clip.
        image_size: Board image size.
        frames_per_move: Average frames between moves.
        augment: Whether to apply augmentations.
        occlusion_prob: Occlusion probability per frame.
        illegal_clip_prob: Probability that a clip contains exactly
            one illegal move (0.2 = 20% of clips, 80% clean).
        render_3d_prob: Probability of using 3D standing-piece rendering
            per clip (1.0 = always 3D, 0.0 = always 2D).
        min_moves: Minimum game length.
        max_moves: Maximum game length.
        output_dir: If set, save clips to disk incrementally.
        seed: Random seed.
        on_progress: Optional callback(completed, total) called after each clip.

    Returns:
        List of clip dicts (see generate_clip).
    """
    rng = random.Random(seed)
    clips: list[dict[str, Any]] = []

    out: Path | None = None
    clip_offset = 0
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Start numbering after existing clips to avoid overwriting
        existing = list(out.glob("clip_*.pt"))
        if existing:
            nums = []
            for f in existing:
                try:
                    nums.append(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    pass
            if nums:
                clip_offset = max(nums) + 1

    for i in range(num_clips):
        game_seed = rng.randint(0, 2**31)
        moves = sample_random_game(min_moves=min_moves, max_moves=max_moves, seed=game_seed)

        if len(moves) < min_moves:
            continue

        # Pick a random starting point in the game
        max_start = max(0, len(moves) - clip_length // max(frames_per_move, 1))
        start_move = rng.randint(0, max_start) if max_start > 0 else 0

        clip = generate_clip(
            moves=moves,
            clip_length=clip_length,
            start_move=start_move,
            image_size=image_size,
            frames_per_move=frames_per_move,
            augment=augment,
            occlusion_prob=occlusion_prob,
            illegal_clip_prob=illegal_clip_prob,
            render_3d_prob=render_3d_prob,
            seed=game_seed + i,
        )
        clips.append(clip)

        if out is not None:
            save_dict = {k: v for k, v in clip.items() if isinstance(v, torch.Tensor)}
            # Preserve FENs so the validator can replay from the correct position
            if "fens" in clip:
                save_dict["fens"] = clip["fens"]
            clip_num = clip_offset + len(clips) - 1
            torch.save(save_dict, out / f"clip_{clip_num:06d}.pt")

        if on_progress is not None:
            on_progress(len(clips), num_clips)

    return clips
