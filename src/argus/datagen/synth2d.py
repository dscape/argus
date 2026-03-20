"""Full 2D synthetic data generator for Argus training.

Renders chess board images using python-chess SVG rendering and PIL,
applies augmentations, and generates temporal clips with ground truth
move annotations.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Callable

import chess
import chess.svg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.data.pgn_sampler import sample_random_game


def render_board_image(
    board: chess.Board,
    size: int = 224,
    flipped: bool = False,
    colors: dict[str, str] | None = None,
) -> Image.Image:
    """Render a chess board position to a PIL Image.

    Args:
        board: Chess board position.
        size: Output image size (square).
        flipped: Whether to flip the board (black perspective).
        colors: Optional dict with 'light' and 'dark' square colors.

    Returns:
        PIL Image of the board.
    """
    default_colors = {"light": "#F0D9B5", "dark": "#B58863"}
    c = colors or default_colors

    svg_text = chess.svg.board(
        board,
        size=size,
        flipped=flipped,
        colors={"square light": c["light"], "square dark": c["dark"]},
    )
    # Convert SVG to PIL Image via cairosvg if available, else fallback
    try:
        import cairosvg

        png_data = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), output_width=size, output_height=size)
        img = Image.open(io.BytesIO(png_data)).convert("RGB")
    except ImportError:
        # Fallback: create a simple colored board representation
        img = _render_simple_board(board, size, flipped, c)
    return img


def _render_simple_board(
    board: chess.Board,
    size: int,
    flipped: bool,
    colors: dict[str, str],
) -> Image.Image:
    """Simple fallback board renderer without SVG dependencies."""
    img = Image.new("RGB", (size, size), "white")
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

            x0 = file * sq_size
            y0 = rank * sq_size
            is_light = (file + rank) % 2 == 0
            color = colors["light"] if is_light else colors["dark"]
            draw.rectangle([x0, y0, x0 + sq_size, y0 + sq_size], fill=color)

            piece = board.piece_at(sq)
            if piece is not None:
                symbol = piece_symbols.get(piece.piece_type, "?")
                if piece.color == chess.BLACK:
                    symbol = symbol.lower()
                text_color = "white" if piece.color == chess.BLACK else "black"
                cx = x0 + sq_size // 2
                cy = y0 + sq_size // 2
                draw.text((cx - 4, cy - 6), symbol, fill=text_color)

    return img


def apply_augmentations(
    img: Image.Image,
    rng: random.Random,
    blur_prob: float = 0.3,
    noise_prob: float = 0.4,
    brightness_range: tuple[float, float] = (0.7, 1.3),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    rotation_range: tuple[float, float] = (-5.0, 5.0),
) -> Image.Image:
    """Apply random augmentations to a board image.

    Args:
        img: Input PIL image.
        rng: Random number generator for reproducibility.
        blur_prob: Probability of applying Gaussian blur.
        noise_prob: Probability of adding noise.
        brightness_range: Min/max brightness multiplier.
        contrast_range: Min/max contrast multiplier.
        rotation_range: Min/max rotation in degrees.

    Returns:
        Augmented PIL image.
    """
    # Random rotation
    angle = rng.uniform(*rotation_range)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # Brightness
    factor = rng.uniform(*brightness_range)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Contrast
    factor = rng.uniform(*contrast_range)
    mean = arr.mean()
    arr = np.array(img, dtype=np.float32)
    arr = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Gaussian blur
    if rng.random() < blur_prob:
        radius = rng.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Gaussian noise
    if rng.random() < noise_prob:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.RandomState(rng.randint(0, 2**31)).normal(0, rng.uniform(5, 20), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def add_occlusion(
    img: Image.Image,
    rng: random.Random,
    prob: float = 0.2,
    max_rects: int = 3,
) -> Image.Image:
    """Add random rectangular occlusions to simulate hands/objects.

    Args:
        img: Input PIL image.
        rng: Random number generator.
        prob: Probability of adding any occlusion.
        max_rects: Maximum number of occlusion rectangles.

    Returns:
        Image with potential occlusions.
    """
    if rng.random() > prob:
        return img

    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    num_rects = rng.randint(1, max_rects)

    for _ in range(num_rects):
        rect_w = rng.randint(w // 10, w // 3)
        rect_h = rng.randint(h // 10, h // 3)
        x0 = rng.randint(0, w - rect_w)
        y0 = rng.randint(0, h - rect_h)
        # Skin-like or dark colors for hand simulation
        color = rng.choice([
            (rng.randint(180, 230), rng.randint(140, 190), rng.randint(100, 150)),
            (rng.randint(30, 80), rng.randint(30, 80), rng.randint(30, 80)),
        ])
        draw.rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=color)

    return img


def generate_clip(
    moves: list[str],
    clip_length: int = 16,
    start_move: int = 0,
    image_size: int = 224,
    frames_per_move: int = 4,
    augment: bool = True,
    occlusion_prob: float = 0.2,
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

    # Consistent augmentation params for the clip (board style)
    board_colors = None
    if augment:
        light_r = rng.randint(200, 255)
        light_g = rng.randint(200, 240)
        light_b = rng.randint(170, 210)
        dark_r = rng.randint(100, 180)
        dark_g = rng.randint(80, 140)
        dark_b = rng.randint(50, 120)
        board_colors = {
            "light": f"#{light_r:02x}{light_g:02x}{light_b:02x}",
            "dark": f"#{dark_r:02x}{dark_g:02x}{dark_b:02x}",
        }

    flipped = rng.random() < 0.5 if augment else False

    while frame_count < clip_length:
        is_move_frame = (
            frame_count == next_move_frame
            and move_idx < len(remaining_moves)
            and not board.is_game_over()
        )

        if is_move_frame:
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
                # Illegal move in sequence, treat as no-move frame
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
        img = render_board_image(board, size=image_size, flipped=flipped, colors=board_colors)
        if augment:
            img = apply_augmentations(img, rng)
            img = add_occlusion(img, rng, prob=occlusion_prob)

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
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

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
            seed=game_seed + i,
        )
        clips.append(clip)

        if out is not None:
            save_dict = {k: v for k, v in clip.items() if isinstance(v, torch.Tensor)}
            torch.save(save_dict, out / f"clip_{len(clips) - 1:06d}.pt")

        if on_progress is not None:
            on_progress(len(clips), num_clips)

    return clips
