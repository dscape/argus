"""Synthetic data generator for Argus training.

Renders realistic chess boards using Blender's EEVEE engine with real 3D
Staunton piece models. Each clip is a temporal sequence of board images
with ground truth move annotations.

Pipeline:
1. Select random board theme, piece material, lighting, piece set
2. Build per-frame game state (FEN + camera angles)
3. Write JSON manifest
4. Call Blender (via persistent server or subprocess) to render all frames
5. Read back rendered images, apply augmentations
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.data.pgn_sampler import sample_random_game
from argus.datagen.board_themes import select_random_theme
from argus.datagen.lighting import LightingConfig, randomize_lighting
from argus.datagen.piece_renderer import (
    PieceMaterial,
    select_random_material,
)


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------


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
) -> Image.Image:
    """Apply random augmentations to a board image.

    When *clip_params* is provided, base values come from there with
    small per-frame jitter — giving temporal coherence across a clip.
    Intensity is also scaled by *image_size* so that small images
    (e.g. 64x64) aren't destroyed by augmentations tuned for 224x224.
    """
    s = _scale_factor(image_size)

    if clip_params is not None:
        brightness = clip_params.brightness + rng.gauss(0, 0.02)
        contrast = clip_params.contrast + rng.gauss(0, 0.02)
        rotation = clip_params.rotation + rng.gauss(0, 0.5)
        do_blur = clip_params.apply_blur
        do_noise = clip_params.apply_noise
        blur_radius = max(0.1, clip_params.blur_radius + rng.gauss(0, 0.1))
        noise_sigma = max(1.0, clip_params.noise_sigma + rng.gauss(0, 1.0))
    else:
        brightness = rng.uniform(1.0 - 0.3 * s, 1.0 + 0.3 * s)
        contrast = rng.uniform(1.0 - 0.2 * s, 1.0 + 0.2 * s)
        rotation = rng.uniform(-5.0, 5.0)
        do_blur = rng.random() < 0.3
        do_noise = rng.random() < 0.4
        blur_radius = rng.uniform(0.5 * s, 2.0 * s)
        noise_sigma = rng.uniform(5 * s, 20 * s)

    img = img.rotate(rotation, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    mean = arr.mean()
    arr = np.array(img, dtype=np.float32)
    arr = np.clip((arr - mean) * contrast + mean, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    if do_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

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
    """Add random rectangular occlusions to simulate hands/objects."""
    if rng.random() > prob:
        return img

    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    s = _scale_factor(image_size)
    max_frac = 0.15 + 0.18 * s
    min_frac = 0.05 + 0.05 * s
    num_rects = rng.randint(1, max_rects)

    for _ in range(num_rects):
        rect_w = rng.randint(max(1, int(w * min_frac)), max(2, int(w * max_frac)))
        rect_h = rng.randint(max(1, int(h * min_frac)), max(2, int(h * max_frac)))
        x0 = rng.randint(0, w - rect_w)
        y0 = rng.randint(0, h - rect_h)
        color = rng.choice([
            (rng.randint(180, 230), rng.randint(140, 190), rng.randint(100, 150)),
            (rng.randint(30, 80), rng.randint(30, 80), rng.randint(30, 80)),
        ])
        draw.rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=color)

    return img


# ---------------------------------------------------------------------------
# Blender executable discovery
# ---------------------------------------------------------------------------

_BLENDER_SEARCH_PATHS = [
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/local/bin/blender",
    "/usr/bin/blender",
    "/opt/homebrew/bin/blender",
]


def _find_blender() -> str:
    """Locate the Blender executable."""
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    which = shutil.which("blender")
    if which:
        return which
    for p in _BLENDER_SEARCH_PATHS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Blender not found. Install it (brew install --cask blender) "
        "or set BLENDER_PATH environment variable."
    )


# ---------------------------------------------------------------------------
# Piece set discovery
# ---------------------------------------------------------------------------

def _get_models_dir() -> Path:
    return Path(__file__).parent.parent.parent.parent / "blender" / "models"


def _get_render_script() -> Path:
    return Path(__file__).parent.parent.parent.parent / "blender" / "render_chess.py"


PIECE_SETS = ["staunton"]


def _select_piece_set(rng: random.Random) -> str:
    """Select a random piece set from available sets."""
    available = []
    models_dir = _get_models_dir()
    for ps in PIECE_SETS:
        ps_dir = models_dir / ps
        if ps_dir.is_dir() and any(ps_dir.glob("*.STL")):
            available.append(ps)
    if not available:
        raise FileNotFoundError(
            f"No piece sets found in {models_dir}. "
            f"Download STL files to blender/models/staunton/"
        )
    return rng.choice(available)


# ---------------------------------------------------------------------------
# Material/theme to manifest conversion
# ---------------------------------------------------------------------------

_MATERIAL_PARAMS = {
    "plastic": {"roughness": 0.35, "metallic": 0.0},
    "wood": {"roughness": 0.50, "metallic": 0.0},
    "metal": {"roughness": 0.15, "metallic": 0.9},
}


def _material_to_dict(mat: PieceMaterial) -> dict:
    """Convert PieceMaterial to manifest dict."""
    params = _MATERIAL_PARAMS.get(mat.material_type, _MATERIAL_PARAMS["plastic"])
    return {
        "name": mat.name,
        "white_color": list(mat.white_color),
        "black_color": list(mat.black_color),
        "type": mat.material_type,
        **params,
    }


def _theme_to_dict(theme) -> dict:
    """Convert BoardTheme to manifest dict."""
    d = {
        "light": theme.light,
        "dark": theme.dark,
        "texture_type": theme.texture_type,
    }
    if theme.border_color is not None:
        d["border_color"] = theme.border_color
    return d


# ---------------------------------------------------------------------------
# Manifest writing and Blender invocation
# ---------------------------------------------------------------------------


def _write_manifest(
    piece_set: str,
    material: PieceMaterial,
    board_theme,
    lighting: dict,
    frames: list[dict],
    path: Path,
) -> None:
    """Write a JSON manifest for the Blender rendering script."""
    manifest = {
        "piece_set": piece_set,
        "material": _material_to_dict(material),
        "board_theme": _theme_to_dict(board_theme),
        "lighting": lighting,
        "frames": frames,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _render_clip_blender(
    manifest_path: Path,
    output_dir: Path,
    resolution: int,
    quality: str = "training",
) -> list[Image.Image]:
    """Call Blender to render all frames in a manifest.

    Returns list of PIL Images.
    """
    blender = _find_blender()
    render_script = str(_get_render_script())

    cmd = [
        blender,
        "--background",
        "--python", render_script,
        "--",
        "--manifest", str(manifest_path),
        "--output-dir", str(output_dir),
        "--resolution", str(resolution),
        "--quality", quality,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Blender rendering failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    images = []
    frame_files = sorted(output_dir.glob("frame_*.png"))
    for fp in frame_files:
        img = Image.open(fp).convert("RGB")
        images.append(img)

    return images


# ---------------------------------------------------------------------------
# Clip / dataset generation
# ---------------------------------------------------------------------------


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
    seed: int | None = None,
    quality: str = "training",
    server: "BlenderServerClient | None" = None,
) -> dict[str, Any]:
    """Generate a synthetic training clip using Blender 3D rendering.

    Renders all frames in a single Blender call for efficiency.

    Args:
        moves: List of UCI move strings for the full game.
        clip_length: Number of frames in the output clip.
        start_move: Which move index to start the clip from.
        image_size: Size of each rendered board image.
        frames_per_move: Average number of frames between moves.
        augment: Whether to apply augmentations.
        occlusion_prob: Probability of occlusion per frame.
        illegal_clip_prob: Probability that this clip contains exactly
            one illegal move.
        seed: Random seed for reproducibility.
        quality: Render quality preset ('training' or 'high').
        server: Optional BlenderServerClient for persistent rendering.
            If not provided, spawns a Blender subprocess per clip.

    Returns:
        Dict with keys: frames, move_targets, detect_targets,
        legal_masks, move_mask, fens.
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
    move_targets: list[int] = []
    detect_targets: list[float] = []
    legal_masks: list[torch.Tensor] = []
    move_mask_list: list[bool] = []
    fens: list[str] = []

    move_idx = 0
    frame_count = 0
    next_move_frame = rng.randint(1, frames_per_move)

    # Per-clip consistent style
    theme_base = select_random_theme(rng)
    board_theme = theme_base.with_perturbation(rng) if augment else theme_base
    mat_base = select_random_material(rng)
    piece_material = mat_base.with_perturbation(rng) if augment else mat_base
    piece_set = _select_piece_set(rng)
    lighting = randomize_lighting(LightingConfig(), seed=rng.randint(0, 2**31))

    elevation = rng.uniform(30.0, 75.0) if augment else 55.0
    azimuth = rng.uniform(0.0, 360.0) if augment else 0.0
    flipped = rng.random() < 0.5 if augment else False
    clip_aug_params = sample_clip_augment_params(rng, image_size) if augment else None

    # Per-clip illegal move decision
    clip_has_illegal = rng.random() < illegal_clip_prob
    expected_moves = max(clip_length // max(frames_per_move, 1) - 1, 1)
    illegal_at_move_occurrence = rng.randint(0, expected_moves - 1)
    move_occurrence_count = 0

    # First pass: advance game state, collect FENs and camera angles
    frame_data: list[dict] = []

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
                illegal_uci = _sample_illegal_move(board, rng)
                legal_masks.append(get_legal_mask(board))
                if illegal_uci is not None and vocab.contains(illegal_uci):
                    move_targets.append(vocab.uci_to_index(illegal_uci))
                else:
                    move_targets.append(NO_MOVE_IDX)
                detect_targets.append(1.0)
                move_mask_list.append(True)
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

        fen = board.fen()
        fens.append(fen)

        frame_elev = elevation + (rng.gauss(0, 1.5) if augment else 0)
        frame_azim = azimuth + (rng.gauss(0, 2.0) if augment else 0)
        if flipped:
            frame_azim = (frame_azim + 180.0) % 360.0

        frame_data.append({
            "fen": fen,
            "elevation": frame_elev,
            "azimuth": frame_azim,
        })
        frame_count += 1

    # Second pass: batch render all frames via Blender
    manifest_dict = {
        "piece_set": piece_set,
        "material": _material_to_dict(piece_material),
        "board_theme": _theme_to_dict(board_theme),
        "lighting": lighting,
        "frames": frame_data,
    }

    if server is not None:
        images = server.render_clip(manifest_dict, image_size)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            output_dir = tmp_path / "frames"
            output_dir.mkdir()

            _write_manifest(
                piece_set, piece_material, board_theme,
                lighting, frame_data, manifest_path,
            )

            images = _render_clip_blender(
                manifest_path, output_dir, image_size, quality=quality,
            )

    # Apply augmentations and convert to tensors
    frames_list: list[torch.Tensor] = []
    for i, img in enumerate(images):
        if augment:
            img = apply_augmentations(img, rng, clip_params=clip_aug_params, image_size=image_size)
            img = add_occlusion(img, rng, prob=occlusion_prob, image_size=image_size)

        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        frames_list.append(tensor)

    # Pad if Blender returned fewer frames than expected
    while len(frames_list) < clip_length:
        fallback = frames_list[-1] if frames_list else torch.zeros(3, image_size, image_size)
        frames_list.append(fallback)

    return {
        "frames": torch.stack(frames_list[:clip_length]),
        "move_targets": torch.tensor(move_targets, dtype=torch.long),
        "detect_targets": torch.tensor(detect_targets, dtype=torch.float32),
        "legal_masks": torch.stack(legal_masks),
        "move_mask": torch.tensor(move_mask_list, dtype=torch.bool),
        "fens": fens,
    }


def _save_clip(clip: dict[str, Any], out: Path, clip_num: int) -> None:
    """Save a clip dict to disk as a .pt file."""
    save_dict = {
        k: v for k, v in clip.items() if isinstance(v, torch.Tensor)
    }
    if "fens" in clip:
        save_dict["fens"] = clip["fens"]
    torch.save(save_dict, out / f"clip_{clip_num:06d}.pt")


def generate_dataset(
    num_clips: int = 1000,
    clip_length: int = 16,
    image_size: int = 224,
    frames_per_move: int = 4,
    augment: bool = True,
    occlusion_prob: float = 0.2,
    illegal_clip_prob: float = 0.2,
    min_moves: int = 10,
    max_moves: int = 80,
    output_dir: str | Path | None = None,
    seed: int = 42,
    on_progress: Callable[[int, int], None] | None = None,
    quality: str = "training",
    num_workers: int = 1,
) -> list[dict[str, Any]]:
    """Generate a dataset of training clips with Blender 3D rendering.

    Args:
        num_clips: Number of clips to generate.
        clip_length: Frames per clip.
        image_size: Board image size.
        frames_per_move: Average frames between moves.
        augment: Whether to apply augmentations.
        occlusion_prob: Occlusion probability per frame.
        illegal_clip_prob: Probability of illegal move injection per clip.
        min_moves: Minimum game length.
        max_moves: Maximum game length.
        output_dir: If set, save clips to disk incrementally.
        seed: Random seed.
        on_progress: Optional callback(completed, total).
        quality: Render quality preset ('training' or 'high').
        num_workers: Number of parallel Blender render workers.
    """
    from argus.datagen.blender_server import BlenderServerClient

    rng = random.Random(seed)
    clips: list[dict[str, Any]] = []

    out: Path | None = None
    clip_offset = 0
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
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

    # Pre-compute all clip parameters for deterministic RNG
    clip_params = []
    for i in range(num_clips):
        game_seed = rng.randint(0, 2**31)
        moves = sample_random_game(
            min_moves=min_moves, max_moves=max_moves, seed=game_seed,
        )
        if len(moves) < min_moves:
            continue
        max_start = max(0, len(moves) - clip_length // max(frames_per_move, 1))
        start_move = rng.randint(0, max_start) if max_start > 0 else 0
        clip_params.append((i, moves, start_move, game_seed))

    # Connect to external BlenderServer (started via `make blender-server`)
    server = None
    if augment:
        try:
            server = BlenderServerClient.connect()
        except ConnectionError:
            import logging
            logging.getLogger(__name__).warning(
                "BlenderServer not running. Start it with `make blender-server`. "
                "Falling back to subprocess-per-clip.",
            )

    for i, moves, start_move, game_seed in clip_params:
        clip = generate_clip(
            moves=moves,
            clip_length=clip_length,
            start_move=start_move,
            image_size=image_size,
            frames_per_move=frames_per_move,
            augment=augment,
            occlusion_prob=occlusion_prob,
            illegal_clip_prob=illegal_clip_prob,
            seed=game_seed + i,
            quality=quality,
            server=server,
        )
        clips.append(clip)

        if out is not None:
            _save_clip(clip, out, clip_offset + len(clips) - 1)

        if on_progress is not None:
            on_progress(len(clips), num_clips)

    return clips
