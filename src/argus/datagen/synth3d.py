"""Blender-based 3D synthetic data generator for Argus training.

Renders realistic chess boards using real 3D Staunton piece models
and Blender's EEVEE engine. Each clip is rendered by calling Blender
as a subprocess with a JSON manifest describing the board theme,
piece material, lighting, and per-frame FEN + camera angles.

Pipeline:
1. Select random board theme, piece material, lighting, piece set
2. Build per-frame game state (FEN + camera angles)
3. Write JSON manifest
4. Call Blender headlessly to render all frames
5. Read back rendered images, apply augmentations
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import chess
import numpy as np
import torch
from PIL import Image

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.data.pgn_sampler import sample_random_game
from argus.datagen.board_themes import select_random_theme
from argus.datagen.lighting import LightingConfig, randomize_lighting
from argus.datagen.piece_renderer import (
    PieceMaterial,
    select_random_material,
)
from argus.datagen.synth2d import apply_augmentations, add_occlusion


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
    # Check environment variable first
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check PATH
    which = shutil.which("blender")
    if which:
        return which

    # Check common install locations
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
    """Get the base models directory."""
    return Path(__file__).parent.parent.parent.parent / "blender" / "models"


def _get_render_script() -> Path:
    """Get the path to the Blender rendering script."""
    return Path(__file__).parent.parent.parent.parent / "blender" / "render_chess.py"


PIECE_SETS = ["staunton"]  # Add more as downloaded


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
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout per clip
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Blender rendering failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    # Read back rendered frames
    images = []
    frame_files = sorted(output_dir.glob("frame_*.png"))
    for fp in frame_files:
        img = Image.open(fp).convert("RGB")
        images.append(img)

    return images


# ---------------------------------------------------------------------------
# Single-frame render (compatibility with synth2d dispatch)
# ---------------------------------------------------------------------------


def render_3d_scene(
    board: chess.Board,
    size: int,
    flipped: bool,
    elevation_deg: float,
    azimuth_deg: float,
    theme,
    piece_material: PieceMaterial,
    piece_cache,  # unused, kept for interface compat
    rng: random.Random,
    mode: str = "isometric",
) -> Image.Image:
    """Render a single frame using Blender.

    Matches the signature expected by synth2d.py's use_3d dispatch.
    """
    fen = board.fen()
    if flipped:
        azimuth_deg = (azimuth_deg + 180.0) % 360.0

    piece_set = _select_piece_set(rng)
    lighting = randomize_lighting(LightingConfig(), seed=rng.randint(0, 2**31))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = tmp_path / "manifest.json"
        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        frames = [{"fen": fen, "elevation": elevation_deg, "azimuth": azimuth_deg}]
        _write_manifest(
            piece_set, piece_material, theme, lighting, frames, manifest_path,
        )

        images = _render_clip_blender(manifest_path, output_dir, size)

    if images:
        return images[0]

    # Fallback: return blank image if render failed
    return Image.new("RGB", (size, size), (128, 128, 128))


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
) -> dict[str, Any]:
    """Generate a synthetic training clip using Blender 3D rendering.

    Same interface as synth2d.generate_clip(). Renders all frames
    in a single Blender subprocess call for efficiency.
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
    board_theme = select_random_theme(rng).with_perturbation(rng) if augment else select_random_theme(rng)
    piece_material = select_random_material(rng).with_perturbation(rng) if augment else select_random_material(rng)
    piece_set = _select_piece_set(rng)
    lighting = randomize_lighting(LightingConfig(), seed=rng.randint(0, 2**31))

    elevation = rng.uniform(30.0, 75.0) if augment else 55.0
    azimuth = rng.uniform(0.0, 360.0) if augment else 0.0
    flipped = rng.random() < 0.5 if augment else False

    # Per-clip illegal move decision: 20% of clips get exactly one
    # illegal move injected at a randomly chosen move frame.
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

        # Camera angle with per-frame jitter
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
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = tmp_path / "manifest.json"
        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        _write_manifest(
            piece_set, piece_material, board_theme,
            lighting, frame_data, manifest_path,
        )

        images = _render_clip_blender(manifest_path, output_dir, image_size)

    # Apply augmentations and convert to tensors
    frames_list: list[torch.Tensor] = []
    for i, img in enumerate(images):
        if augment:
            img = apply_augmentations(img, rng)
            img = add_occlusion(img, rng, prob=occlusion_prob)

        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        frames_list.append(tensor)

    # Pad if Blender returned fewer frames than expected
    while len(frames_list) < clip_length:
        frames_list.append(frames_list[-1] if frames_list else torch.zeros(3, image_size, image_size))

    return {
        "frames": torch.stack(frames_list[:clip_length]),
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
    min_moves: int = 10,
    max_moves: int = 80,
    output_dir: str | Path | None = None,
    seed: int = 42,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate a dataset of training clips with Blender 3D rendering.

    Same interface as synth2d.generate_dataset().
    """
    rng = random.Random(seed)
    clips: list[dict[str, Any]] = []

    out: Path | None = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    for i in range(num_clips):
        game_seed = rng.randint(0, 2**31)
        moves = sample_random_game(
            min_moves=min_moves, max_moves=max_moves, seed=game_seed,
        )

        if len(moves) < min_moves:
            continue

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
            seed=game_seed + i,
        )
        clips.append(clip)

        if out is not None:
            save_dict = {
                k: v for k, v in clip.items() if isinstance(v, torch.Tensor)
            }
            torch.save(save_dict, out / f"clip_{len(clips) - 1:06d}.pt")

        if on_progress is not None:
            on_progress(len(clips), num_clips)

    return clips
