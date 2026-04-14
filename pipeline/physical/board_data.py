"""Datasets and rendering helpers for physical-board state probes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from argus.chess.board_state import fen_to_square_targets
from argus.data.pgn_sampler import sample_random_game
from argus.datagen.board_themes import render_textured_board, select_random_theme
from argus.datagen.piece_renderer import (
    PieceRenderCache,
    render_pieces_layer,
    select_random_material,
)
from argus.datagen.synth import generate_dataset as generate_synthetic_clips
from pipeline.physical import splits
from pipeline.shared import SQUARE_CLASS_NAMES

INPUT_SIZE = 224
NUM_SQUARE_CLASSES = len(SQUARE_CLASS_NAMES)
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_DEFAULT_TRAIN_ROOT = _PROJECT_ROOT / "data" / "physical" / "train"


@dataclass(frozen=True)
class PhysicalEvalBoardRow:
    annotation_id: str
    board_path: str
    labels: tuple[int, ...]
    source_video_id: str | None
    clip_path: str | None = None
    frame_index: int | None = None


class PhysicalSyntheticBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random physical-board renders with exact square-state targets."""

    def __init__(
        self,
        *,
        num_positions: int,
        image_size: int = INPUT_SIZE,
        seed: int = 42,
        augment: bool = True,
        min_moves: int = 10,
        max_moves: int = 80,
        min_ply: int = 8,
    ) -> None:
        self.num_positions = num_positions
        self.image_size = image_size
        self.seed = seed
        self.augment = augment
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.min_ply = min_ply
        self._piece_cache = PieceRenderCache()

    def __len__(self) -> int:
        return self.num_positions

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_bgr, targets = render_random_physical_board(
            size=self.image_size,
            seed=self.seed + index,
            augment=self.augment,
            min_moves=self.min_moves,
            max_moves=self.max_moves,
            min_ply=self.min_ply,
            cache=self._piece_cache,
        )
        return preprocess_board_image(image_bgr, size=self.image_size), targets


class PhysicalSyntheticRenderedBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Perspective-rendered 3D synthetic boards from the physical synthetic pipeline."""

    def __init__(
        self,
        *,
        num_positions: int,
        image_size: int = INPUT_SIZE,
        seed: int = 42,
        augment: bool = True,
        min_moves: int = 10,
        max_moves: int = 80,
        occlusion_prob: float = 0.2,
    ) -> None:
        clips = generate_synthetic_clips(
            num_clips=num_positions,
            clip_length=1,
            image_size=image_size,
            frames_per_move=4,
            augment=augment,
            occlusion_prob=occlusion_prob,
            min_moves=min_moves,
            max_moves=max_moves,
            seed=seed,
        )
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for clip in clips:
            frames = clip["frames"]
            fens = clip.get("fens") or []
            board_flipped = bool(clip.get("board_flipped", False))
            if frames.shape[0] == 0 or not fens:
                continue
            frame = frames[0]
            if frame.dtype != torch.float32:
                frame = frame.to(torch.float32)
            targets = fen_to_square_targets(fens[0], board_flipped=board_flipped)
            self.samples.append((normalize_rgb_tensor(frame), targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


class PhysicalAnnotatedBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Rectified boards loaded from one physical annotation root."""

    def __init__(
        self,
        *,
        annotation_root: str | Path,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalBoardRow] | None = None,
    ) -> None:
        self.annotation_root = Path(annotation_root)
        self.image_size = image_size
        self.rows = rows or load_annotated_board_rows(self.annotation_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = cv2.imread(str(_PROJECT_ROOT / row.board_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load rectified board image: {row.board_path}")
        targets = torch.tensor(row.labels, dtype=torch.long)
        return preprocess_board_image(image, size=self.image_size), targets


class PhysicalValBoardDataset(PhysicalAnnotatedBoardDataset):
    """Held-out real rectified boards used for physical validation."""

    def __init__(
        self,
        *,
        annotation_root: str | Path = _DEFAULT_VAL_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalBoardRow] | None = None,
    ) -> None:
        super().__init__(annotation_root=annotation_root, image_size=image_size, rows=rows)


class PhysicalTrainBoardDataset(PhysicalAnnotatedBoardDataset):
    """Manually labeled non-held-out boards for physical training."""

    def __init__(
        self,
        *,
        annotation_root: str | Path = _DEFAULT_TRAIN_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalBoardRow] | None = None,
    ) -> None:
        super().__init__(annotation_root=annotation_root, image_size=image_size, rows=rows)


PhysicalEvalBoardDataset = PhysicalValBoardDataset
PhysicalManualTrainBoardDataset = PhysicalTrainBoardDataset



def load_annotated_board_rows(annotation_root: str | Path) -> list[PhysicalEvalBoardRow]:
    splits.ensure_annotation_layout_migrated()
    annotations_path = Path(annotation_root) / "board_annotations.jsonl"
    if not annotations_path.exists():
        raise ValueError(f"Physical board annotations not found: {annotations_path}")

    rows: list[PhysicalEvalBoardRow] = []
    for line in annotations_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list) or len(raw_labels) != 64:
            continue
        if any(label is None for label in raw_labels):
            continue
        rows.append(
            PhysicalEvalBoardRow(
                annotation_id=str(payload["annotation_id"]),
                board_path=str(payload["rectified_board_path"]),
                labels=tuple(int(label) for label in raw_labels),
                source_video_id=(
                    str(payload["source_video_id"])
                    if payload.get("source_video_id") is not None
                    else None
                ),
                clip_path=(
                    str(payload["clip_path"]) if payload.get("clip_path") is not None else None
                ),
                frame_index=(
                    int(payload["frame_index"])
                    if payload.get("frame_index") is not None
                    else None
                ),
            )
        )
    return rows



def load_val_board_rows(
    annotation_root: str | Path = _DEFAULT_VAL_ROOT,
) -> list[PhysicalEvalBoardRow]:
    return load_annotated_board_rows(annotation_root)


def load_eval_board_rows(
    eval_root: str | Path = _DEFAULT_VAL_ROOT,
) -> list[PhysicalEvalBoardRow]:
    return load_annotated_board_rows(eval_root)


def preprocess_board_image(image_bgr: np.ndarray, *, size: int = INPUT_SIZE) -> torch.Tensor:
    """Resize BGR input, convert to RGB, and apply DINO/ImageNet normalization."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return normalize_rgb_tensor(tensor)


def normalize_rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to an RGB tensor in [0, 1]."""
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def render_random_physical_board(
    *,
    size: int,
    seed: int,
    augment: bool,
    min_moves: int,
    max_moves: int,
    min_ply: int,
    cache: PieceRenderCache | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Render one synthetic physical board plus 64 square targets."""
    rng = random.Random(seed)
    moves = sample_random_game(min_moves=min_moves, max_moves=max_moves, seed=seed)
    board = chess.Board()
    lower_bound = min(min_ply, len(moves))
    ply = rng.randint(lower_bound, len(moves)) if moves else 0
    for move_uci in moves[:ply]:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)

    render_size = size * 2 if augment else size
    theme = select_random_theme(rng).with_perturbation(rng)
    material = select_random_material(rng).with_perturbation(rng)
    light_dir = (
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.7, -0.3),
        rng.uniform(0.5, 1.0),
    )
    board_rgb = render_textured_board(size=render_size, theme=theme, flipped=False, rng=rng)
    pieces_rgba = render_pieces_layer(
        board,
        size=render_size,
        flipped=False,
        material=material,
        light_dir=light_dir,
        cache=cache,
        rng=rng,
    )
    if augment:
        pieces_rgba = _apply_piece_rectification_artifacts(pieces_rgba, rng)
    board_pil = Image.fromarray(board_rgb)
    board_pil.paste(pieces_rgba, (0, 0), pieces_rgba)
    image_bgr = cv2.cvtColor(np.array(board_pil), cv2.COLOR_RGB2BGR)
    if augment:
        image_bgr = augment_physical_board_image(image_bgr, rng)
    targets = fen_to_square_targets(board.fen(), board_flipped=False)
    return image_bgr, targets


def augment_physical_board_image(image_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply photometric and rectification-like artifacts to a rectified board."""
    image = image_bgr.astype(np.uint8)

    if rng.random() < 0.55:
        image = _apply_resampling_artifacts(image, rng)

    image_f = image.astype(np.float32)
    image_f = np.clip(image_f * rng.uniform(0.85, 1.15) + rng.uniform(-16.0, 16.0), 0, 255)

    if rng.random() < 0.25:
        image_f = _apply_directional_blur(image_f.astype(np.uint8), rng).astype(np.float32)

    if rng.random() < 0.30:
        ksize = rng.choice([3, 5])
        image_f = cv2.GaussianBlur(image_f.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)

    if rng.random() < 0.25:
        quality = rng.randint(60, 92)
        _, encoded = cv2.imencode(
            ".jpg",
            image_f.astype(np.uint8),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            image_f = decoded.astype(np.float32)

    if rng.random() < 0.20:
        shadow = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(1.0, 1.1), image_f.shape[0])
        image_f *= shadow[:, None, None]

    return np.clip(image_f, 0, 255).astype(np.uint8)


def _apply_piece_rectification_artifacts(
    pieces_rgba: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Approximate board-plane rectification artifacts on elevated pieces only."""
    rgba = np.array(pieces_rgba, dtype=np.uint8)
    height, width = rgba.shape[:2]
    square_size = height // 8
    far_to_near = rng.random() < 0.5
    distorted = np.zeros_like(rgba)

    far_scale_x = rng.uniform(1.15, 1.65)
    near_scale_x = rng.uniform(0.95, 1.20)
    far_shear_x = rng.uniform(-0.18, 0.18)
    near_shear_x = rng.uniform(-0.05, 0.05)
    far_shift_x = rng.uniform(-0.30, 0.30) * square_size
    near_shift_x = rng.uniform(-0.08, 0.08) * square_size
    far_shift_y = rng.uniform(-0.12, 0.03) * square_size
    near_shift_y = rng.uniform(-0.03, 0.03) * square_size

    for row in range(8):
        y1 = row * square_size
        y2 = y1 + square_size
        strip = rgba[y1:y2]
        t = row / 7.0
        if not far_to_near:
            t = 1.0 - t

        matrix = _affine_matrix_about_center(
            width,
            strip.shape[0],
            scale_x=_lerp(far_scale_x, near_scale_x, t),
            shear_x=_lerp(far_shear_x, near_shear_x, t),
            shift_x=_lerp(far_shift_x, near_shift_x, t),
            shift_y=_lerp(far_shift_y, near_shift_y, t),
        )
        distorted[y1:y2] = cv2.warpAffine(
            strip,
            matrix,
            (width, strip.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    if rng.random() < 0.80:
        distorted = _apply_directional_blur(distorted, rng, max_angle_degrees=18.0)

    mix = rng.uniform(0.65, 0.90)
    blended = np.clip(
        rgba.astype(np.float32) * (1.0 - mix) + distorted.astype(np.float32) * mix,
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(blended, "RGBA")


def _apply_resampling_artifacts(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Introduce anisotropic down/up-sampling similar to broadcast rectification."""
    height, width = image.shape[:2]
    down_width = max(16, int(width * rng.uniform(0.55, 0.85)))
    down_height = max(16, int(height * rng.uniform(0.65, 0.95)))
    downsampled = cv2.resize(
        image,
        (down_width, down_height),
        interpolation=rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR]),
    )
    return cv2.resize(
        downsampled,
        (width, height),
        interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]),
    )


def _apply_directional_blur(
    image: np.ndarray,
    rng: random.Random,
    *,
    max_angle_degrees: float = 12.0,
) -> np.ndarray:
    """Apply a mostly horizontal line blur to mimic motion or rectification smear."""
    kernel_size = rng.choice([3, 5, 7, 9])
    angle = rng.uniform(-max_angle_degrees, max_angle_degrees)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(
        ((kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0),
        angle,
        1.0,
    )
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel /= max(float(kernel.sum()), 1e-6)

    blurred = cv2.filter2D(image.astype(np.float32), -1, kernel)
    mix = rng.uniform(0.35, 0.75)
    return np.clip(image.astype(np.float32) * (1.0 - mix) + blurred * mix, 0, 255).astype(
        np.uint8
    )


def _affine_matrix_about_center(
    width: int,
    height: int,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shear_x: float = 0.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> np.ndarray:
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    linear = np.array(
        [[scale_x, shear_x], [0.0, scale_y]],
        dtype=np.float32,
    )
    translation = center - linear @ center + np.array([shift_x, shift_y], dtype=np.float32)
    return np.concatenate([linear, translation[:, None]], axis=1)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t
