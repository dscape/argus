"""Datasets and preprocessing for physical-board square classification."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.physical import splits
from pipeline.shared import NUM_SQUARE_CLASSES, SQUARE_CLASS_NAMES

try:
    from argus.datagen.board_themes import BOARD_THEMES as DATAGEN_THEMES
    from argus.datagen.board_themes import generate_square_texture
    from argus.datagen.piece_renderer import (
        PieceRenderCache,
        render_piece_sprite,
        select_random_material,
    )
except ImportError as exc:  # pragma: no cover - surfaced when training code runs
    raise RuntimeError(
        "Physical square data generation requires argus.datagen rendering helpers"
    ) from exc

CLASS_NAMES = SQUARE_CLASS_NAMES
NUM_CLASSES = NUM_SQUARE_CLASSES
INPUT_SIZE = 224
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"


@dataclass(frozen=True)
class PhysicalEvalRow:
    annotation_id: str
    crop_path: str
    label_index: int
    label_name: str
    source_video_id: str | None
    square_index: int


class PhysicalSyntheticSquareDataset(Dataset[tuple[torch.Tensor, int]]):
    """Deterministic synthetic physical square crops for linear-probe training."""

    def __init__(
        self,
        *,
        num_samples_per_class: int,
        image_size: int = INPUT_SIZE,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.seed = seed
        self.augment = augment
        self._render_cache = PieceRenderCache()

    def __len__(self) -> int:
        return self.num_samples_per_class * NUM_CLASSES

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        class_index = index // self.num_samples_per_class
        sample_index = index % self.num_samples_per_class
        rng = random.Random(self.seed + class_index * 1_000_003 + sample_index)
        image = render_physical_square(
            class_index,
            size=self.image_size,
            rng=rng,
            cache=self._render_cache,
        )
        if self.augment:
            image = augment_physical_square_image(image, rng)
        return preprocess_square_image(image, size=self.image_size), class_index


class PhysicalValSquareDataset(Dataset[tuple[torch.Tensor, int]]):
    """Held-out real square crops used for physical validation."""

    def __init__(
        self,
        *,
        annotation_root: str | Path = _DEFAULT_VAL_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalEvalRow] | None = None,
        max_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        self.annotation_root = Path(annotation_root)
        self.image_size = image_size
        if rows is None:
            rows = load_eval_rows(self.annotation_root)
        if max_per_class is not None:
            rows = _sample_rows_per_class(rows, max_per_class=max_per_class, seed=seed)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = cv2.imread(str(_PROJECT_ROOT / row.crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load square crop: {row.crop_path}")
        return preprocess_square_image(image, size=self.image_size), row.label_index


PhysicalEvalSquareDataset = PhysicalValSquareDataset


def load_eval_rows(
    eval_root: str | Path = _DEFAULT_VAL_ROOT,
) -> list[PhysicalEvalRow]:
    splits.ensure_annotation_layout_migrated()
    manifest_path = Path(eval_root) / "square_manifest.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"Physical validation square manifest not found: {manifest_path}")

    rows: list[PhysicalEvalRow] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            PhysicalEvalRow(
                annotation_id=str(payload["annotation_id"]),
                crop_path=str(payload["crop_path"]),
                label_index=int(payload["label_index"]),
                label_name=str(payload["label_name"]),
                source_video_id=(
                    str(payload["source_video_id"])
                    if payload.get("source_video_id") is not None
                    else None
                ),
                square_index=int(payload["square_index"]),
            )
        )
    return rows


def split_rectified_board_into_squares(board_bgr: np.ndarray) -> list[np.ndarray]:
    """Split a rectified board image into 64 equal square crops."""
    height, width = board_bgr.shape[:2]
    if height != width:
        raise ValueError(f"Board must be square, got {width}x{height}")

    square_size = height // 8
    crops: list[np.ndarray] = []
    for row in range(8):
        for col in range(8):
            y1 = row * square_size
            x1 = col * square_size
            crops.append(board_bgr[y1 : y1 + square_size, x1 : x1 + square_size].copy())
    return crops


def preprocess_square_image(image_bgr: np.ndarray, *, size: int = INPUT_SIZE) -> torch.Tensor:
    """Resize BGR input, convert to RGB, and apply DINO/ImageNet normalization."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


def render_physical_square(
    class_index: int,
    *,
    size: int,
    rng: random.Random,
    cache: PieceRenderCache | None = None,
) -> np.ndarray:
    """Render one square crop from the dedicated physical synthetic lineage."""
    if class_index < 0 or class_index >= NUM_CLASSES:
        raise ValueError(f"class_index must be in [0, {NUM_CLASSES}), got {class_index}")

    theme = rng.choice(DATAGEN_THEMES).with_perturbation(rng)
    is_light_square = rng.random() < 0.5
    background_rgb = generate_square_texture(size, theme, is_light_square, rng)

    if class_index == 0:
        return cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR)

    is_white = class_index <= 6
    piece_type = PIECE_TYPES[(class_index - 1) % len(PIECE_TYPES)]
    material = select_random_material(rng).with_perturbation(rng)
    light_dir = np.array(
        [rng.uniform(-0.5, 0.5), rng.uniform(-0.7, -0.3), rng.uniform(0.5, 1.0)],
        dtype=np.float32,
    )
    light_dir = light_dir / np.linalg.norm(light_dir)

    if cache is not None:
        sprite = cache.get_or_render(piece_type, is_white, material, size, light_dir, rng)
    else:
        sprite = render_piece_sprite(piece_type, material, is_white, size, light_dir, rng)

    sprite_rgba = np.array(sprite.convert("RGBA"))
    alpha = sprite_rgba[:, :, 3:4].astype(np.float32) / 255.0
    sprite_rgb = sprite_rgba[:, :, :3].astype(np.float32)
    background_f = background_rgb.astype(np.float32)
    composited = (sprite_rgb * alpha + background_f * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)


def augment_physical_square_image(image_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply simple photometric augmentation for physical square crops."""
    image = image_bgr.astype(np.float32)

    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-18.0, 18.0)
    image = np.clip(image * alpha + beta, 0, 255)

    if rng.random() < 0.25:
        ksize = rng.choice([3, 5])
        image = cv2.GaussianBlur(image.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)

    if rng.random() < 0.25:
        quality = rng.randint(60, 92)
        _, encoded = cv2.imencode(
            ".jpg",
            image.astype(np.uint8),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            image = decoded.astype(np.float32)

    if rng.random() < 0.20:
        shadow = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(1.0, 1.1), image.shape[1])
        image *= shadow[None, :, None]

    return np.clip(image, 0, 255).astype(np.uint8)


def _sample_rows_per_class(
    rows: list[PhysicalEvalRow],
    *,
    max_per_class: int,
    seed: int,
) -> list[PhysicalEvalRow]:
    by_class: dict[int, list[PhysicalEvalRow]] = {
        class_index: [] for class_index in range(NUM_CLASSES)
    }
    for row in rows:
        by_class[row.label_index].append(row)

    rng = random.Random(seed)
    sampled: list[PhysicalEvalRow] = []
    for class_index in range(NUM_CLASSES):
        class_rows = list(by_class[class_index])
        rng.shuffle(class_rows)
        sampled.extend(class_rows[:max_per_class])
    return sampled
