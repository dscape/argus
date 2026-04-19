"""PyTorch Dataset loading chesscog-style PNG ImageFolder trees for argus training.

Used by Study 2 (argus DINOv2 head on chesscog warp+crop PNGs). Mirrors the
`OccupancySquareDataset` / `PieceSquareDataset` surface so `train_square_classifier.py`
can swap it in.

Occupancy folder layout:
    root/{train|val}/{empty|occupied}/*.png
Piece folder layout:
    root/{train|val}/{black_bishop|...|white_rook}/*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pipeline.physical.chesscog_baseline import CHESSCOG_FOLDER_TO_ARGUS_CLASS
from pipeline.physical.two_stage.classifier_data import preprocess_square_crop
from pipeline.physical.two_stage.classifiers import (
    square_class_to_occupancy_label,
    square_class_to_piece_label,
)


@dataclass(frozen=True)
class _PngSample:
    path: Path
    square_class: int  # 0..12 in argus SQUARE_CLASS_NAMES


class _PngSquareDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        samples: list[_PngSample],
        input_size: int | tuple[int, int],
        augment: bool,
    ) -> None:
        self.samples = samples
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _load_bgr(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8)
        # preprocess_square_crop expects BGR, so convert
        return rgb[:, :, ::-1].copy()


class OccupancyPngDataset(_PngSquareDataset):
    def __init__(self, *, root: Path, split: str, input_size: int = 100, augment: bool = False):
        base = root / split
        samples: list[_PngSample] = []
        class_dirs = {
            "empty": 0,
            "occupied": 1,  # placeholder; actual class is non-zero
        }
        for folder_name, occupancy_label in class_dirs.items():
            d = base / folder_name
            if not d.exists():
                continue
            for png in sorted(d.glob("*.png")):
                # square_class 0 = empty; for occupied, assign 1 (any non-zero piece class)
                cls = 0 if occupancy_label == 0 else 1
                samples.append(_PngSample(path=png, square_class=cls))
        if not samples:
            raise RuntimeError(f"no PNGs found under {base}")
        super().__init__(samples=samples, input_size=input_size, augment=augment)
        self.indices = samples  # compat w/ train script's _class_weights

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[index]
        crop_bgr = self._load_bgr(s.path)
        size = self.input_size if isinstance(self.input_size, int) else self.input_size[0]
        image = preprocess_square_crop(crop_bgr, size=size, augment=self.augment)
        label = torch.tensor(
            square_class_to_occupancy_label(s.square_class),
            dtype=torch.long,
        )
        return image, label


class PiecePngDataset(_PngSquareDataset):
    def __init__(self, *, root: Path, split: str, input_size: int = 200, augment: bool = False):
        base = root / split
        samples: list[_PngSample] = []
        for folder_name, argus_cls in CHESSCOG_FOLDER_TO_ARGUS_CLASS.items():
            d = base / folder_name
            if not d.exists():
                continue
            for png in sorted(d.glob("*.png")):
                samples.append(_PngSample(path=png, square_class=argus_cls))
        if not samples:
            raise RuntimeError(f"no PNGs found under {base}")
        super().__init__(samples=samples, input_size=input_size, augment=augment)
        self.indices = samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[index]
        crop_bgr = self._load_bgr(s.path)
        size = self.input_size if isinstance(self.input_size, int) else self.input_size[0]
        image = preprocess_square_crop(crop_bgr, size=size, augment=self.augment)
        label = torch.tensor(
            square_class_to_piece_label(s.square_class),
            dtype=torch.long,
        )
        return image, label


__all__ = ["OccupancyPngDataset", "PiecePngDataset"]
