"""Two-stage board reader: occupancy classifier gates a piece classifier.

Given one oblique frame and its board corners, this module:

1. Extracts 64 symmetric occupancy crops (``extract_all_occupancy_crops``).
2. Runs the occupancy model to decide empty vs occupied per square.
3. For each occupied square, extracts an asymmetric piece crop and runs the
   piece classifier to pick one of 12 piece classes.
4. Composes a ``(64,)`` class-id array in the ``SQUARE_CLASS_NAMES`` space
   (0 = empty, 1..12 = piece classes).

The classifier forward pass runs on batches to amortize encoder cost. Squares
predicted empty by the occupancy stage are not passed to the piece classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pipeline.physical.square_classifier_data import preprocess_square_crop
from pipeline.physical.square_classifiers import (
    SquareClassifier,
    piece_label_to_square_class,
)
from pipeline.physical.square_crop import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    extract_all_occupancy_crops,
    extract_all_piece_crops,
)

_EMPTY_CLASS_ID = 0


@dataclass(frozen=True)
class TwoStageBoardReaderResult:
    class_ids: tuple[int, ...]  # 64 values in [0, 12]
    occupancy_probs: tuple[float, ...]  # P(occupied) per square, shape 64
    piece_probs: tuple[tuple[float, ...], ...]  # 64 × 12 softmax probs (zeros for empty)


def read_board(
    frame_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    occupancy_model: SquareClassifier,
    piece_model: SquareClassifier,
    occupancy_crop_size: int = DEFAULT_OCCUPANCY_CROP_SIZE,
    piece_crop_size: int = DEFAULT_PIECE_CROP_SIZE,
    device: torch.device | str = "cpu",
) -> TwoStageBoardReaderResult:
    """Run the two-stage reader on one frame and return per-square class ids."""
    occupancy_crops = extract_all_occupancy_crops(
        frame_bgr, corners, output_size=occupancy_crop_size
    )
    occupancy_tensor = torch.stack(
        [preprocess_square_crop(crop, size=occupancy_crop_size) for crop in occupancy_crops],
        dim=0,
    ).to(device)

    occupancy_model_was_training = occupancy_model.training
    piece_model_was_training = piece_model.training
    occupancy_model.eval()
    piece_model.eval()
    try:
        with torch.no_grad():
            occupancy_logits = occupancy_model(occupancy_tensor)
            occupancy_probs = torch.softmax(occupancy_logits, dim=-1)
            occupied_mask = occupancy_probs[:, 1] >= 0.5

            piece_probs_all = torch.zeros((64, piece_model.config.num_classes), device=device)
            class_ids = [_EMPTY_CLASS_ID] * 64
            occupied_indices = torch.nonzero(occupied_mask, as_tuple=False).flatten().tolist()
            if occupied_indices:
                piece_crops = extract_all_piece_crops(
                    frame_bgr, corners, output_size=piece_crop_size
                )
                piece_tensor = torch.stack(
                    [
                        preprocess_square_crop(piece_crops[idx], size=piece_crop_size)
                        for idx in occupied_indices
                    ],
                    dim=0,
                ).to(device)
                piece_logits = piece_model(piece_tensor)
                piece_softmax = torch.softmax(piece_logits, dim=-1)
                for local_index, square_index in enumerate(occupied_indices):
                    piece_label = int(piece_softmax[local_index].argmax().item())
                    class_ids[square_index] = piece_label_to_square_class(piece_label)
                    piece_probs_all[square_index] = piece_softmax[local_index]
    finally:
        if occupancy_model_was_training:
            occupancy_model.train()
        if piece_model_was_training:
            piece_model.train()

    return TwoStageBoardReaderResult(
        class_ids=tuple(int(value) for value in class_ids),
        occupancy_probs=tuple(float(value) for value in occupancy_probs[:, 1].tolist()),
        piece_probs=tuple(
            tuple(float(value) for value in row.tolist()) for row in piece_probs_all
        ),
    )
