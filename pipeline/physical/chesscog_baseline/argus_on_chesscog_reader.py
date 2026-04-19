"""Argus-backbone inference using chesscog warp + crop geometry (Study 2).

Uses our corner-bypass `warp_board_argus_corners` + chesscog's `crop_square`
to produce per-square crops, then forwards them through argus's
`SquareClassifier` (DINOv2 + linear head trained on chesscog-style PNGs).
"""

from __future__ import annotations

import chess
import cv2
import numpy as np
import torch

from pipeline.physical.chesscog_baseline import ensure_chesscog_on_path
from pipeline.physical.chesscog_baseline.dataset_export import (
    _square_from_argus_index,
    warp_board_argus_corners,
)
from pipeline.physical.two_stage.classifier_data import preprocess_square_crop
from pipeline.physical.two_stage.classifiers import (
    SquareClassifier,
    piece_label_to_square_class,
)
from pipeline.physical.two_stage.reader import TwoStageBoardReaderResult

ensure_chesscog_on_path()

from chesscog.occupancy_classifier.create_dataset import (  # noqa: E402
    crop_square as chesscog_crop_square_occupancy,
)
from chesscog.piece_classifier.create_dataset import (  # noqa: E402
    crop_square as chesscog_crop_square_piece,
)


def read_board_argus_on_chesscog_crops(
    frame_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    occupancy_model: SquareClassifier,
    piece_model: SquareClassifier,
    occupancy_input_size: int = 100,
    piece_input_size: int = 200,
    occupancy_threshold: float = 0.5,
    device: torch.device | str = "cpu",
) -> TwoStageBoardReaderResult:
    """Run argus DINOv2 + linear head classifiers on chesscog-style warped crops.

    Pipeline:
    1. Compute the two warped canvases (occ 500x500, piece 800x800) via
       argus's corner-order (board-coord order) → dst TL/TR/BR/BL mapping.
    2. For each square, crop from the warped canvas using chesscog's crop_square.
    3. Feed through argus preprocess (BGR→RGB, resize, ImageNet normalize).
    4. Argus SquareClassifier → logits → softmax.

    occupancy_input_size/piece_input_size are the sizes argus's classifier was
    trained on. preprocess_square_crop always resizes to a square; for piece
    it resizes to size×size (i.e. 200x200 from a 200x100 chesscog crop). This
    is fine because argus's DINOv2 is input-size-robust.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    warped_occ = warp_board_argus_corners(rgb, corners, canvas="occupancy")
    warped_piece = warp_board_argus_corners(rgb, corners, canvas="piece")

    def _to_bgr(arr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    occ_crops_bgr = [
        _to_bgr(
            chesscog_crop_square_occupancy(
                warped_occ, _square_from_argus_index(i), chess.WHITE
            )
        )
        for i in range(64)
    ]
    occ_tensor = torch.stack(
        [preprocess_square_crop(c, size=occupancy_input_size) for c in occ_crops_bgr],
        dim=0,
    ).to(device)

    occupancy_model.eval()
    piece_model.eval()
    with torch.no_grad():
        occ_logits = occupancy_model(occ_tensor)
        occ_probs = torch.softmax(occ_logits, dim=-1)
        occupied_mask = occ_probs[:, 1] >= occupancy_threshold

        piece_probs_all = torch.zeros(
            (64, piece_model.config.num_classes), device=device
        )
        class_ids = [0] * 64
        occupied_indices = (
            torch.nonzero(occupied_mask, as_tuple=False).flatten().tolist()
        )
        if occupied_indices:
            piece_crops_bgr = [
                _to_bgr(
                    chesscog_crop_square_piece(
                        warped_piece, _square_from_argus_index(i), chess.WHITE
                    )
                )
                for i in occupied_indices
            ]
            piece_tensor = torch.stack(
                [preprocess_square_crop(c, size=piece_input_size) for c in piece_crops_bgr],
                dim=0,
            ).to(device)
            piece_logits = piece_model(piece_tensor)
            piece_softmax = torch.softmax(piece_logits, dim=-1)
            for local_idx, board_idx in enumerate(occupied_indices):
                piece_label = int(piece_softmax[local_idx].argmax().item())
                class_ids[board_idx] = piece_label_to_square_class(piece_label)
                piece_probs_all[board_idx] = piece_softmax[local_idx]

    return TwoStageBoardReaderResult(
        class_ids=tuple(int(v) for v in class_ids),
        occupancy_probs=tuple(float(v) for v in occ_probs[:, 1].tolist()),
        piece_probs=tuple(
            tuple(float(v) for v in row.tolist()) for row in piece_probs_all
        ),
    )
