"""Mixed-stack reader: chesscog ResNet18 occupancy + argus DINOv2 piece.

Phase A of the Apr-19+ follow-up plan. Study 1 showed ResNet18 on argus
projection crops hits 86.2% occupancy classifier-val (vs DINOv2's 83.3%).
But Study 1 used ResNet18 for BOTH stages and the piece stage collapsed
(47% val), which drowned any occupancy gain. This reader mixes the two:
- occupancy: chesscog ResNet18 on argus-projected crops
- piece: argus DINOv2 (the existing `piece_corrected` weights) on argus-projected crops

Both stages consume the SAME argus-projection crops, so geometry is unchanged
vs the baseline reader. The only difference is which backbone classifies
occupancy.

Result shape matches `TwoStageBoardReaderResult` from two_stage/reader.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline.physical.chesscog_baseline.reader import ChessCogClassifier
from pipeline.physical.piece_projection import (
    extract_projected_occupancy_crop,
    extract_projected_piece_crop,
)
from pipeline.physical.two_stage.classifier_data import preprocess_square_crop
from pipeline.physical.two_stage.classifiers import (
    SquareClassifier,
    piece_label_to_square_class,
)
from pipeline.physical.two_stage.reader import TwoStageBoardReaderResult


def _apply_chesscog(transform, crops_rgb: list[np.ndarray]) -> torch.Tensor:
    return torch.stack(
        [transform(Image.fromarray(c, mode="RGB")) for c in crops_rgb], dim=0
    )


def read_board_mixed_stack(
    frame_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    occupancy_chesscog: ChessCogClassifier,
    piece_argus: SquareClassifier,
    occupancy_crop_size: int = 100,
    piece_crop_size: int = 224,
    occupancy_threshold: float = 0.7,
    occupancy_pad_ratio: float = 0.3,
) -> TwoStageBoardReaderResult:
    """Mixed-stack two-stage reader.

    Occupancy via chesscog ResNet18 (trained on argus-projection PNGs),
    piece via argus DINOv2 (frozen, linear head). Both operate on argus
    `extract_projected_*_crop` output.
    """
    # --- Occupancy stage (chesscog) ---
    occ_crops_rgb = [
        cv2.cvtColor(
            extract_projected_occupancy_crop(
                frame_bgr,
                corners,
                row=i // 8,
                col=i % 8,
                output_size=occupancy_crop_size,
                pad_ratio=occupancy_pad_ratio,
            ),
            cv2.COLOR_BGR2RGB,
        )
        for i in range(64)
    ]
    occ_tensor = _apply_chesscog(occupancy_chesscog.transform, occ_crops_rgb).to(
        occupancy_chesscog.device
    )
    occupied_idx = occupancy_chesscog.classes.index("occupied")

    with torch.no_grad():
        occ_logits = occupancy_chesscog.model(occ_tensor)
        occ_probs = torch.softmax(occ_logits, dim=-1)
        occ_prob_np = occ_probs[:, occupied_idx].detach().cpu().numpy()
        occupied_mask = occ_prob_np >= occupancy_threshold

        # --- Piece stage (argus DINOv2) ---
        piece_model_was_training = piece_argus.training
        piece_argus.eval()

        piece_probs_all = torch.zeros(
            (64, piece_argus.config.num_classes), device=next(piece_argus.parameters()).device
        )
        class_ids = [0] * 64
        occupied_indices = [i for i, flag in enumerate(occupied_mask) if flag]
        if occupied_indices:
            piece_crops_bgr = [
                extract_projected_piece_crop(
                    frame_bgr,
                    corners,
                    row=idx // 8,
                    col=idx % 8,
                    output_size=piece_crop_size,
                )
                for idx in occupied_indices
            ]
            piece_device = next(piece_argus.parameters()).device
            piece_tensor = torch.stack(
                [
                    preprocess_square_crop(c, size=piece_crop_size)
                    for c in piece_crops_bgr
                ],
                dim=0,
            ).to(piece_device)
            piece_logits = piece_argus(piece_tensor)
            piece_softmax = torch.softmax(piece_logits, dim=-1)
            for local_idx, square_index in enumerate(occupied_indices):
                piece_label = int(piece_softmax[local_idx].argmax().item())
                class_ids[square_index] = piece_label_to_square_class(piece_label)
                piece_probs_all[square_index] = piece_softmax[local_idx]

        if piece_model_was_training:
            piece_argus.train()

    return TwoStageBoardReaderResult(
        class_ids=tuple(int(v) for v in class_ids),
        occupancy_probs=tuple(float(p) for p in occ_prob_np),
        piece_probs=tuple(
            tuple(float(v) for v in row.tolist()) for row in piece_probs_all
        ),
    )
