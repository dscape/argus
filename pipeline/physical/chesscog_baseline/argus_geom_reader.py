"""Chesscog backbones (ResNet18/InceptionV3) × argus-projection crops (Study 1).

Uses argus's `extract_projected_*_crop` to build per-square crops, then forwards
through chesscog-style classifiers (whole-module pickles loaded via
`load_chesscog_classifier`).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline.physical.chesscog_baseline import build_chesscog_to_argus_remap
from pipeline.physical.chesscog_baseline.reader import ChessCogClassifier
from pipeline.physical.piece_projection import (
    extract_projected_occupancy_crop,
    extract_projected_piece_crop,
)
from pipeline.physical.two_stage.reader import TwoStageBoardReaderResult


def _apply(transform, crops_rgb: list[np.ndarray]) -> torch.Tensor:
    return torch.stack(
        [transform(Image.fromarray(c, mode="RGB")) for c in crops_rgb], dim=0
    )


def read_board_chesscog_on_argus_crops(
    frame_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    occupancy: ChessCogClassifier,
    piece: ChessCogClassifier,
    occupancy_size: int = 100,
    piece_size: int = 200,
    occupancy_threshold: float = 0.5,
    occupancy_pad_ratio: float = 0.3,
) -> TwoStageBoardReaderResult:
    """Run chesscog models on argus-projection crops and return argus-shaped result."""
    # Chesscog models were trained on RGB. Argus crops come out BGR; convert per crop.
    occ_crops_rgb = [
        cv2.cvtColor(
            extract_projected_occupancy_crop(
                frame_bgr,
                corners,
                row=i // 8,
                col=i % 8,
                output_size=occupancy_size,
                pad_ratio=occupancy_pad_ratio,
            ),
            cv2.COLOR_BGR2RGB,
        )
        for i in range(64)
    ]
    occ_tensor = _apply(occupancy.transform, occ_crops_rgb).to(occupancy.device)

    occupied_idx = occupancy.classes.index("occupied")

    with torch.no_grad():
        occ_logits = occupancy.model(occ_tensor)
        occ_probs = torch.softmax(occ_logits, dim=-1)
        occ_prob_np = occ_probs[:, occupied_idx].detach().cpu().numpy()
        occupied_mask = occ_prob_np >= occupancy_threshold

        class_ids = [0] * 64
        piece_probs_all = np.zeros((64, 12), dtype=np.float32)
        occupied_indices = [i for i, flag in enumerate(occupied_mask) if flag]
        if occupied_indices:
            piece_crops_rgb = [
                cv2.cvtColor(
                    extract_projected_piece_crop(
                        frame_bgr,
                        corners,
                        row=i // 8,
                        col=i % 8,
                        output_size=piece_size,
                        flip_left_half=True,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                for i in occupied_indices
            ]
            piece_tensor = _apply(piece.transform, piece_crops_rgb).to(piece.device)
            piece_logits = piece.model(piece_tensor)
            piece_softmax = torch.softmax(piece_logits, dim=-1).detach().cpu().numpy()

            chesscog_to_argus = build_chesscog_to_argus_remap(piece.classes)
            for local_idx, board_idx in enumerate(occupied_indices):
                chesscog_class = int(piece_softmax[local_idx].argmax())
                class_ids[board_idx] = chesscog_to_argus[chesscog_class]
                for ccl_idx, argus_class_id in enumerate(chesscog_to_argus):
                    piece_probs_all[board_idx, argus_class_id - 1] = float(
                        piece_softmax[local_idx, ccl_idx]
                    )

    return TwoStageBoardReaderResult(
        class_ids=tuple(class_ids),
        occupancy_probs=tuple(float(p) for p in occ_prob_np),
        piece_probs=tuple(
            tuple(float(v) for v in row) for row in piece_probs_all
        ),
    )
