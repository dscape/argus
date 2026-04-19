"""Adapter that runs chesscog-style classifiers as an argus two-stage reader.

Returns a `TwoStageBoardReaderResult` shaped identically to
`pipeline.physical.two_stage.reader.read_board`, so a single comparison script
can run both systems and tabulate results.

Chesscog's classifier weights are loaded as whole-module pickles (produced by
`torch.save(model, ...)` in chesscog's training loop). We load them with
`weights_only=False` because that's what they require.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from PIL import Image

from pipeline.physical.chesscog_baseline import (
    build_chesscog_to_argus_remap,
    ensure_chesscog_on_path,
)
from pipeline.physical.chesscog_baseline.dataset_export import (
    _square_from_argus_index,
    warp_board_argus_corners,
)
from pipeline.physical.two_stage.reader import TwoStageBoardReaderResult

ensure_chesscog_on_path()

from recap import CfgNode as CN  # noqa: E402

from chesscog.core.dataset import Datasets, build_transforms  # noqa: E402
from chesscog.occupancy_classifier.create_dataset import (  # noqa: E402
    crop_square as chesscog_crop_square_occupancy,
)
from chesscog.piece_classifier.create_dataset import (  # noqa: E402
    crop_square as chesscog_crop_square_piece,
)


@dataclass
class ChessCogClassifier:
    """A loaded chesscog model + its config + precomputed transform."""
    model: torch.nn.Module
    cfg: CN
    transform: object  # torchvision.transforms.Compose
    classes: list[str]
    device: torch.device


def load_chesscog_classifier(
    checkpoint_path: Path,
    cfg_path: Path,
    device: torch.device,
) -> ChessCogClassifier:
    """Load a chesscog whole-model pickle and its YAML config.

    chesscog saves via `torch.save(model, ...)` so `weights_only=False` is
    required on torch >= 2.2.
    """
    cfg = CN.load_yaml_with_base(str(cfg_path))
    model = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    model.to(device).eval()
    transform = build_transforms(cfg, Datasets.TEST)
    classes = list(cfg.DATASET.CLASSES)
    return ChessCogClassifier(
        model=model,
        cfg=cfg,
        transform=transform,
        classes=classes,
        device=device,
    )


def _apply_transform(
    crops: list[np.ndarray], transform
) -> torch.Tensor:
    pil_crops = [Image.fromarray(c, mode="RGB") for c in crops]
    return torch.stack([transform(p) for p in pil_crops], dim=0)


def read_board_chesscog(
    frame_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    occupancy: ChessCogClassifier,
    piece: ChessCogClassifier,
    occupancy_threshold: float = 0.5,
) -> TwoStageBoardReaderResult:
    """Run chesscog-style occupancy + piece classification on one frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    warped_occ = warp_board_argus_corners(rgb, corners, canvas="occupancy")
    warped_piece = warp_board_argus_corners(rgb, corners, canvas="piece")

    occ_crops = [
        chesscog_crop_square_occupancy(
            warped_occ, _square_from_argus_index(i), chess.WHITE
        )
        for i in range(64)
    ]
    occ_tensor = _apply_transform(occ_crops, occupancy.transform).to(occupancy.device)

    # ImageFolder sorts class names alphabetically: ["empty", "occupied"].
    # "occupied" index = 1.
    occupied_idx = occupancy.classes.index("occupied")

    with torch.no_grad():
        occ_logits = occupancy.model(occ_tensor)
        occ_probs = torch.softmax(occ_logits, dim=-1)
        occupancy_probs_np = occ_probs[:, occupied_idx].detach().cpu().numpy()
        occupied_mask = occupancy_probs_np >= occupancy_threshold

        class_ids = [0] * 64
        piece_probs_all = np.zeros((64, 12), dtype=np.float32)

        occupied_indices = [i for i, flag in enumerate(occupied_mask) if flag]
        if occupied_indices:
            piece_crops = [
                chesscog_crop_square_piece(
                    warped_piece, _square_from_argus_index(i), chess.WHITE
                )
                for i in occupied_indices
            ]
            piece_tensor = _apply_transform(piece_crops, piece.transform).to(piece.device)
            piece_logits = piece.model(piece_tensor)
            piece_probs = torch.softmax(piece_logits, dim=-1).detach().cpu().numpy()

            chesscog_to_argus = build_chesscog_to_argus_remap(piece.classes)
            for local_idx, board_idx in enumerate(occupied_indices):
                chesscog_class = int(piece_probs[local_idx].argmax())
                class_ids[board_idx] = chesscog_to_argus[chesscog_class]
                # Rearrange piece_probs into argus ordering [P, N, B, R, Q, K, p, n, b, r, q, k]
                for ccl_idx, argus_class_id in enumerate(chesscog_to_argus):
                    piece_probs_all[board_idx, argus_class_id - 1] = float(
                        piece_probs[local_idx, ccl_idx]
                    )

    return TwoStageBoardReaderResult(
        class_ids=tuple(class_ids),
        occupancy_probs=tuple(float(p) for p in occupancy_probs_np),
        piece_probs=tuple(tuple(float(v) for v in row) for row in piece_probs_all),
    )
