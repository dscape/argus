"""Physical-board per-square classifier interface.

This module is intentionally separate from ``pipeline.overlay``. The overlay
classifier must never be reused on real-board camera crops.
"""

from __future__ import annotations

import numpy as np

from pipeline.shared import BoardObservation


def read_board_observation_from_frame(
    _board_crop: np.ndarray,
    *,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
) -> BoardObservation | None:
    """TODO: train a dedicated physical-board square classifier."""
    del timestamp_seconds, device
    return None


def read_fen_from_frame(
    board_crop: np.ndarray,
    *,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
) -> str | None:
    """Return the physical-board FEN when the dedicated classifier exists."""
    observation = read_board_observation_from_frame(
        board_crop,
        timestamp_seconds=timestamp_seconds,
        device=device,
    )
    return None if observation is None else observation.fen
