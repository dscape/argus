"""Frame-type classification for physical-board broadcasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FrameType = Literal["board_view", "overlay", "player_closeup", "crowd", "other"]


@dataclass(frozen=True)
class FrameClassification:
    """Predicted coarse frame type for a broadcast frame."""

    label: FrameType
    confidence: float


def classify_frame(_frame: np.ndarray, *, device: str = "cpu") -> FrameClassification | None:
    """TODO: train a dedicated physical-board frame classifier."""
    del device
    return None
