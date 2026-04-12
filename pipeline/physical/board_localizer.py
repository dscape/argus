"""Physical-board localization for real broadcast frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoardLocalization:
    """Physical-board location in one frame."""

    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    confidence: float


def localize_board(_frame: np.ndarray, *, device: str = "cpu") -> BoardLocalization | None:
    """TODO: train a dedicated physical-board localizer and return board corners."""
    del device
    return None
