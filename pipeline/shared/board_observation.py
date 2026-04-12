"""Source-agnostic board observations consumed by temporal fusion."""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_SQUARE_CONFIDENCES = (1.0,) * 64


@dataclass(frozen=True, slots=True)
class BoardObservation:
    """One timestamped board-state observation from any vision frontend."""

    fen: str
    square_confidences: tuple[float, ...] = _DEFAULT_SQUARE_CONFIDENCES
    timestamp_seconds: float = 0.0
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.fen:
            raise ValueError("fen must be non-empty")
        if len(self.square_confidences) != 64:
            raise ValueError(
                f"square_confidences must contain 64 values, got {len(self.square_confidences)}"
            )
        if any(confidence < 0.0 or confidence > 1.0 for confidence in self.square_confidences):
            raise ValueError("square_confidences must be between 0.0 and 1.0")
