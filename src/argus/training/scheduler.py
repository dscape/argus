"""Learning rate scheduling and curriculum pacing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """A stage in the curriculum training schedule."""

    max_boards: int
    max_occlusion: float
    min_resolution: int
    epochs: int


class CurriculumScheduler:
    """Manages curriculum learning progression.

    Gradually increases difficulty across training:
    - Number of boards
    - Occlusion severity
    - Resolution degradation
    """

    def __init__(self, stages: list[CurriculumStage]) -> None:
        self.stages = stages
        self._cumulative_epochs = []
        total = 0
        for stage in stages:
            total += stage.epochs
            self._cumulative_epochs.append(total)

    def get_stage(self, epoch: int) -> CurriculumStage:
        """Get the curriculum stage for the given epoch."""
        for i, cum_epoch in enumerate(self._cumulative_epochs):
            if epoch <= cum_epoch:
                return self.stages[i]
        return self.stages[-1]

    def get_difficulty_params(self, epoch: int) -> dict[str, float | int]:
        """Get data generation parameters for the current difficulty level."""
        stage = self.get_stage(epoch)
        return {
            "max_boards": stage.max_boards,
            "max_occlusion": stage.max_occlusion,
            "min_resolution": stage.min_resolution,
        }

    @property
    def total_epochs(self) -> int:
        return self._cumulative_epochs[-1] if self._cumulative_epochs else 0
