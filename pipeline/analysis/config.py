"""Configuration for local video analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_VLM_MODEL = "gemma4_local"

SceneBackend = Literal["none", "vlm"]
ReaderBackend = Literal["overlay", "hybrid"]


@dataclass
class VideoAnalysisConfig:
    """Configuration for the shared local video analysis pipeline."""

    fps: float = 2.0
    device: str = "mps"
    reader_backend: ReaderBackend = "overlay"
    scene_backend: SceneBackend = "none"
    stability_window: int = 1

    annotate: bool = True
    tts: bool = False
    font_scale: float = 1.0
    font_thickness: int = 2

    output_dir: Path = field(default_factory=lambda: Path("outputs/analysis"))

    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_max_tokens: int = 2048
    vlm_sample_count: int = 8

    sam_confidence_threshold: float = 0.5
    use_piece_classifier: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
