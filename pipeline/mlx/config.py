"""Configuration for the MLX chess vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MLXPipelineConfig:
    """Configuration for the MLX chess analysis pipeline."""

    # VLM settings
    vlm_model: str = "mlx-community/gemma-4-26b-a4b-it-4bit"
    vlm_max_tokens: int = 2048
    enable_thinking: bool = False

    # Frame extraction — 2fps is the minimum for blitz (3+2) chess.
    # At 1fps, compound moves (two half-moves between frames) are common.
    fps: float = 2.0
    vlm_sample_count: int = 8  # Frames to sample for VLM scene analysis

    # Board segmentation (SAM 3)
    sam_model: str = "mlx-community/sam3-image"
    sam_confidence_threshold: float = 0.5

    # Piece detection (RF-DETR)
    rfdetr_confidence_threshold: float = 0.4
    use_piece_classifier: bool = True  # Use DINOv2 PieceClassifier as fallback

    # Move resolution
    stability_window: int = 1
    max_move_changed_squares: int = 4

    # Video annotation
    annotate: bool = True
    tts: bool = False
    font_scale: float = 1.0
    font_thickness: int = 2

    # Output
    output_dir: Path = field(default_factory=lambda: Path("output/mlx"))

    # Starting position (None = standard start)
    initial_fen: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
