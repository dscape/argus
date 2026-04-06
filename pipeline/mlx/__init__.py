"""MLX-based chess video analysis pipeline.

Experimental module using Apple Silicon MLX models to analyze chess footage:
- Gemma 4 VLM for scene understanding
- SAM 3 for board segmentation
- RF-DETR for piece detection
- FEN-diffing for move resolution
"""

from pipeline.mlx.config import MLXPipelineConfig
from pipeline.mlx.pipeline import MLXChessPipeline

__all__ = ["MLXChessPipeline", "MLXPipelineConfig"]
