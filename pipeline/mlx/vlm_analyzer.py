"""Gemma 4 VLM scene understanding for chess videos.

Uses mlx-vlm to run Gemma 4 on Apple Silicon for:
- Scene description (OTB vs online, overlay detection)
- Board position reading
- Player identification
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.mlx.prompts import BOARD_READING, SCENE_ANALYSIS

logger = logging.getLogger(__name__)

# Lazy-loaded model cache (singleton pattern from piece_classifier.py)
_model: Any = None
_processor: Any = None
_loaded_model_name: str | None = None


@dataclass
class GameContext:
    """Scene understanding result from VLM analysis."""

    description: str = ""
    scene_type: str = "unknown"  # "otb", "online", "broadcast"
    has_overlay: bool = False
    board_location: str = ""
    players: dict[str, str] = field(default_factory=lambda: {"white": "?", "black": "?"})
    game_phase: str = "unknown"
    time_control: str = "unknown"
    additional_notes: str = ""
    raw_responses: list[str] = field(default_factory=list)


def _load_model(model_name: str) -> None:
    """Lazy-load the VLM model and processor."""
    global _model, _processor, _loaded_model_name

    if _loaded_model_name == model_name:
        return

    from mlx_vlm import load

    logger.info("Loading VLM model: %s", model_name)
    _model, _processor = load(model_name)
    _loaded_model_name = model_name
    logger.info("VLM model loaded")


def _generate(
    prompt: str,
    images: list[Image.Image],
    config: VideoAnalysisConfig,
) -> str:
    """Run VLM generation on images with a prompt."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    _load_model(config.vlm_model)

    # Save images to temp files for mlx_vlm (expects file paths)
    import tempfile

    image_paths: list[str] = []
    for i, img in enumerate(images):
        path = tempfile.mktemp(suffix=f"_{i}.png")
        img.save(path)
        image_paths.append(path)

    formatted_prompt = apply_chat_template(
        _processor,
        _model.config,
        prompt,
        num_images=len(image_paths),
    )

    result = generate(
        model=_model,
        processor=_processor,
        prompt=formatted_prompt,
        image=image_paths if image_paths else None,
        max_tokens=config.vlm_max_tokens,
        verbose=False,
    )

    # Clean up temp files
    import os

    for p in image_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    # mlx_vlm.generate returns a GenerationResult object with a .text attribute
    return result.text


def _parse_scene_json(text: str) -> dict[str, Any]:
    """Extract JSON from VLM response, handling markdown code blocks."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (``` markers)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def analyze_scene(
    frames: list[np.ndarray],
    config: VideoAnalysisConfig,
) -> GameContext:
    """Analyze sampled video frames to understand the chess scene.

    Args:
        frames: List of (H, W, 3) RGB uint8 frames.
        config: Pipeline configuration.

    Returns:
        GameContext with scene understanding results.
    """
    if not frames:
        return GameContext()

    # Use the middle frame for primary analysis
    mid = len(frames) // 2
    images = [Image.fromarray(frames[mid])]

    logger.info("Running VLM scene analysis on %d frame(s)", len(images))

    response = _generate(SCENE_ANALYSIS, images, config)
    parsed = _parse_scene_json(response)

    context = GameContext(
        description=response,
        scene_type=parsed.get("scene_type", "unknown"),
        has_overlay=parsed.get("has_overlay", False),
        board_location=parsed.get("board_location", ""),
        players=parsed.get("players", {"white": "?", "black": "?"}),
        game_phase=parsed.get("game_phase", "unknown"),
        time_control=parsed.get("time_control", "unknown"),
        additional_notes=parsed.get("additional_notes", ""),
        raw_responses=[response],
    )

    logger.info(
        "Scene: type=%s, overlay=%s, phase=%s, players=%s",
        context.scene_type,
        context.has_overlay,
        context.game_phase,
        context.players,
    )

    return context


def read_board_position(
    frame: np.ndarray,
    config: VideoAnalysisConfig,
) -> str | None:
    """Use VLM to read the board position and return a FEN string.

    This is a fallback when grid detection + piece classifier fails
    (e.g., on 3D/OTB boards where the overlay classifier doesn't work).

    Args:
        frame: (H, W, 3) RGB uint8 image of the board region.
        config: Pipeline configuration.

    Returns:
        FEN piece placement string, or None if parsing failed.
    """
    image = Image.fromarray(frame)
    response = _generate(BOARD_READING, [image], config)

    # Extract just the FEN string (strip whitespace and extra text)
    fen = response.strip().split("\n")[0].strip()

    # Validate basic FEN structure: should have 8 ranks separated by /
    parts = fen.split("/")
    if len(parts) != 8:
        logger.warning("VLM returned invalid FEN: %s", fen)
        return None

    return fen


def unload_model() -> None:
    """Free VLM model memory."""
    global _model, _processor, _loaded_model_name
    _model = None
    _processor = None
    _loaded_model_name = None
    logger.info("VLM model unloaded")
