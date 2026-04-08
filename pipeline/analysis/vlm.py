"""Local vision-language helpers for video analysis."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from pipeline.analysis.config import DEFAULT_VLM_MODEL, VideoAnalysisConfig
from pipeline.analysis.prompts import BOARD_READING, SCENE_ANALYSIS

logger = logging.getLogger(__name__)

_MODEL_ALIASES = {
    DEFAULT_VLM_MODEL: "mlx-community/gemma-4-26b-a4b-it-4bit",
}

_model: Any = None
_processor: Any = None
_loaded_model_name: str | None = None


@dataclass
class GameContext:
    """Scene understanding result from video analysis."""

    description: str = ""
    scene_type: str = "unknown"
    has_overlay: bool = False
    board_location: str = ""
    players: dict[str, str] = field(default_factory=lambda: {"white": "?", "black": "?"})
    game_phase: str = "unknown"
    time_control: str = "unknown"
    additional_notes: str = ""
    raw_responses: list[str] = field(default_factory=list)


def _resolve_model_name(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name, model_name)


def _load_model(model_name: str) -> None:
    """Lazy-load the VLM model and processor."""
    global _model, _processor, _loaded_model_name

    resolved_model_name = _resolve_model_name(model_name)
    if _loaded_model_name == resolved_model_name:
        return

    try:
        from mlx_vlm import load
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "VLM analysis requires optional local vision-language dependencies."
        ) from exc

    logger.info("Loading VLM model: %s", resolved_model_name)
    _model, _processor = load(resolved_model_name)
    _loaded_model_name = resolved_model_name
    logger.info("VLM model loaded")


def _generate(prompt: str, images: list[Image.Image], config: VideoAnalysisConfig) -> str:
    """Run VLM generation on images with a prompt."""
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "VLM analysis requires optional local vision-language dependencies."
        ) from exc

    _load_model(config.vlm_model)

    image_paths: list[str] = []
    try:
        for index, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=f"_{index}.png", delete=False) as handle:
                image.save(handle.name)
                image_paths.append(handle.name)

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
        return result.text
    finally:
        for path in image_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def _parse_scene_json(text: str) -> dict[str, Any]:
    """Extract JSON from a model response, handling markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.split("\n") if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def analyze_scene(frames: list[np.ndarray], config: VideoAnalysisConfig) -> GameContext:
    """Analyze sampled video frames to understand the scene."""
    if not frames:
        return GameContext()

    image = Image.fromarray(frames[len(frames) // 2])
    logger.info("Running scene analysis on %d frame(s)", 1)

    response = _generate(SCENE_ANALYSIS, [image], config)
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


def read_board_position(frame: np.ndarray, config: VideoAnalysisConfig) -> str | None:
    """Read the board position from an RGB board crop."""
    response = _generate(BOARD_READING, [Image.fromarray(frame)], config)
    fen = response.strip().split("\n")[0].strip()
    if len(fen.split("/")) != 8:
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
