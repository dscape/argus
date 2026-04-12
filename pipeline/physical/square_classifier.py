"""Runtime physical-board square classifier backed by a frozen DINO probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.square_data import (
    CLASS_NAMES,
    INPUT_SIZE,
    preprocess_square_image,
    split_rectified_board_into_squares,
)
from pipeline.physical.square_probe import PhysicalSquareLinearProbe, load_probe_checkpoint
from pipeline.shared import BoardObservation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
_CLASS_TO_SYMBOL = {index: name for index, name in enumerate(CLASS_NAMES)}

_cached_model: tuple[VisionEncoder, PhysicalSquareLinearProbe] | None = None
_cached_weights_path: Path | None = None


def read_board_observation_from_frame(
    board_crop: np.ndarray,
    *,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
) -> BoardObservation | None:
    """Read a rectified physical board crop into a source-agnostic observation."""
    try:
        encoder, probe = _get_runtime_model(device=device)
    except FileNotFoundError:
        return None

    square_crops = split_rectified_board_into_squares(board_crop)
    batch = torch.stack(
        [preprocess_square_image(crop, size=INPUT_SIZE) for crop in square_crops],
        dim=0,
    )
    batch = batch.to(next(probe.parameters()).device)
    encoder.eval()
    probe.eval()
    with torch.no_grad():
        embeddings = encoder.forward_pooled(batch)
        logits = probe(embeddings)
        probabilities = torch.softmax(logits, dim=1)
        class_ids = probabilities.argmax(dim=1).cpu().tolist()
        confidences = probabilities.max(dim=1).values.cpu().tolist()

    fen = _class_ids_to_board_fen(class_ids)
    return BoardObservation(
        fen=fen,
        square_confidences=tuple(float(value) for value in confidences),
        timestamp_seconds=timestamp_seconds,
        source="physical",
    )


def read_fen_from_frame(
    board_crop: np.ndarray,
    *,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
) -> str | None:
    """Return the physical-board FEN when runtime weights are available."""
    observation = read_board_observation_from_frame(
        board_crop,
        timestamp_seconds=timestamp_seconds,
        device=device,
    )
    return None if observation is None else observation.fen


def _class_ids_to_board_fen(class_ids: list[int]) -> str:
    if len(class_ids) != 64:
        raise ValueError(f"Expected 64 class ids, got {len(class_ids)}")

    ranks: list[str] = []
    for row in range(8):
        empty_run = 0
        rank_parts: list[str] = []
        for col in range(8):
            class_id = class_ids[row * 8 + col]
            symbol = _CLASS_TO_SYMBOL.get(class_id)
            if symbol is None:
                raise ValueError(f"Unknown class id: {class_id}")
            if symbol == "empty":
                empty_run += 1
                continue
            if empty_run > 0:
                rank_parts.append(str(empty_run))
                empty_run = 0
            rank_parts.append(symbol)
        if empty_run > 0:
            rank_parts.append(str(empty_run))
        ranks.append("".join(rank_parts) or "8")
    return "/".join(ranks)


def _get_runtime_model(*, device: str) -> tuple[VisionEncoder, PhysicalSquareLinearProbe]:
    global _cached_model, _cached_weights_path
    weights_path = _resolve_weights_path()
    if _cached_model is not None and _cached_weights_path == weights_path:
        return _cached_model

    checkpoint = load_probe_checkpoint(weights_path)
    model_name = str(checkpoint.get("model_name", "facebook/dinov2-base"))
    encoder = VisionEncoder(model_name=model_name, frozen=True).to(torch.device(device))
    probe = PhysicalSquareLinearProbe(encoder.embed_dim)
    probe.load_state_dict(checkpoint["state_dict"])
    probe.to(torch.device(device))

    _cached_model = (encoder, probe)
    _cached_weights_path = weights_path
    return _cached_model


def _resolve_weights_path() -> Path:
    if not _DEFAULT_WEIGHTS_PATH.exists():
        raise FileNotFoundError("No physical square-classifier weights found")
    return _DEFAULT_WEIGHTS_PATH


def load_metadata() -> dict[str, Any] | None:
    metadata_path = WEIGHTS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())
