"""Runtime physical-board reader backed by a frozen feature probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.board_data import preprocess_board_image
from pipeline.physical.board_probe import (
    PhysicalBoardStateEnsembleProbe,
    build_board_state_probe,
    board_probe_config_from_checkpoint,
    dino_patches_to_square_tokens,
)
from pipeline.physical.square_data import (
    CLASS_NAMES,
    INPUT_SIZE,
    preprocess_square_image,
    split_rectified_board_into_squares,
)
from pipeline.physical.square_probe import PhysicalSquareLinearProbe, load_probe_checkpoint
from pipeline.shared import (
    BoardLogitsExponentialSmoother,
    BoardObservation,
    constrained_board_class_ids,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
_DEFAULT_TEMPORAL_EMA_ALPHA = 0.1
_CLASS_TO_SYMBOL = {index: name for index, name in enumerate(CLASS_NAMES)}

_cached_model: tuple[dict[str, Any], VisionEncoder, nn.Module] | None = None
_cached_weights_path: Path | None = None
_cached_device: str | None = None


def read_board_observation_from_frame(
    board_crop: np.ndarray,
    *,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
) -> BoardObservation | None:
    """Read a rectified physical board crop into a source-agnostic observation."""
    try:
        checkpoint, encoder, probe = _get_runtime_model(device=device)
    except FileNotFoundError:
        return None

    logits = _predict_board_logits(
        checkpoint=checkpoint,
        encoder=encoder,
        probe=probe,
        board_crop=board_crop,
    )
    return _board_observation_from_logits(logits, timestamp_seconds=timestamp_seconds)


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


class PhysicalBoardSequenceReader:
    """Stateful physical-board reader with optional causal logit smoothing."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        ema_alpha: float | None = _DEFAULT_TEMPORAL_EMA_ALPHA,
    ) -> None:
        self.device = device
        self._smoother = (
            None
            if ema_alpha is None or ema_alpha <= 0.0
            else BoardLogitsExponentialSmoother(alpha=ema_alpha)
        )

    def reset(self) -> None:
        if self._smoother is not None:
            self._smoother.reset()

    def read_board_observation_from_frame(
        self,
        board_crop: np.ndarray,
        *,
        timestamp_seconds: float = 0.0,
    ) -> BoardObservation | None:
        try:
            checkpoint, encoder, probe = _get_runtime_model(device=self.device)
        except FileNotFoundError:
            return None

        logits = _predict_board_logits(
            checkpoint=checkpoint,
            encoder=encoder,
            probe=probe,
            board_crop=board_crop,
        )
        if self._smoother is not None:
            logits = self._smoother.update(logits)
        return _board_observation_from_logits(logits, timestamp_seconds=timestamp_seconds)

    def read_fen_from_frame(
        self,
        board_crop: np.ndarray,
        *,
        timestamp_seconds: float = 0.0,
    ) -> str | None:
        observation = self.read_board_observation_from_frame(
            board_crop,
            timestamp_seconds=timestamp_seconds,
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


def _board_observation_from_logits(
    square_logits: torch.Tensor,
    *,
    timestamp_seconds: float,
) -> BoardObservation:
    probabilities = torch.softmax(square_logits, dim=1)
    class_ids = constrained_board_class_ids(square_logits)
    confidences = probabilities.gather(1, class_ids.unsqueeze(1)).squeeze(1).cpu().tolist()
    fen = _class_ids_to_board_fen(class_ids.cpu().tolist())
    return BoardObservation(
        fen=fen,
        square_confidences=tuple(float(value) for value in confidences),
        timestamp_seconds=timestamp_seconds,
        source="physical",
    )


def _predict_board_logits(
    *,
    checkpoint: dict[str, Any],
    encoder: VisionEncoder,
    probe: nn.Module,
    board_crop: np.ndarray,
) -> torch.Tensor:
    architecture = str(checkpoint.get("architecture", "square_probe"))
    input_size = int(checkpoint.get("input_size", INPUT_SIZE))
    probe_device = next(probe.parameters()).device
    encoder.eval()
    probe.eval()

    with torch.no_grad():
        if architecture in {"board_probe", "board_probe_ensemble"}:
            board_tensor = preprocess_board_image(board_crop, size=input_size).unsqueeze(0)
            board_tensor = board_tensor.to(probe_device)
            patch_tokens = encoder.forward_patches(board_tensor)
            square_tokens = dino_patches_to_square_tokens(patch_tokens)
            logits = probe(square_tokens).squeeze(0).cpu()
            return logits

        square_crops = split_rectified_board_into_squares(board_crop)
        batch = torch.stack(
            [preprocess_square_image(crop, size=input_size) for crop in square_crops],
            dim=0,
        )
        batch = batch.to(probe_device)
        embeddings = encoder.forward_pooled(batch)
        return probe(embeddings).cpu()



def _get_runtime_model(*, device: str) -> tuple[dict[str, Any], VisionEncoder, nn.Module]:
    global _cached_model, _cached_weights_path, _cached_device
    weights_path = _resolve_weights_path()
    if (
        _cached_model is not None
        and _cached_weights_path == weights_path
        and _cached_device == device
    ):
        return _cached_model

    checkpoint = load_probe_checkpoint(weights_path)
    encoder = VisionEncoder(**_encoder_kwargs_from_checkpoint(checkpoint)).to(torch.device(device))
    probe = _build_probe_from_checkpoint(checkpoint, embed_dim=encoder.embed_dim)
    if str(checkpoint.get("architecture", "square_probe")) != "board_probe_ensemble":
        probe.load_state_dict(checkpoint["state_dict"])
    probe.to(torch.device(device))

    _cached_model = (checkpoint, encoder, probe)
    _cached_weights_path = weights_path
    _cached_device = device
    return _cached_model


def _build_probe_from_checkpoint(checkpoint: dict[str, Any], *, embed_dim: int) -> nn.Module:
    architecture = str(checkpoint.get("architecture", "square_probe"))
    if architecture == "board_probe_ensemble":
        members = checkpoint.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError("board_probe_ensemble checkpoint is missing members")
        weights = checkpoint.get("ensemble_weights")
        normalized_weights = None
        if isinstance(weights, list):
            normalized_weights = [float(weight) for weight in weights]
        probes: list[nn.Module] = []
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("Invalid board_probe_ensemble member payload")
            probe = build_board_state_probe(
                embed_dim,
                probe_config=board_probe_config_from_checkpoint(member),
            )
            probe.load_state_dict(member["state_dict"])
            probes.append(probe)
        return PhysicalBoardStateEnsembleProbe(probes, weights=normalized_weights)
    if architecture == "board_probe":
        return build_board_state_probe(
            embed_dim,
            probe_config=board_probe_config_from_checkpoint(checkpoint),
        )
    return PhysicalSquareLinearProbe(embed_dim)



def _encoder_kwargs_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    raw_feature_layer_indices = metadata.get("feature_layer_indices")
    feature_layer_indices = None
    if isinstance(raw_feature_layer_indices, list):
        feature_layer_indices = [int(index) for index in raw_feature_layer_indices]

    raw_encoder_type = metadata.get("encoder_type")
    encoder_type = str(raw_encoder_type) if raw_encoder_type is not None else "dinov2"
    raw_output_grid_size = metadata.get("output_grid_size")
    output_grid_size = int(raw_output_grid_size) if raw_output_grid_size is not None else 14

    return {
        "model_name": str(checkpoint.get("model_name", "facebook/dinov2-base")),
        "frozen": True,
        "encoder_type": encoder_type,
        "feature_layer_indices": feature_layer_indices,
        "output_grid_size": output_grid_size,
    }


def _resolve_weights_path() -> Path:
    if not _DEFAULT_WEIGHTS_PATH.exists():
        raise FileNotFoundError("No physical square-classifier weights found")
    return _DEFAULT_WEIGHTS_PATH


def load_metadata() -> dict[str, Any] | None:
    metadata_path = WEIGHTS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())
