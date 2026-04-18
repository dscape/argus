"""Runtime physical-board reader backed by a frozen feature probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from argus.model.vision_encoder import VisionEncoder
from pipeline.physical.board_probe.board_data import (
    INPUT_SIZE,
    preprocess_board_image,
    preprocess_board_neighborhood_image,
)
from pipeline.physical.board_probe.probe import (
    PhysicalBoardStateEnsembleProbe,
    board_probe_config_from_checkpoint,
    build_board_state_probe,
    dino_patches_to_square_tokens,
    sample_projected_square_tokens_from_patch_tokens,
)
from pipeline.physical.board_probe.square_probe import load_probe_checkpoint
from pipeline.shared import (
    AdaptiveBoardLogitsExponentialSmoother,
    BoardLogitsExponentialSmoother,
    BoardObservation,
    SQUARE_CLASS_NAMES,
    apply_board_logit_bias,
    constrained_board_class_ids,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
_DEFAULT_TEMPORAL_LOW_ALPHA = 0.02
_DEFAULT_TEMPORAL_HIGH_ALPHA = 0.12
_DEFAULT_TEMPORAL_CHANGE_THRESHOLD = 8
_CLASS_TO_SYMBOL = {index: name for index, name in enumerate(SQUARE_CLASS_NAMES)}

_cached_model: tuple[dict[str, Any], VisionEncoder, nn.Module] | None = None
_cached_weights_path: Path | None = None
_cached_device: str | None = None
_METADATA_UNSET = object()
_cached_metadata: dict[str, Any] | None | object = _METADATA_UNSET


def read_board_logits_from_frame(
    board_crop: np.ndarray,
    *,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    device: str = "cpu",
    weights_path: str | Path | None = None,
) -> torch.Tensor | None:
    """Read raw per-square logits from one physical board crop."""
    try:
        if weights_path is None:
            checkpoint, encoder, probe = _get_runtime_model(device=device)
        else:
            checkpoint, encoder, probe = _get_runtime_model(
                device=device,
                weights_path=weights_path,
            )
    except FileNotFoundError:
        return None

    logits = _predict_board_logits(
        checkpoint=checkpoint,
        encoder=encoder,
        probe=probe,
        board_crop=board_crop,
        corners=corners,
    )
    return apply_board_logit_bias(logits, _load_runtime_logit_bias(weights_path))


def read_board_logits_batch_from_frames(
    board_crops: list[np.ndarray],
    *,
    corners_list: list[tuple[tuple[float, float], ...] | list[list[float]] | None] | None = None,
    device: str = "cpu",
    weights_path: str | Path | None = None,
    batch_size: int = 16,
) -> list[torch.Tensor] | None:
    """Read raw per-square logits from multiple physical board crops."""
    if not board_crops:
        return []
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    try:
        if weights_path is None:
            checkpoint, encoder, probe = _get_runtime_model(device=device)
        else:
            checkpoint, encoder, probe = _get_runtime_model(
                device=device,
                weights_path=weights_path,
            )
    except FileNotFoundError:
        return None

    logits_list = _predict_board_logits_batch(
        checkpoint=checkpoint,
        encoder=encoder,
        probe=probe,
        board_crops=board_crops,
        corners_list=corners_list,
        batch_size=batch_size,
    )
    logit_bias = _load_runtime_logit_bias(weights_path)
    return [apply_board_logit_bias(logits, logit_bias) for logits in logits_list]


def board_observation_from_logits(
    square_logits: torch.Tensor,
    *,
    timestamp_seconds: float = 0.0,
) -> BoardObservation:
    return _board_observation_from_logits(square_logits, timestamp_seconds=timestamp_seconds)


def read_board_observation_from_frame(
    board_crop: np.ndarray,
    *,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
    weights_path: str | Path | None = None,
) -> BoardObservation | None:
    """Read a rectified physical board crop into a source-agnostic observation."""
    logits = read_board_logits_from_frame(
        board_crop,
        corners=corners,
        device=device,
        weights_path=weights_path,
    )
    if logits is None:
        return None
    return board_observation_from_logits(logits, timestamp_seconds=timestamp_seconds)


def read_fen_from_frame(
    board_crop: np.ndarray,
    *,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    timestamp_seconds: float = 0.0,
    device: str = "cpu",
    weights_path: str | Path | None = None,
) -> str | None:
    """Return the physical-board FEN when runtime weights are available."""
    observation = read_board_observation_from_frame(
        board_crop,
        corners=corners,
        timestamp_seconds=timestamp_seconds,
        device=device,
        weights_path=weights_path,
    )
    return None if observation is None else observation.fen


class PhysicalBoardLogitsSequenceReader:
    """Stateful physical-board logits reader with optional causal smoothing."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        ema_alpha: float | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.weights_path = None if weights_path is None else Path(weights_path)
        self._smoother = _build_temporal_smoother(
            ema_alpha=ema_alpha,
            weights_path=self.weights_path,
        )
        self._logit_bias = _load_runtime_logit_bias(self.weights_path)

    def reset(self) -> None:
        if self._smoother is not None:
            self._smoother.reset()

    def read_board_logits_from_frame(
        self,
        board_crop: np.ndarray,
        *,
        corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    ) -> torch.Tensor | None:
        try:
            if self.weights_path is None:
                checkpoint, encoder, probe = _get_runtime_model(device=self.device)
            else:
                checkpoint, encoder, probe = _get_runtime_model(
                    device=self.device,
                    weights_path=self.weights_path,
                )
        except FileNotFoundError:
            return None

        logits = _predict_board_logits(
            checkpoint=checkpoint,
            encoder=encoder,
            probe=probe,
            board_crop=board_crop,
            corners=corners,
        )
        logits = apply_board_logit_bias(logits, self._logit_bias)
        return self.smooth_logits(logits)

    def smooth_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self._smoother is not None:
            return self._smoother.update(logits)
        return logits


class PhysicalBoardSequenceReader:
    """Stateful physical-board reader with optional causal logit smoothing."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        ema_alpha: float | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self._logits_reader = PhysicalBoardLogitsSequenceReader(
            device=device,
            ema_alpha=ema_alpha,
            weights_path=weights_path,
        )
        self._smoother = self._logits_reader._smoother

    def reset(self) -> None:
        self._logits_reader.reset()

    def read_board_observation_from_frame(
        self,
        board_crop: np.ndarray,
        *,
        corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
        timestamp_seconds: float = 0.0,
    ) -> BoardObservation | None:
        logits = self._logits_reader.read_board_logits_from_frame(board_crop, corners=corners)
        if logits is None:
            return None
        return board_observation_from_logits(logits, timestamp_seconds=timestamp_seconds)

    def read_fen_from_frame(
        self,
        board_crop: np.ndarray,
        *,
        corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
        timestamp_seconds: float = 0.0,
    ) -> str | None:
        observation = self.read_board_observation_from_frame(
            board_crop,
            corners=corners,
            timestamp_seconds=timestamp_seconds,
        )
        return None if observation is None else observation.fen


def _build_temporal_smoother(
    *,
    ema_alpha: float | None,
    weights_path: str | Path | None = None,
) -> BoardLogitsExponentialSmoother | AdaptiveBoardLogitsExponentialSmoother | None:
    if ema_alpha is not None:
        if ema_alpha <= 0.0:
            return None
        return BoardLogitsExponentialSmoother(alpha=ema_alpha)

    metadata = load_metadata() if weights_path is None else load_metadata(weights_path)
    if metadata is not None:
        smoothing_config = metadata.get("recommended_temporal_smoothing")
        if isinstance(smoothing_config, dict):
            mode = str(smoothing_config.get("mode", "adaptive_ema"))
            if mode == "off":
                return None
            if mode == "fixed_ema":
                return BoardLogitsExponentialSmoother(
                    alpha=float(smoothing_config.get("alpha", _DEFAULT_TEMPORAL_LOW_ALPHA))
                )
            if mode == "adaptive_ema":
                return AdaptiveBoardLogitsExponentialSmoother(
                    low_alpha=float(smoothing_config.get("low_alpha", _DEFAULT_TEMPORAL_LOW_ALPHA)),
                    high_alpha=float(
                        smoothing_config.get("high_alpha", _DEFAULT_TEMPORAL_HIGH_ALPHA)
                    ),
                    high_alpha_change_threshold=int(
                        smoothing_config.get(
                            "change_threshold",
                            _DEFAULT_TEMPORAL_CHANGE_THRESHOLD,
                        )
                    ),
                )
            raise ValueError(f"Unsupported temporal smoothing mode: {mode}")

        recommended_alpha = metadata.get("recommended_temporal_ema_alpha")
        if recommended_alpha is not None:
            return BoardLogitsExponentialSmoother(alpha=float(recommended_alpha))

    return AdaptiveBoardLogitsExponentialSmoother(
        low_alpha=_DEFAULT_TEMPORAL_LOW_ALPHA,
        high_alpha=_DEFAULT_TEMPORAL_HIGH_ALPHA,
        high_alpha_change_threshold=_DEFAULT_TEMPORAL_CHANGE_THRESHOLD,
    )


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
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
) -> torch.Tensor:
    architecture = str(checkpoint.get("architecture", "board_probe"))
    if architecture not in {"board_probe", "board_probe_ensemble"}:
        raise ValueError(f"Unsupported physical runtime architecture: {architecture}")

    input_size = int(checkpoint.get("input_size", INPUT_SIZE))
    probe_device = next(probe.parameters()).device
    encoder.eval()
    probe.eval()

    board_tensor, corners_tensor = _prepare_runtime_board_input(
        board_crop,
        corners=corners,
        input_size=input_size,
    )

    with torch.no_grad():
        patch_tokens = encoder.forward_patches(board_tensor.unsqueeze(0).to(probe_device))
        square_tokens = _pool_square_tokens_batch(
            patch_tokens,
            corners_list=[corners_tensor],
            image_size=input_size,
        )
        return probe(square_tokens).squeeze(0).cpu()


def _predict_board_logits_batch(
    *,
    checkpoint: dict[str, Any],
    encoder: VisionEncoder,
    probe: nn.Module,
    board_crops: list[np.ndarray],
    corners_list: list[tuple[tuple[float, float], ...] | list[list[float]] | None] | None,
    batch_size: int,
) -> list[torch.Tensor]:
    if not board_crops:
        return []

    if corners_list is None:
        corners_per_board = [None] * len(board_crops)
    else:
        if len(corners_list) != len(board_crops):
            raise ValueError(
                "corners_list length must match board_crops length, got "
                f"{len(corners_list)} and {len(board_crops)}"
            )
        corners_per_board = corners_list

    architecture = str(checkpoint.get("architecture", "board_probe"))
    if architecture not in {"board_probe", "board_probe_ensemble"}:
        raise ValueError(f"Unsupported physical runtime architecture: {architecture}")

    input_size = int(checkpoint.get("input_size", INPUT_SIZE))
    probe_device = next(probe.parameters()).device
    encoder.eval()
    probe.eval()

    prepared = [
        _prepare_runtime_board_input(
            board_crop,
            corners=corners,
            input_size=input_size,
        )
        for board_crop, corners in zip(board_crops, corners_per_board)
    ]
    tensors = [board_tensor for board_tensor, _corners in prepared]
    prepared_corners = [corners_tensor for _board_tensor, corners_tensor in prepared]

    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch_tensors = tensors[start : start + batch_size]
            batch_corners = prepared_corners[start : start + batch_size]
            batch = torch.stack(batch_tensors, dim=0).to(probe_device)
            patch_tokens = encoder.forward_patches(batch)
            square_tokens = _pool_square_tokens_batch(
                patch_tokens,
                corners_list=batch_corners,
                image_size=input_size,
            )
            logits_batch = probe(square_tokens).cpu()
            logits_list.extend(logits_batch.unbind(0))
    return logits_list


def _prepare_runtime_board_input(
    board_crop: np.ndarray,
    *,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _should_use_direct_square_pooling(board_crop, corners):
        return preprocess_board_image(board_crop, size=input_size), None
    board_tensor, corners_tensor = preprocess_board_neighborhood_image(
        board_crop,
        _runtime_corners(board_crop, corners),
        size=input_size,
    )
    return board_tensor, corners_tensor


def _pool_square_tokens_batch(
    patch_tokens: torch.Tensor,
    *,
    corners_list: list[torch.Tensor | None],
    image_size: int,
) -> torch.Tensor:
    batch_size = patch_tokens.shape[0]
    if len(corners_list) != batch_size:
        raise ValueError(
            f"corners_list length {len(corners_list)} must match batch size {batch_size}"
        )

    square_tokens = torch.empty(
        (batch_size, 64, patch_tokens.shape[-1]),
        device=patch_tokens.device,
        dtype=patch_tokens.dtype,
    )

    direct_indices = [index for index, corners in enumerate(corners_list) if corners is None]
    if direct_indices:
        direct_index_tensor = torch.tensor(
            direct_indices,
            device=patch_tokens.device,
            dtype=torch.long,
        )
        direct_square_tokens = dino_patches_to_square_tokens(
            patch_tokens.index_select(0, direct_index_tensor)
        )
        square_tokens.index_copy_(0, direct_index_tensor, direct_square_tokens)

    projected_indices = [index for index, corners in enumerate(corners_list) if corners is not None]
    if projected_indices:
        projected_index_tensor = torch.tensor(
            projected_indices,
            device=patch_tokens.device,
            dtype=torch.long,
        )
        projected_corners = torch.stack(
            [corners for corners in corners_list if corners is not None],
            dim=0,
        ).to(patch_tokens.device)
        projected_square_tokens = sample_projected_square_tokens_from_patch_tokens(
            patch_tokens.index_select(0, projected_index_tensor),
            corners=projected_corners,
            image_size=image_size,
        )
        square_tokens.index_copy_(0, projected_index_tensor, projected_square_tokens)

    return square_tokens


def _should_use_direct_square_pooling(
    board_crop: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None,
) -> bool:
    if corners is None:
        return True
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        return False
    return np.allclose(points, np.asarray(_full_frame_corners(board_crop), dtype=np.float32), atol=1e-3)


def _full_frame_corners(board_crop: np.ndarray) -> tuple[tuple[float, float], ...]:
    height, width = board_crop.shape[:2]
    return (
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    )


def _get_runtime_model(
    *,
    device: str,
    weights_path: str | Path | None = None,
) -> tuple[dict[str, Any], VisionEncoder, nn.Module]:
    global _cached_model, _cached_weights_path, _cached_device
    if weights_path is None:
        resolved_weights_path = _resolve_weights_path()
    else:
        resolved_weights_path = _resolve_weights_path(weights_path)
    if (
        _cached_model is not None
        and _cached_weights_path == resolved_weights_path
        and _cached_device == device
    ):
        return _cached_model

    checkpoint = load_probe_checkpoint(resolved_weights_path)
    encoder = VisionEncoder(**_encoder_kwargs_from_checkpoint(checkpoint)).to(torch.device(device))
    probe = _build_probe_from_checkpoint(checkpoint, embed_dim=encoder.embed_dim)
    if str(checkpoint.get("architecture", "board_probe")) != "board_probe_ensemble":
        probe.load_state_dict(checkpoint["state_dict"])
    probe.to(torch.device(device))

    _cached_model = (checkpoint, encoder, probe)
    _cached_weights_path = resolved_weights_path
    _cached_device = device
    return _cached_model


def _build_probe_from_checkpoint(checkpoint: dict[str, Any], *, embed_dim: int) -> nn.Module:
    architecture = str(checkpoint.get("architecture", "board_probe"))
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
    raise ValueError(f"Unsupported physical runtime architecture: {architecture}")


def _runtime_corners(
    board_crop: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None,
) -> tuple[tuple[float, float], ...] | list[list[float]]:
    if corners is not None:
        return corners
    return _full_frame_corners(board_crop)


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


def _resolve_weights_path(weights_path: str | Path | None = None) -> Path:
    if weights_path is None:
        resolved = _DEFAULT_WEIGHTS_PATH
    else:
        resolved = Path(weights_path)
        if not resolved.is_absolute():
            resolved = (_PROJECT_ROOT / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"No physical square-classifier weights found: {resolved}")
    return resolved


def _load_runtime_logit_bias(weights_path: str | Path | None = None) -> list[float] | None:
    metadata = load_metadata() if weights_path is None else load_metadata(weights_path)
    if metadata is None:
        return None
    raw_bias = metadata.get("class_logit_bias")
    if raw_bias is None:
        return None
    if not isinstance(raw_bias, list):
        raise ValueError("class_logit_bias metadata must be a list of floats")
    return [float(value) for value in raw_bias]


def load_metadata(weights_path: str | Path | None = None) -> dict[str, Any] | None:
    global _cached_metadata
    if weights_path is not None:
        checkpoint = load_probe_checkpoint(_resolve_weights_path(weights_path))
        metadata = checkpoint.get("metadata")
        return metadata if isinstance(metadata, dict) else None
    if _cached_metadata is not _METADATA_UNSET:
        return None if _cached_metadata is None else _cached_metadata
    metadata_path = WEIGHTS_DIR / "metadata.json"
    if not metadata_path.exists():
        _cached_metadata = None
        return None
    _cached_metadata = json.loads(metadata_path.read_text())
    return _cached_metadata
