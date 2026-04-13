#!/usr/bin/env python3
"""Build a physical board-probe ensemble checkpoint for runtime use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.board_probe import board_probe_config_from_checkpoint
from pipeline.physical.square_probe import load_probe_checkpoint

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_MODEL_CODE_VERSION = "v4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a board-probe ensemble checkpoint")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Compatible board-probe checkpoints to ensemble",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional ensemble weights; defaults to uniform",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["weight_average", "logit_average"],
        default="weight_average",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promote-to-weights", action="store_true")
    args = parser.parse_args()

    checkpoints = [load_probe_checkpoint(path) for path in args.checkpoints]
    weights = resolve_weights(args.weights, count=len(checkpoints))
    if args.mode == "weight_average":
        payload = build_weight_averaged_payload(
            checkpoints=checkpoints,
            checkpoint_paths=list(args.checkpoints),
            weights=weights,
        )
    else:
        payload = build_logit_ensemble_payload(
            checkpoints=checkpoints,
            checkpoint_paths=list(args.checkpoints),
            weights=weights,
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    if args.promote_to_weights:
        promote_to_runtime_weights(output_path, payload)


def resolve_weights(raw_weights: list[float] | None, *, count: int) -> list[float]:
    if not raw_weights:
        return [1.0 / count] * count
    if len(raw_weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(raw_weights)}")
    total = float(sum(raw_weights))
    if total <= 0.0:
        raise ValueError("Ensemble weights must sum to a positive value")
    return [float(weight) / total for weight in raw_weights]


def build_weight_averaged_payload(
    *,
    checkpoints: list[dict[str, Any]],
    checkpoint_paths: list[Path],
    weights: list[float],
) -> dict[str, Any]:
    ensure_weight_average_compatibility(checkpoints)
    averaged_state_dict = average_state_dicts(
        [checkpoint["state_dict"] for checkpoint in checkpoints],
        weights=weights,
    )
    base = checkpoints[0]
    return {
        "state_dict": averaged_state_dict,
        "model_name": base["model_name"],
        "input_size": int(base["input_size"]),
        "num_classes": int(base["num_classes"]),
        "metadata": {
            **(base.get("metadata") or {}),
            "ensemble_mode": "weight_average",
            "ensemble_members": [str(path) for path in checkpoint_paths],
            "ensemble_weights": weights,
        },
        "probe_config": board_probe_config_from_checkpoint(base),
        "architecture": str(base.get("architecture", "board_probe")),
    }


def build_logit_ensemble_payload(
    *,
    checkpoints: list[dict[str, Any]],
    checkpoint_paths: list[Path],
    weights: list[float],
) -> dict[str, Any]:
    ensure_logit_ensemble_compatibility(checkpoints)
    base = checkpoints[0]
    metadata = base.get("metadata")
    base_metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "model_name": base["model_name"],
        "input_size": int(base["input_size"]),
        "num_classes": int(base["num_classes"]),
        "metadata": {
            **base_metadata,
            "ensemble_mode": "logit_average",
            "ensemble_members": [str(path) for path in checkpoint_paths],
            "ensemble_weights": weights,
        },
        "ensemble_weights": weights,
        "members": [
            {
                "state_dict": checkpoint["state_dict"],
                "probe_config": board_probe_config_from_checkpoint(checkpoint),
            }
            for checkpoint in checkpoints
        ],
        "architecture": "board_probe_ensemble",
    }


def ensure_weight_average_compatibility(checkpoints: list[dict[str, Any]]) -> None:
    first = checkpoints[0]
    first_probe_config = board_probe_config_from_checkpoint(first)
    for checkpoint in checkpoints[1:]:
        for key in ["model_name", "input_size", "num_classes", "architecture"]:
            if checkpoint.get(key) != first.get(key):
                raise ValueError(
                    f"Checkpoint mismatch for {key}: {checkpoint.get(key)} != {first.get(key)}"
                )
        if board_probe_config_from_checkpoint(checkpoint) != first_probe_config:
            raise ValueError("Checkpoint probe_config values do not match")
        if checkpoint["state_dict"].keys() != first["state_dict"].keys():
            raise ValueError("Checkpoint state_dict keys do not match")


def ensure_logit_ensemble_compatibility(checkpoints: list[dict[str, Any]]) -> None:
    first = checkpoints[0]
    first_metadata = first.get("metadata")
    first_metadata = first_metadata if isinstance(first_metadata, dict) else {}
    for checkpoint in checkpoints[1:]:
        metadata = checkpoint.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        for key in ["model_name", "input_size", "num_classes"]:
            if checkpoint.get(key) != first.get(key):
                raise ValueError(
                    f"Checkpoint mismatch for {key}: {checkpoint.get(key)} != {first.get(key)}"
                )
        if str(checkpoint.get("architecture", "board_probe")) != "board_probe":
            raise ValueError("Logit ensembles currently support only board_probe members")
        for key in ["encoder_type", "feature_layer_indices", "output_grid_size"]:
            if metadata.get(key) != first_metadata.get(key):
                raise ValueError(
                    f"Checkpoint metadata mismatch for {key}:"
                    f" {metadata.get(key)} != {first_metadata.get(key)}"
                )


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    *,
    weights: list[float],
) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        tensors = [state_dict[key].to(torch.float32) for state_dict in state_dicts]
        combined = sum(weight * tensor for weight, tensor in zip(weights, tensors, strict=True))
        averaged[key] = combined
    return averaged


def promote_to_runtime_weights(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    weights_dir = _DEFAULT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = weights_dir / "metadata.json"
    revision = 1
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("code_version") == _MODEL_CODE_VERSION:
            revision = int(metadata.get("revision", 0)) + 1
    version = f"{_MODEL_CODE_VERSION}r{revision}"
    versioned_path = weights_dir / f"{version}.pt"
    best_path = weights_dir / "best.pt"
    versioned_path.write_bytes(checkpoint_path.read_bytes())
    best_path.write_bytes(checkpoint_path.read_bytes())
    metadata = {
        "code_version": _MODEL_CODE_VERSION,
        "revision": revision,
        "version": version,
        "model_name": payload["model_name"],
        "input_size": payload["input_size"],
        "num_classes": payload["num_classes"],
        "architecture": payload["architecture"],
        "probe_config": payload.get("probe_config"),
        "member_probe_configs": [
            member.get("probe_config") for member in payload.get("members", [])
        ],
        "ensemble_mode": payload["metadata"].get("ensemble_mode"),
        "ensemble_members": payload["metadata"].get("ensemble_members", []),
        "ensemble_weights": payload["metadata"].get("ensemble_weights", []),
        "runtime_format": "pytorch",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
