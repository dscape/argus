#!/usr/bin/env python3
"""Average compatible physical board-probe checkpoints into one runtime head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.square_probe import load_probe_checkpoint

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_MODEL_CODE_VERSION = "v3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a board-probe ensemble checkpoint")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Compatible board-probe checkpoints to average",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional averaging weights; defaults to uniform",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promote-to-weights", action="store_true")
    args = parser.parse_args()

    checkpoints = [load_probe_checkpoint(path) for path in args.checkpoints]
    weights = resolve_weights(args.weights, count=len(checkpoints))
    ensure_compatible_checkpoints(checkpoints)

    averaged_state_dict = average_state_dicts(
        [checkpoint["state_dict"] for checkpoint in checkpoints],
        weights=weights,
    )
    base = checkpoints[0]
    payload = {
        "state_dict": averaged_state_dict,
        "model_name": base["model_name"],
        "input_size": int(base["input_size"]),
        "num_classes": int(base["num_classes"]),
        "metadata": {
            **(base.get("metadata") or {}),
            "ensemble_members": [str(path) for path in args.checkpoints],
            "ensemble_weights": weights,
        },
        "architecture": str(base.get("architecture", "board_probe")),
    }

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


def ensure_compatible_checkpoints(checkpoints: list[dict[str, Any]]) -> None:
    first = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        for key in ["model_name", "input_size", "num_classes", "architecture"]:
            if checkpoint.get(key) != first.get(key):
                raise ValueError(
                    f"Checkpoint mismatch for {key}: {checkpoint.get(key)} != {first.get(key)}"
                )
        if checkpoint["state_dict"].keys() != first["state_dict"].keys():
            raise ValueError("Checkpoint state_dict keys do not match")


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
        "ensemble_members": payload["metadata"].get("ensemble_members", []),
        "ensemble_weights": payload["metadata"].get("ensemble_weights", []),
        "runtime_format": "pytorch",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
