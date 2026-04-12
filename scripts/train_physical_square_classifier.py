#!/usr/bin/env python3
"""Train a frozen-DINO linear probe for physical square classification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.square_data import (
    INPUT_SIZE,
    PhysicalEvalSquareDataset,
    PhysicalSyntheticSquareDataset,
    load_eval_rows,
)
from pipeline.physical.square_probe import (
    evaluate_probe,
    extract_features,
    save_probe_checkpoint,
    train_linear_probe,
)

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_square_probe"
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
MODEL_CODE_VERSION = "v1"


def _next_version(weights_dir: Path) -> tuple[int, str]:
    metadata_path = weights_dir / "metadata.json"
    revision = 1
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("code_version") == MODEL_CODE_VERSION:
            revision = int(metadata.get("revision", 0)) + 1
    return revision, f"{MODEL_CODE_VERSION}r{revision}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physical square linear probe")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-train-samples", type=int, default=600)
    parser.add_argument("--synthetic-val-samples", type=int, default=150)
    parser.add_argument("--eval-max-per-class", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for metrics/checkpoints (default: outputs/physical_square_probe/<timestamp>)"
        ),
    )
    parser.add_argument(
        "--promote-to-weights",
        action="store_true",
        help="Copy the trained linear head to weights/physical/ for runtime use",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = args.output_dir or (_DEFAULT_OUTPUT_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading frozen encoder %s on %s", args.model_name, device)
    encoder = VisionEncoder(model_name=args.model_name, frozen=True).to(device)
    encoder.eval()

    logger.info("Building synthetic physical datasets at %dx%d", INPUT_SIZE, INPUT_SIZE)
    train_dataset = PhysicalSyntheticSquareDataset(
        num_samples_per_class=args.synthetic_train_samples,
        image_size=INPUT_SIZE,
        seed=args.seed,
        augment=True,
    )
    val_dataset = PhysicalSyntheticSquareDataset(
        num_samples_per_class=args.synthetic_val_samples,
        image_size=INPUT_SIZE,
        seed=args.seed + 1,
        augment=False,
    )

    eval_rows = load_eval_rows()
    eval_dataset = PhysicalEvalSquareDataset(
        rows=eval_rows,
        image_size=INPUT_SIZE,
        max_per_class=(args.eval_max_per_class or None),
        seed=args.seed,
    )
    eval_annotation_ids = [row.annotation_id for row in eval_dataset.rows]

    logger.info("Extracting train features")
    train_features, train_labels = extract_features(
        train_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting synthetic val features")
    val_features, val_labels = extract_features(
        val_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting held-out real eval features")
    eval_features, eval_labels = extract_features(
        eval_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    logger.info("Training linear probe")
    probe, best_synth_val_accuracy = train_linear_probe(
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    synth_val_metrics = evaluate_probe(
        probe,
        val_features,
        val_labels,
        device=device,
    )
    real_eval_metrics = evaluate_probe(
        probe,
        eval_features,
        eval_labels,
        device=device,
        board_annotation_ids=eval_annotation_ids,
    )

    output_dir = output_dir.resolve()
    checkpoint_path = output_dir / "linear_probe.pt"
    save_probe_checkpoint(
        checkpoint_path,
        probe=probe.cpu(),
        model_name=args.model_name,
        input_size=INPUT_SIZE,
        metadata={
            "synthetic_train_samples": args.synthetic_train_samples,
            "synthetic_val_samples": args.synthetic_val_samples,
            "eval_max_per_class": args.eval_max_per_class,
            "best_synth_val_accuracy": best_synth_val_accuracy,
        },
    )

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "device": str(device),
        "input_size": INPUT_SIZE,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "synthetic_train_size": len(train_dataset),
        "synthetic_val_size": len(val_dataset),
        "real_eval_size": len(eval_dataset),
        "best_synthetic_val_accuracy": best_synth_val_accuracy,
        "synthetic_val_metrics": synth_val_metrics.to_dict(),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT.resolve())),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    summary_lines = [
        "# Physical square linear probe",
        "",
        f"- model: `{args.model_name}`",
        f"- synthetic train size: `{len(train_dataset)}`",
        f"- synthetic val size: `{len(val_dataset)}`",
        f"- real eval size: `{len(eval_dataset)}`",
        f"- best synthetic val accuracy: `{best_synth_val_accuracy:.4f}`",
        f"- real eval square accuracy: `{real_eval_metrics.accuracy:.4f}`",
        f"- real eval non-empty accuracy: `{real_eval_metrics.non_empty_accuracy:.4f}`",
        f"- real eval macro F1: `{real_eval_metrics.macro_f1:.4f}`",
        f"- real eval board exact match: `{(real_eval_metrics.board_exact_match or 0.0):.4f}`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    if args.promote_to_weights:
        weights_dir = _DEFAULT_WEIGHTS_DIR
        weights_dir.mkdir(parents=True, exist_ok=True)
        revision, version = _next_version(weights_dir)
        versioned_path = weights_dir / f"{version}.pt"
        best_path = weights_dir / "best.pt"
        versioned_path.write_bytes(checkpoint_path.read_bytes())
        best_path.write_bytes(checkpoint_path.read_bytes())
        metadata = {
            "code_version": MODEL_CODE_VERSION,
            "revision": revision,
            "version": version,
            "trained_at": report["trained_at"],
            "model_name": args.model_name,
            "input_size": INPUT_SIZE,
            "best_synthetic_val_accuracy": round(best_synth_val_accuracy, 4),
            "real_eval_metrics": real_eval_metrics.to_dict(),
            "sources": {
                "synthetic_train_samples": args.synthetic_train_samples,
                "synthetic_val_samples": args.synthetic_val_samples,
                "held_out_eval_size": len(eval_dataset),
            },
            "runtime_format": "pytorch",
            "architecture": "dinov2_linear_probe",
        }
        (weights_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("Promoted runtime weights to %s", best_path)

    logger.info("Synthetic val accuracy: %.4f", synth_val_metrics.accuracy)
    logger.info("Real eval square accuracy: %.4f", real_eval_metrics.accuracy)
    logger.info("Real eval non-empty accuracy: %.4f", real_eval_metrics.non_empty_accuracy)
    logger.info("Real eval macro F1: %.4f", real_eval_metrics.macro_f1)
    logger.info("Real eval board exact match: %.4f", real_eval_metrics.board_exact_match or 0.0)
    logger.info("Saved report to %s", output_dir)


if __name__ == "__main__":
    main()
