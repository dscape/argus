#!/usr/bin/env python3
"""Train a frozen-DINO board-context probe for physical per-square state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.board_data import (
    INPUT_SIZE,
    PhysicalEvalBoardDataset,
    PhysicalSyntheticRenderedBoardDataset,
)
from pipeline.physical.board_probe import (
    evaluate_board_probe,
    extract_square_token_features,
    save_board_probe_checkpoint,
    train_board_probe,
)

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_board_probe"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physical board-context square probe")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-train-positions", type=int, default=1200)
    parser.add_argument("--synthetic-val-positions", type=int, default=300)
    parser.add_argument("--synthetic-min-moves", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    output_dir = (
        args.output_dir
        or (_DEFAULT_OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading frozen encoder %s on %s", args.model_name, device)
    encoder = VisionEncoder(model_name=args.model_name, frozen=True).to(device)
    encoder.eval()

    train_dataset = PhysicalSyntheticRenderedBoardDataset(
        num_positions=args.synthetic_train_positions,
        image_size=INPUT_SIZE,
        seed=args.seed,
        augment=True,
        min_moves=args.synthetic_min_moves,
    )
    val_dataset = PhysicalSyntheticRenderedBoardDataset(
        num_positions=args.synthetic_val_positions,
        image_size=INPUT_SIZE,
        seed=args.seed + 1,
        augment=False,
        min_moves=args.synthetic_min_moves,
    )
    eval_dataset = PhysicalEvalBoardDataset(image_size=INPUT_SIZE)
    eval_annotation_ids = [row.annotation_id for row in eval_dataset.rows]

    logger.info("Extracting train square-token features")
    train_tokens, train_labels = extract_square_token_features(
        train_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting synthetic val square-token features")
    val_tokens, val_labels = extract_square_token_features(
        val_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting held-out real board features")
    eval_tokens, eval_labels = extract_square_token_features(
        eval_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    class_counts = torch.bincount(train_labels.reshape(-1), minlength=13).float()
    class_weights = class_counts.max() / class_counts.clamp_min(1.0)

    probe, best_synth_val_accuracy = train_board_probe(
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
    )

    synth_val_metrics = evaluate_board_probe(probe, val_tokens, val_labels, device=device)
    real_eval_metrics = evaluate_board_probe(
        probe,
        eval_tokens,
        eval_labels,
        device=device,
        annotation_ids=eval_annotation_ids,
    )

    checkpoint_path = output_dir / "board_probe.pt"
    save_board_probe_checkpoint(
        checkpoint_path,
        probe=probe.cpu(),
        model_name=args.model_name,
        input_size=INPUT_SIZE,
        metadata={
            "synthetic_train_positions": args.synthetic_train_positions,
            "synthetic_val_positions": args.synthetic_val_positions,
            "synthetic_min_moves": args.synthetic_min_moves,
            "best_synth_val_accuracy": best_synth_val_accuracy,
        },
    )

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "device": str(device),
        "input_size": INPUT_SIZE,
        "epochs": args.epochs,
        "synthetic_train_positions": args.synthetic_train_positions,
        "synthetic_val_positions": args.synthetic_val_positions,
        "synthetic_min_moves": args.synthetic_min_moves,
        "real_eval_positions": len(eval_dataset),
        "best_synthetic_val_accuracy": best_synth_val_accuracy,
        "synthetic_val_metrics": synth_val_metrics.to_dict(),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT.resolve())),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    logger.info("Synthetic val square accuracy: %.4f", synth_val_metrics.accuracy)
    logger.info("Real eval square accuracy: %.4f", real_eval_metrics.accuracy)
    logger.info("Real eval non-empty accuracy: %.4f", real_eval_metrics.non_empty_accuracy)
    logger.info("Real eval macro F1: %.4f", real_eval_metrics.macro_f1)
    logger.info("Real eval board exact match: %.4f", real_eval_metrics.board_exact_match or 0.0)
    logger.info("Saved report to %s", output_dir)


if __name__ == "__main__":
    main()
