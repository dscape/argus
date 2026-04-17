#!/usr/bin/env python3
"""Train one stage of the two-stage square classifier (occupancy or piece)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.oblique_square_context import load_annotated_oblique_rows
from pipeline.physical.square_classifier_data import (
    OccupancySquareDataset,
    PieceSquareDataset,
    class_counts,
)
from pipeline.physical.square_classifiers import (
    OCCUPANCY_CLASS_NAMES,
    OCCUPANCY_NUM_CLASSES,
    PIECE_CLASS_NAMES,
    PIECE_NUM_CLASSES,
    SquareClassifier,
    SquareClassifierConfig,
)
from pipeline.physical.square_crop import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
)

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "weights" / "physical" / "square_classifier"


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(resolve_device(args.device))
    output_dir = (args.output_dir or (_DEFAULT_OUTPUT_ROOT / args.task)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    train_rows = load_annotated_oblique_rows(args.physical_train_root)
    val_rows = load_annotated_oblique_rows(args.physical_val_root)

    num_classes, input_size, train_dataset, val_dataset = _build_datasets(
        task=args.task,
        train_rows=train_rows,
        val_rows=val_rows,
        input_size_override=args.input_size,
        augment_train=args.augment,
    )

    model_name = args.model_name or default_model_name_for_encoder_type(args.encoder_type)
    encoder = VisionEncoder(
        encoder_type=args.encoder_type,
        model_name=model_name,
        frozen=args.freeze_encoder,
    )
    if args.unfreeze_last_n > 0:
        encoder.unfreeze_last_n_layers(args.unfreeze_last_n)
    classifier = SquareClassifier(
        vision_encoder=encoder,
        config=SquareClassifierConfig(num_classes=num_classes, dropout=args.dropout),
    ).to(device)

    if args.class_weighting:
        class_weights = _class_weights(train_dataset, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
    )

    best_val_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for epoch in range(args.epochs):
        classifier.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = classifier(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]
            running_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            running_total += images.shape[0]

        scheduler.step()
        train_loss = running_loss / max(running_total, 1)
        train_accuracy = running_correct / max(running_total, 1)
        val_accuracy = _evaluate(classifier, val_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}

    if best_state is None:
        best_state = classifier.state_dict()

    checkpoint = {
        "architecture": f"square_classifier_{args.task}",
        "state_dict": best_state,
        "encoder_config": {
            "encoder_type": args.encoder_type,
            "model_name": model_name,
            "frozen": args.freeze_encoder,
        },
        **classifier.checkpoint_config(),
        "input_size": input_size,
        "task": args.task,
        "best_val_accuracy": best_val_accuracy,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / f"{args.task}_classifier.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "task": args.task,
        "num_classes": num_classes,
        "class_names": OCCUPANCY_CLASS_NAMES if args.task == "occupancy" else PIECE_CLASS_NAMES,
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "input_size": input_size,
        "encoder_type": args.encoder_type,
        "model_name": model_name,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "best_val_accuracy": best_val_accuracy,
        "train_class_distribution": class_counts(train_dataset),
        "val_class_distribution": class_counts(val_dataset),
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_datasets(
    *,
    task: str,
    train_rows,
    val_rows,
    input_size_override: int | None,
    augment_train: bool,
):
    if task == "occupancy":
        input_size = input_size_override or DEFAULT_OCCUPANCY_CROP_SIZE
        train_dataset = OccupancySquareDataset(
            rows=train_rows, input_size=input_size, augment=augment_train
        )
        val_dataset = OccupancySquareDataset(rows=val_rows, input_size=input_size)
        return OCCUPANCY_NUM_CLASSES, input_size, train_dataset, val_dataset
    if task == "piece":
        input_size = input_size_override or DEFAULT_PIECE_CROP_SIZE
        train_dataset = PieceSquareDataset(
            rows=train_rows, input_size=input_size, augment=augment_train
        )
        val_dataset = PieceSquareDataset(rows=val_rows, input_size=input_size)
        return PIECE_NUM_CLASSES, input_size, train_dataset, val_dataset
    raise ValueError(f"unknown task: {task}")


def _class_weights(dataset, *, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for sample in dataset.indices:
        if num_classes == OCCUPANCY_NUM_CLASSES:
            label = 0 if sample.square_class == 0 else 1
        else:
            label = sample.square_class - 1
        counts[label] += 1
    # Inverse-frequency weights, normalized to mean 1.
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights * (num_classes / weights.sum())
    return weights.float()


def _evaluate(model: SquareClassifier, loader: DataLoader, *, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += images.shape[0]
    return correct / max(total, 1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a two-stage square classifier.")
    parser.add_argument("--task", choices=("occupancy", "piece"), required=True)
    parser.add_argument(
        "--physical-train-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "train",
    )
    parser.add_argument(
        "--physical-val-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "val",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-type", choices=("dinov2", "siglip2"), default="dinov2")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=0,
        help="Unfreeze the last N transformer blocks of the encoder for fine-tuning.",
    )
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--class-weighting",
        action="store_true",
        default=False,
        help="Apply inverse-frequency class weighting in the loss (can hurt majority classes).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable color/affine/blur augmentation on the training set (not val).",
    )
    return parser


if __name__ == "__main__":
    main()
