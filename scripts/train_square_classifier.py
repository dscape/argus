#!/usr/bin/env python3
"""Train one stage of the two-stage square classifier (occupancy or piece)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    OccupancySquareDataset,
    PieceSquareDataset,
    class_counts,
    load_synthetic_oblique_rows,
)
from pipeline.physical.two_stage.classifiers import (
    OCCUPANCY_CLASS_NAMES,
    OCCUPANCY_NUM_CLASSES,
    PIECE_CLASS_NAMES,
    PIECE_NUM_CLASSES,
    SquareClassifier,
    SquareClassifierConfig,
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
    _validate_args(args)

    physical_train_row_count: int | None = None
    selected_synthetic_train_row_count: int | None = None
    train_allow_clip_fallback = False

    if args.source == "chesscog-png":
        from pipeline.physical.chesscog_baseline.png_square_dataset import (
            OccupancyPngDataset,
            PiecePngDataset,
        )

        print(f"loading chesscog PNG dataset from {args.chesscog_png_root}")
        if args.task == "occupancy":
            input_size = args.input_size or 100
            train_dataset = OccupancyPngDataset(
                root=args.chesscog_png_root / "occupancy",
                split="train",
                input_size=input_size,
                augment=args.augment,
            )
            val_dataset = OccupancyPngDataset(
                root=args.chesscog_png_root / "occupancy",
                split="val",
                input_size=input_size,
            )
            num_classes = OCCUPANCY_NUM_CLASSES
        else:
            input_size = args.input_size or 200
            train_dataset = PiecePngDataset(
                root=args.chesscog_png_root / "pieces",
                split="train",
                input_size=input_size,
                augment=args.augment,
            )
            val_dataset = PiecePngDataset(
                root=args.chesscog_png_root / "pieces",
                split="val",
                input_size=input_size,
            )
            num_classes = PIECE_NUM_CLASSES
    else:
        if args.combined_split_seed is not None:
            # Merge train+val and randomly split for an apples-to-apples model-capacity test
            # (breaks cross-game held-out semantics).
            merged = load_annotated_oblique_rows(
                args.physical_train_root
            ) + load_annotated_oblique_rows(args.physical_val_root)
            rng = random.Random(args.combined_split_seed)
            shuffled = list(merged)
            rng.shuffle(shuffled)
            cutoff = int(len(shuffled) * 0.8)
            train_rows = shuffled[:cutoff]
            val_rows = shuffled[cutoff:]
            print(
                f"combined split seed={args.combined_split_seed}: "
                f"train={len(train_rows)} val={len(val_rows)}"
            )
        else:
            train_rows = load_annotated_oblique_rows(args.physical_train_root)
            val_rows = load_annotated_oblique_rows(args.physical_val_root)

        physical_train_row_count = len(train_rows)
        if args.synthetic_train_root is not None:
            synthetic_train_rows = load_synthetic_oblique_rows(args.synthetic_train_root)
            train_rows, selected_synthetic_train_row_count = _merge_train_rows(
                physical_rows=train_rows,
                synthetic_rows=synthetic_train_rows,
                mix_ratio=args.mix_ratio,
                seed=args.seed,
            )
            train_allow_clip_fallback = bool(selected_synthetic_train_row_count)
            print(
                "training rows: "
                f"physical={physical_train_row_count} "
                f"synthetic={selected_synthetic_train_row_count} "
                f"total={len(train_rows)}"
            )

        num_classes, input_size, train_dataset, val_dataset = _build_datasets(
            task=args.task,
            train_rows=train_rows,
            val_rows=val_rows,
            input_size_override=args.input_size,
            augment_train=args.augment,
            occupancy_pad_ratio=args.occupancy_pad_ratio,
            piece_augment_shear=args.piece_augment_shear,
            train_allow_clip_fallback=train_allow_clip_fallback,
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
    if args.initialize_checkpoint is not None:
        checkpoint = torch.load(args.initialize_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Checkpoint is missing model_state_dict/state_dict: {args.initialize_checkpoint}"
            )
        classifier.load_state_dict(state_dict)

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

    initial_val_accuracy = _evaluate(classifier, val_loader, device=device)
    best_val_accuracy = initial_val_accuracy
    best_state = {
        key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()
    }

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
            # Always materialize on CPU to avoid MPS hangs in torch.save
            # for large models with non-standard architectures (e.g. DINOv3).
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

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
        "source": args.source,
        "num_classes": num_classes,
        "class_names": OCCUPANCY_CLASS_NAMES if args.task == "occupancy" else PIECE_CLASS_NAMES,
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "input_size": input_size,
        "encoder_type": args.encoder_type,
        "model_name": model_name,
        "initialize_checkpoint": (
            None if args.initialize_checkpoint is None else str(args.initialize_checkpoint)
        ),
        "synthetic_train_root": (
            None if args.synthetic_train_root is None else str(args.synthetic_train_root)
        ),
        "mix_ratio": args.mix_ratio,
        "physical_train_rows": physical_train_row_count,
        "synthetic_train_rows": selected_synthetic_train_row_count,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "best_val_accuracy": best_val_accuracy,
        "train_class_distribution": class_counts(train_dataset),
        "val_class_distribution": class_counts(val_dataset),
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _validate_args(args: argparse.Namespace) -> None:
    if args.mix_ratio is not None and args.synthetic_train_root is None:
        raise ValueError("--mix-ratio requires --synthetic-train-root")
    if args.source == "chesscog-png" and args.synthetic_train_root is not None:
        raise ValueError("--synthetic-train-root is only supported with --source argus-projection")
    if args.source == "chesscog-png" and args.mix_ratio is not None:
        raise ValueError("--mix-ratio is only supported with --source argus-projection")


def _merge_train_rows(
    *,
    physical_rows,
    synthetic_rows,
    mix_ratio: float | None,
    seed: int,
):
    if mix_ratio is not None and not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"--mix-ratio must be in [0, 1], got {mix_ratio}")
    if not synthetic_rows:
        return list(physical_rows), 0
    if mix_ratio is None:
        return [*physical_rows, *synthetic_rows], len(synthetic_rows)
    if mix_ratio == 0.0:
        return list(physical_rows), 0
    if mix_ratio == 1.0 or not physical_rows:
        return list(synthetic_rows), len(synthetic_rows)

    target_synthetic = max(1, int(round(len(physical_rows) * mix_ratio / (1.0 - mix_ratio))))
    if target_synthetic >= len(synthetic_rows):
        selected_synthetic_rows = list(synthetic_rows)
    else:
        shuffled = list(synthetic_rows)
        random.Random(seed).shuffle(shuffled)
        selected_synthetic_rows = shuffled[:target_synthetic]
    return [*physical_rows, *selected_synthetic_rows], len(selected_synthetic_rows)


def _build_datasets(
    *,
    task: str,
    train_rows,
    val_rows,
    input_size_override: int | None,
    augment_train: bool,
    occupancy_pad_ratio: float = 0.3,
    piece_augment_shear: bool = False,
    train_allow_clip_fallback: bool = False,
):
    if task == "occupancy":
        input_size = input_size_override or DEFAULT_OCCUPANCY_CROP_SIZE
        train_dataset = OccupancySquareDataset(
            rows=train_rows,
            input_size=input_size,
            augment=augment_train,
            pad_ratio=occupancy_pad_ratio,
            allow_clip_fallback=train_allow_clip_fallback,
        )
        val_dataset = OccupancySquareDataset(
            rows=val_rows,
            input_size=input_size,
            pad_ratio=occupancy_pad_ratio,
        )
        return OCCUPANCY_NUM_CLASSES, input_size, train_dataset, val_dataset
    if task == "piece":
        input_size = input_size_override or DEFAULT_PIECE_CROP_SIZE
        if piece_augment_shear:
            # Paper's λ ∈ [-0.1, 0.25] corresponds to shear angles [-5.7°, 14.3°].
            # Also nudge brightness/contrast up slightly per paper Ch. 5.2.1.3.
            import torchvision.transforms.v2 as T
            from pipeline.physical.two_stage import classifier_data as _cd

            _cd._AUGMENTATION_PIPELINE = T.Compose(
                [
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.04),
                    T.RandomAffine(
                        degrees=8,
                        translate=(0.05, 0.05),
                        scale=(0.92, 1.08),
                        shear=(-5.7, 14.3),
                    ),
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                    T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0),
                ]
            )
            print("enabled piece shear augmentation (shear=±(-5.7, 14.3)°)")
        train_dataset = PieceSquareDataset(
            rows=train_rows,
            input_size=input_size,
            augment=augment_train,
            allow_clip_fallback=train_allow_clip_fallback,
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
    parser.add_argument("--initialize-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder-type",
        choices=("dinov2", "dinov3", "siglip2"),
        default="dinov2",
    )
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
    parser.add_argument(
        "--occupancy-pad-ratio",
        type=float,
        default=0.3,
        help=(
            "Padding ratio for occupancy crops (Study 3a). Applied to both train and val datasets."
        ),
    )
    parser.add_argument(
        "--piece-augment-shear",
        action="store_true",
        default=False,
        help="Add shear to piece augmentation pipeline (Study 3b).",
    )
    parser.add_argument(
        "--source",
        choices=("argus-projection", "chesscog-png"),
        default="argus-projection",
        help=(
            "argus-projection: on-the-fly projection from annotations. "
            "chesscog-png: load PNGs from a chesscog-style ImageFolder tree "
            "(Study 2)."
        ),
    )
    parser.add_argument(
        "--chesscog-png-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "chesscog_baseline",
        help="Root of chesscog-style PNG dataset (must contain occupancy/ and pieces/).",
    )
    parser.add_argument(
        "--synthetic-train-root",
        type=Path,
        default=None,
        help=(
            "Optional argus synthetic clip directory. Clips must contain fens + board_corners; "
            "rows are mixed into the physical training split only."
        ),
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=None,
        help=(
            "Optional target synthetic row fraction in the training set. "
            "0 keeps only physical rows, 1 keeps only synthetic rows."
        ),
    )
    parser.add_argument(
        "--combined-split-seed",
        type=int,
        default=None,
        help=(
            "If set, merge train+val annotations and randomly split 80/20 with this seed. "
            "Tests model capacity free of the cross-game held-out gap."
        ),
    )
    return parser


if __name__ == "__main__":
    main()
