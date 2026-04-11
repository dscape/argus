#!/usr/bin/env python3
"""Train and export the tiny ONNX square classifier for overlay board reading.

Usage:
    .venv/bin/python scripts/train_piece_classifier.py
    .venv/bin/python scripts/train_piece_classifier.py --device mps --epochs 12
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.chess_positions_data import sample_chess_positions_squares
from pipeline.overlay.piece_classifier_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    augment_square_image,
    generate_dataset,
)
from pipeline.overlay.real_board_data import sample_real_board_squares
from pipeline.overlay.square_classifier_model import (
    INPUT_SIZE,
    MODEL_CODE_VERSION,
    TinySquareClassifier,
)

from argus.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "overlay"
DEFAULT_CP_TRAIN_DIR = _PROJECT_ROOT / "data" / "overlay" / "train"
DEFAULT_CP_VAL_DIR = _PROJECT_ROOT / "data" / "overlay" / "val"
DEFAULT_REAL_BOARD_TRAIN_DIR = _PROJECT_ROOT / "data" / "overlay" / "val_real"


def _to_tensor_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """Convert uint8 BGR images to an in-memory float tensor dataset."""
    rgb = np.ascontiguousarray(images[:, :, :, ::-1])
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(labels.astype(np.int64, copy=False))
    return TensorDataset(x, y)


def _load_synthetic_splits(
    *,
    train_samples_per_class: int,
    val_samples_per_class: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    train_images, train_labels = generate_dataset(
        num_samples_per_class=train_samples_per_class,
        size=INPUT_SIZE,
        seed=seed,
        output_dir=_PROJECT_ROOT / "outputs" / "overlay_classifier" / "synthetic_train_cache",
    )
    val_images, val_labels = generate_dataset(
        num_samples_per_class=val_samples_per_class,
        size=INPUT_SIZE,
        seed=seed + 1,
        output_dir=_PROJECT_ROOT / "outputs" / "overlay_classifier" / "synthetic_val_cache",
    )
    return (train_images, train_labels), (val_images, val_labels)


def _load_chess_positions_split(
    data_dir: Path | None,
    *,
    max_per_class: int,
    empty_multiplier: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if data_dir is None:
        return None
    if not data_dir.exists():
        logger.warning("Chess-positions directory not found: %s", data_dir)
        return None

    targets = {class_index: max_per_class for class_index in range(NUM_CLASSES)}
    targets[0] = max_per_class * empty_multiplier

    images, labels = sample_chess_positions_squares(
        data_dir,
        max_per_class=targets,
        size=INPUT_SIZE,
        seed=seed,
    )
    return images, labels


def _load_real_board_train_split(
    data_dir: Path | None,
    *,
    seed: int,
    augment_copies: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if data_dir is None:
        return None
    if not data_dir.exists():
        logger.warning("Real board directory not found: %s", data_dir)
        return None

    images, labels = sample_real_board_squares(
        data_dir,
        max_per_class=None,
        size=INPUT_SIZE,
        seed=seed,
    )
    if augment_copies <= 0:
        return images, labels

    augmented_images: list[np.ndarray] = [img for img in images]
    augmented_labels: list[int] = labels.astype(np.int64, copy=False).tolist()
    for index, (image, label) in enumerate(zip(images, labels.tolist())):
        for copy_index in range(augment_copies):
            aug_seed = seed + index * 997 + copy_index * 7919
            rng = random.Random(aug_seed)
            augmented_images.append(augment_square_image(image.copy(), rng))
            augmented_labels.append(int(label))

    augmented_arr = np.array(augmented_images, dtype=np.uint8)
    labels_arr = np.array(augmented_labels, dtype=np.int64)
    perm = np.random.RandomState(seed).permutation(len(augmented_arr))
    return augmented_arr[perm], labels_arr[perm]


def _concat_splits(
    parts: list[tuple[np.ndarray, np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray]:
    valid = [part for part in parts if part is not None]
    if not valid:
        raise ValueError("No training data loaded")
    images = np.concatenate([part[0] for part in valid], axis=0)
    labels = np.concatenate([part[1] for part in valid], axis=0)
    return images, labels


def _build_loaders(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    *,
    batch_size: int,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    train_loader = DataLoader(
        _to_tensor_dataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        _to_tensor_dataset(val_images, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def _evaluate(
    model: TinySquareClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total = 0
    correct = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()
            for pred, target in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                class_total[target] += 1
                if pred == target:
                    class_correct[target] += 1
    return correct / max(total, 1), class_correct, class_total


def _train(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    device: torch.device,
) -> tuple[TinySquareClassifier, float]:
    model = TinySquareClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        val_acc, _, _ = _evaluate(model, val_loader, device=device)
        train_acc = train_correct / max(train_total, 1)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_acc=%.4f",
            epoch + 1,
            epochs,
            train_loss / max(train_total, 1),
            train_acc,
            val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            logger.info("  -> New best (val_acc=%.4f)", val_acc)

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")

    best_model = TinySquareClassifier()
    best_model.load_state_dict(best_state)
    best_model.to(device)
    best_model.eval()
    return best_model, best_val_acc


def _export_onnx(model: TinySquareClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _log_per_class_accuracy(
    model: TinySquareClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> None:
    val_acc, class_correct, class_total = _evaluate(model, loader, device=device)
    logger.info("Final val accuracy: %.4f", val_acc)
    logger.info("Per-class val accuracy:")
    for class_index, name in enumerate(CLASS_NAMES):
        acc = class_correct[class_index] / max(class_total[class_index], 1)
        logger.info(
            "  %6s: %d/%d (%.1f%%)",
            name,
            class_correct[class_index],
            class_total[class_index],
            acc * 100,
        )


def _next_version() -> tuple[int, str]:
    meta_path = WEIGHTS_DIR / "metadata.json"
    revision = 1
    if meta_path.exists():
        with meta_path.open() as handle:
            old_meta = json.load(handle)
        if old_meta.get("code_version") == MODEL_CODE_VERSION:
            revision = int(old_meta.get("revision", 0)) + 1
    return revision, f"{MODEL_CODE_VERSION}r{revision}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train overlay square classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--synthetic-train-samples", type=int, default=1200)
    parser.add_argument("--synthetic-val-samples", type=int, default=300)
    parser.add_argument("--chess-positions-train-dir", type=Path, default=DEFAULT_CP_TRAIN_DIR)
    parser.add_argument("--chess-positions-val-dir", type=Path, default=DEFAULT_CP_VAL_DIR)
    parser.add_argument("--chess-positions-train-samples", type=int, default=2500)
    parser.add_argument("--chess-positions-val-samples", type=int, default=1200)
    parser.add_argument("--chess-positions-empty-multiplier", type=int, default=6)
    parser.add_argument("--real-board-train-dir", type=Path, default=None)
    parser.add_argument("--real-board-augment-copies", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(resolve_device(args.device))

    logger.info("Loading datasets at %dx%d…", INPUT_SIZE, INPUT_SIZE)
    synthetic_train, synthetic_val = _load_synthetic_splits(
        train_samples_per_class=args.synthetic_train_samples,
        val_samples_per_class=args.synthetic_val_samples,
        seed=args.seed,
    )
    cp_train = _load_chess_positions_split(
        args.chess_positions_train_dir,
        max_per_class=args.chess_positions_train_samples,
        empty_multiplier=args.chess_positions_empty_multiplier,
        seed=args.seed,
    )
    cp_val = _load_chess_positions_split(
        args.chess_positions_val_dir,
        max_per_class=args.chess_positions_val_samples,
        empty_multiplier=args.chess_positions_empty_multiplier,
        seed=args.seed,
    )
    real_train = _load_real_board_train_split(
        args.real_board_train_dir,
        seed=args.seed,
        augment_copies=args.real_board_augment_copies,
    )

    train_images, train_labels = _concat_splits([synthetic_train, cp_train, real_train])
    val_images, val_labels = _concat_splits([synthetic_val, cp_val])

    logger.info("Train: %d images", len(train_images))
    logger.info("Val:   %d images", len(val_images))

    train_loader, val_loader = _build_loaders(
        train_images,
        train_labels,
        val_images,
        val_labels,
        batch_size=args.batch_size,
    )

    model, best_val_acc = _train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        device=device,
    )
    _log_per_class_accuracy(model, val_loader, device=device)

    revision, version_str = _next_version()
    best_path = WEIGHTS_DIR / "best.onnx"
    versioned_path = WEIGHTS_DIR / f"{version_str}.onnx"

    _export_onnx(model, best_path)
    _export_onnx(model, versioned_path)

    metadata = {
        "code_version": MODEL_CODE_VERSION,
        "revision": revision,
        "version": version_str,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": args.epochs,
        "train_size": int(len(train_images)),
        "val_size": int(len(val_images)),
        "seed": args.seed,
        "input_size": INPUT_SIZE,
        "runtime_format": "onnx",
        "architecture": "tiny_square_cnn",
        "sources": {
            "synthetic_train_per_class": args.synthetic_train_samples,
            "synthetic_val_per_class": args.synthetic_val_samples,
            "chess_positions_train_per_piece_class": args.chess_positions_train_samples,
            "chess_positions_val_per_piece_class": args.chess_positions_val_samples,
            "chess_positions_empty_multiplier": args.chess_positions_empty_multiplier,
            "real_board_train_dir": (
                str(args.real_board_train_dir) if args.real_board_train_dir else None
            ),
            "real_board_augment_copies": args.real_board_augment_copies,
            "label_smoothing": args.label_smoothing,
        },
    }
    with (WEIGHTS_DIR / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("  Model version: %s", version_str)
    logger.info("  Weights: %s", versioned_path)


if __name__ == "__main__":
    main()
