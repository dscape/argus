#!/usr/bin/env python3
"""Train the DINOv2-based piece classifier on synthetic data.

Usage:
    .venv/bin/python scripts/train_piece_classifier.py [--epochs N] [--samples N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.piece_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    WEIGHTS_DIR,
    PieceClassifier,
)
from pipeline.overlay.piece_classifier_data import CLASS_NAMES, generate_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _to_tensor_dataset(
    images: np.ndarray,
    labels: np.ndarray,
) -> TensorDataset:
    """Convert (N, H, W, 3) uint8 BGR images to normalised (N, 3, 224, 224) tensors."""
    import cv2

    tensors: list[torch.Tensor] = []
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        t = (t - mean) / std
        tensors.append(t)

    x = torch.stack(tensors)
    y = torch.from_numpy(labels).long()
    return TensorDataset(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=400, help="Samples per class")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Generate data
    logger.info(
        "Generating %d samples per class (%d total)…", args.samples, args.samples * NUM_CLASSES
    )
    images, labels = generate_dataset(
        output_dir="",  # not used currently
        num_samples_per_class=args.samples,
        size=128,
        seed=args.seed,
    )
    logger.info("Dataset: %d images, shape=%s", len(images), images.shape)

    # Split 80/20
    n = len(images)
    split = int(n * 0.8)
    train_ds = _to_tensor_dataset(images[:split], labels[:split])
    val_ds = _to_tensor_dataset(images[split:], labels[split:])
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = PieceClassifier().to(device)
    model.train()
    # Only train the head (encoder is frozen)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f",
            epoch + 1,
            args.epochs,
            train_loss / max(train_total, 1),
            train_acc,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = WEIGHTS_DIR / "best.pt"
            torch.save({"head": model.head.state_dict(), "val_acc": val_acc}, save_path)
            logger.info("  → Saved best model (val_acc=%.4f) to %s", val_acc, save_path)

    # Per-class accuracy
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1

    logger.info("Per-class val accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        acc = class_correct[i] / max(class_total[i], 1)
        logger.info("  %6s: %d/%d (%.1f%%)", name, class_correct[i], class_total[i], acc * 100)

    logger.info("Best val accuracy: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
