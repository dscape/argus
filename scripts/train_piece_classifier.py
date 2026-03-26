#!/usr/bin/env python3
"""Train the DINOv2-based piece classifier on synthetic + chess-positions data.

Usage:
    .venv/bin/python scripts/train_piece_classifier.py [--epochs N] [--samples N]
    .venv/bin/python scripts/train_piece_classifier.py \\
        --chess-positions-dir data/chess_positions/train

Supports two training modes:
- **Feature-cached** (default): pre-compute DINOv2 embeddings once, then train
  the MLP head on cached 768-D vectors.  Extremely fast (~seconds per epoch).
- **Full forward**: pass images through the frozen encoder each epoch (slower,
  useful for debugging).  Enable with ``--no-cache-features``.

Follows the same version-control pattern as pipeline/screen/ai_train.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
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
    MODEL_CODE_VERSION,
    NUM_CLASSES,
    WEIGHTS_DIR,
    PieceClassifier,
)
from pipeline.overlay.piece_classifier_data import (
    CLASS_NAMES,
    DATASET_DIR,
    generate_dataset,
    load_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image → tensor helpers (used in non-cached mode)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def _train_on_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict, float]:
    """Train the MLP head on pre-computed 768-D features.  Returns (best_state, best_val_acc)."""
    n = len(features)
    split = int(n * 0.8)

    # Shuffle deterministically
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    features, labels = features[perm], labels[perm]

    train_ds = TensorDataset(features[:split], labels[:split])
    val_ds = TensorDataset(features[split:], labels[split:])
    logger.info("Train: %d, Val: %d (feature-cached mode)", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build only the head (no encoder needed)
    model = PieceClassifier()
    head = model.head.to(device)
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = head(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = head(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f",
            epoch + 1,
            epochs,
            train_loss / max(train_total, 1),
            train_acc,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"head": {k: v.cpu().clone() for k, v in head.state_dict().items()}}
            logger.info("  → New best (val_acc=%.4f)", val_acc)

    # Per-class accuracy
    if best_state is not None:
        head.load_state_dict(best_state["head"])
    head.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = head(x).argmax(1)
            for p, t in zip(preds.cpu().tolist(), y.cpu().tolist()):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1

    logger.info("Per-class val accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        acc = class_correct[i] / max(class_total[i], 1)
        logger.info("  %6s: %d/%d (%.1f%%)", name, class_correct[i], class_total[i], acc * 100)

    return best_state, best_val_acc, len(train_ds), len(val_ds)


def _train_full_forward(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict, float, int, int]:
    """Train with full DINOv2 forward pass each epoch (original mode)."""
    n = len(images)
    split = int(n * 0.8)
    train_ds = _to_tensor_dataset(images[:split], labels[:split])
    val_ds = _to_tensor_dataset(images[split:], labels[split:])
    logger.info("Train: %d, Val: %d (full forward mode)", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = PieceClassifier().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(epochs):
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
            epochs,
            train_loss / max(train_total, 1),
            train_acc,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"head": {k: v.cpu().clone() for k, v in model.head.state_dict().items()}}
            logger.info("  → New best (val_acc=%.4f)", val_acc)

    # Per-class accuracy
    if best_state is not None:
        model.head.load_state_dict(best_state["head"])
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

    return best_state, best_val_acc, len(train_ds), len(val_ds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train piece classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=400, help="Synthetic samples per class")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of the synthetic dataset even if cached",
    )
    # Chess positions integration
    parser.add_argument(
        "--chess-positions-dir",
        type=str,
        default=None,
        help="Path to chess-positions train/ directory (e.g. data/chess_positions/train)",
    )
    parser.add_argument(
        "--chess-positions-samples",
        type=int,
        default=1500,
        help="Max chess-positions samples per class",
    )
    # Feature caching
    parser.add_argument(
        "--cache-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use feature caching for fast training (default: on)",
    )
    parser.add_argument(
        "--recache",
        action="store_true",
        help="Force re-extraction of cached features",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # -------------------------------------------------------------------
    # Load or generate synthetic dataset
    # -------------------------------------------------------------------
    cached = None if args.regenerate else load_dataset(DATASET_DIR)

    if cached is not None:
        synth_images, synth_labels = cached
        logger.info(
            "Using cached synthetic dataset: %d images from %s", len(synth_images), DATASET_DIR
        )
    else:
        logger.info(
            "Generating %d synthetic samples per class (%d total)…",
            args.samples,
            args.samples * NUM_CLASSES,
        )
        synth_images, synth_labels = generate_dataset(
            num_samples_per_class=args.samples,
            size=128,
            seed=args.seed,
        )

    logger.info("Synthetic: %d images, shape=%s", len(synth_images), synth_images.shape)

    # -------------------------------------------------------------------
    # Load chess-positions dataset (optional)
    # -------------------------------------------------------------------
    cp_images: np.ndarray | None = None
    cp_labels: np.ndarray | None = None

    if args.chess_positions_dir:
        from pipeline.overlay.chess_positions_data import sample_chess_positions_squares

        cp_dir = Path(args.chess_positions_dir)
        if not cp_dir.exists():
            logger.error("Chess positions directory not found: %s", cp_dir)
            sys.exit(1)

        cp_images, cp_labels = sample_chess_positions_squares(
            cp_dir,
            max_per_class=args.chess_positions_samples,
            size=128,
            seed=args.seed,
        )
        logger.info("Chess positions: %d images, shape=%s", len(cp_images), cp_images.shape)

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    if args.cache_features:
        from pipeline.overlay.feature_cache import (
            extract_and_cache_features,
            load_cached_features,
        )

        # Cache synthetic features
        synth_cached = None if args.recache else load_cached_features("synthetic")
        if synth_cached is None:
            extract_and_cache_features(
                synth_images,
                synth_labels,
                "synthetic",
                device=str(device),
                batch_size=args.batch_size,
            )
            synth_cached = load_cached_features("synthetic")
        synth_feats, synth_labs = synth_cached

        # Cache chess-positions features
        if cp_images is not None:
            cp_cached = None if args.recache else load_cached_features("chess_positions")
            if cp_cached is None:
                extract_and_cache_features(
                    cp_images,
                    cp_labels,
                    "chess_positions",
                    device=str(device),
                    batch_size=args.batch_size,
                )
                cp_cached = load_cached_features("chess_positions")
            cp_feats, cp_labs = cp_cached
            all_features = torch.cat([synth_feats, cp_feats], dim=0)
            all_labels = torch.cat([synth_labs, cp_labs], dim=0)
        else:
            all_features = synth_feats
            all_labels = synth_labs

        logger.info("Total features: %d", len(all_features))
        best_state, best_val_acc, train_size, val_size = _train_on_features(
            all_features,
            all_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
    else:
        # Full forward mode — combine images
        if cp_images is not None:
            all_images = np.concatenate([synth_images, cp_images], axis=0)
            all_labels_np = np.concatenate([synth_labels, cp_labels], axis=0)
        else:
            all_images = synth_images
            all_labels_np = synth_labels

        logger.info("Total images: %d", len(all_images))
        best_state, best_val_acc, train_size, val_size = _train_full_forward(
            all_images,
            all_labels_np,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    # -------------------------------------------------------------------
    # Version control — match pipeline/screen/ai_train.py pattern
    # -------------------------------------------------------------------
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = WEIGHTS_DIR / "metadata.json"
    revision = 1
    if meta_path.exists():
        with open(meta_path) as f:
            old_meta = json.load(f)
        if old_meta.get("code_version") == MODEL_CODE_VERSION:
            revision = old_meta.get("revision", 0) + 1

    version_str = f"{MODEL_CODE_VERSION}r{revision}"

    metadata = {
        "code_version": MODEL_CODE_VERSION,
        "revision": revision,
        "version": version_str,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": args.epochs,
        "train_size": train_size,
        "val_size": val_size,
        "seed": args.seed,
        "chess_positions": args.chess_positions_dir is not None,
        "feature_cached": args.cache_features,
    }

    versioned_path = WEIGHTS_DIR / f"{version_str}.pt"
    torch.save(best_state, versioned_path)
    torch.save(best_state, WEIGHTS_DIR / "best.pt")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("  Model version: %s", version_str)
    logger.info("  Weights: %s", versioned_path)


if __name__ == "__main__":
    main()
