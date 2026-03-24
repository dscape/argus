"""Evaluate the screening classifier and calibrate confidence thresholds.

Provides per-class precision/recall/F1 and finds the confidence threshold
that achieves a target precision for auto-deciding videos.
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.screen.ai_classifier import CLASS_NAMES, NUM_CLASSES, ScreeningClassifier
from pipeline.screen.ai_train import CHECKPOINT_DIR, build_datasets

logger = logging.getLogger(__name__)


def evaluate(checkpoint_path: str | None = None) -> dict:
    """Run evaluation on the validation set.

    Returns dict with per-class metrics and overall accuracy.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    _, val_ds = build_datasets()
    if len(val_ds) == 0:
        print("No validation data.")
        return {}

    model = ScreeningClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for emb, scan, otb, labels in val_loader:
            logits = model(emb, scan, otb)
            probs = F.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_confidences.extend(confs.tolist())

    # Compute per-class metrics
    results = {"classes": {}}
    total_correct = 0

    for cls_idx in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_idx]
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l == cls_idx)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l != cls_idx)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != cls_idx and l == cls_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_correct += tp

        results["classes"][cls_name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
        }

    results["accuracy"] = round(total_correct / len(all_labels), 3) if all_labels else 0.0
    results["total"] = len(all_labels)

    # Print
    print(f"\nEvaluation ({len(all_labels)} samples)")
    print(f"  Overall accuracy: {results['accuracy']:.3f}")
    print()
    print(f"  {'Class':<12} {'Prec':>6} {'Recall':>6} {'F1':>6}   TP    FP    FN")
    print(f"  {'-'*60}")
    for cls_name, m in results["classes"].items():
        print(f"  {cls_name:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}   "
              f"{m['tp']:<5} {m['fp']:<5} {m['fn']:<5}")

    return results


def calibrate_threshold(
    checkpoint_path: str | None = None,
    target_precision: float = 0.95,
) -> dict:
    """Find confidence thresholds that achieve target precision per class.

    Sweeps thresholds and reports the auto-decision rate at each.
    Returns optimal threshold and expected auto-decision rate.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    _, val_ds = build_datasets()
    if len(val_ds) == 0:
        print("No validation data.")
        return {}

    model = ScreeningClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for emb, scan, otb, labels in val_loader:
            logits = model(emb, scan, otb)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
            all_labels.extend(labels.tolist())

    all_probs = torch.cat(all_probs, dim=0)  # (N, 3)
    all_labels_t = torch.tensor(all_labels)

    # Sweep thresholds
    thresholds = [t / 100.0 for t in range(50, 100)]
    print(f"\nThreshold calibration (target precision >= {target_precision:.2f})")
    print(f"  {'Thresh':>7} {'Auto%':>6} {'Prec':>6} {'Correct':>8} {'Wrong':>6} {'Skipped':>8}")
    print(f"  {'-'*50}")

    best_threshold = 0.95
    best_auto_rate = 0.0

    for thresh in thresholds:
        confidences, preds = all_probs.max(dim=-1)
        auto_mask = confidences >= thresh

        if not auto_mask.any():
            continue

        auto_preds = preds[auto_mask]
        auto_labels = all_labels_t[auto_mask]

        correct = (auto_preds == auto_labels).sum().item()
        wrong = auto_preds.size(0) - correct
        precision = correct / auto_preds.size(0) if auto_preds.size(0) > 0 else 0.0
        auto_rate = auto_mask.sum().item() / len(all_labels)

        if thresh in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]:
            print(f"  {thresh:>7.2f} {auto_rate:>5.1%} {precision:>6.3f} {correct:>8} {wrong:>6} "
                  f"{len(all_labels) - auto_mask.sum().item():>8}")

        if precision >= target_precision and auto_rate > best_auto_rate:
            best_auto_rate = auto_rate
            best_threshold = thresh

    print(f"\n  Recommended threshold: {best_threshold:.2f}")
    print(f"  Expected auto-decision rate: {best_auto_rate:.1%}")
    print(f"  Expected precision at threshold: >= {target_precision:.2f}")

    return {
        "threshold": best_threshold,
        "auto_decision_rate": round(best_auto_rate, 3),
        "target_precision": target_precision,
    }
