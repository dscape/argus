#!/usr/bin/env python3
# ruff: noqa: E402
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

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from data import (
    CROP_MODES,
    PROJECTED_CROP_MODE,
    TYPE_CLASS_NAMES,
    BaseHeadSquareDataset,
    infer_square_labels,
    load_annotation_rows,
    load_replay_rows,
    select_rows,
)
from model import build_model

from argus.device import resolve_device


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(resolve_device(args.device))
    output_dir = (args.output_dir or (_THIS_DIR / "models" / timestamp_slug())).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = load_rows(args)
    train_rows, val_rows = split_rows(rows, val_fraction=args.val_fraction, seed=args.seed)
    if not train_rows or not val_rows:
        raise ValueError(
            f"Need non-empty train/val splits, got train={len(train_rows)} val={len(val_rows)}"
        )

    train_dataset = BaseHeadSquareDataset(
        rows=train_rows,
        input_size=args.input_size,
        piece_height=args.piece_height,
        augment=args.augment,
        body_repeat_factor=args.body_repeat_factor,
        body_overlap_threshold=args.body_overlap_threshold,
        crop_mode=args.crop_mode,
    )
    val_dataset = BaseHeadSquareDataset(
        rows=val_rows,
        input_size=args.input_size,
        piece_height=args.piece_height,
        augment=False,
        body_repeat_factor=1,
        body_overlap_threshold=args.body_overlap_threshold,
        crop_mode=args.crop_mode,
    )

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

    model = build_model(
        encoder_type=args.encoder_type,
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
        num_type_classes=len(TYPE_CLASS_NAMES),
    ).to(device)
    if args.unfreeze_last_n > 0:
        model.vision_encoder.unfreeze_last_n_layers(args.unfreeze_last_n)

    type_criterion = nn.CrossEntropyLoss()
    base_positive_fraction = train_dataset.base_positive_fraction()
    positive = max(base_positive_fraction, 1e-6)
    negative = max(1.0 - base_positive_fraction, 1e-6)
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    base_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.1,
    )

    history: list[dict[str, float]] = []
    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    try:
        initial_metrics = evaluate(
            model,
            val_loader,
            device=device,
            type_criterion=type_criterion,
            base_criterion=base_criterion,
            lambda_base=args.lambda_base,
        )
        best_metric = initial_metrics["decision_accuracy"]
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            for images, type_targets, base_targets in train_loader:
                images = images.to(device)
                type_targets = type_targets.to(device)
                base_targets = base_targets.to(device)
                type_logits, base_logits = model(images)
                loss_type = type_criterion(type_logits, type_targets)
                loss_base = base_criterion(base_logits.squeeze(-1), base_targets)
                loss = loss_type + args.lambda_base * loss_base
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = infer_square_labels(type_logits.detach(), base_logits.detach())
                running_correct += int(
                    (predicted == square_targets_from_types(type_targets, base_targets))
                    .sum()
                    .item()
                )
                running_total += int(images.shape[0])
                running_loss += float(loss.item()) * int(images.shape[0])

            scheduler.step()
            train_loss = running_loss / max(running_total, 1)
            train_accuracy = running_correct / max(running_total, 1)
            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                type_criterion=type_criterion,
                base_criterion=base_criterion,
                lambda_base=args.lambda_base,
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_decision_accuracy": train_accuracy,
                    **val_metrics,
                }
            )
            print(
                f"epoch {epoch}: "
                f"train_loss={train_loss:.4f} "
                f"train_decision_acc={train_accuracy:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_decision_acc={val_metrics['decision_accuracy']:.4f} "
                f"val_type_acc={val_metrics['type_accuracy']:.4f} "
                f"val_base_acc={val_metrics['base_accuracy']:.4f}"
            )
            if val_metrics["decision_accuracy"] > best_metric:
                best_metric = val_metrics["decision_accuracy"]
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
    finally:
        train_dataset.close()
        val_dataset.close()

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }

    checkpoint = {
        "architecture": "study_base_head",
        "state_dict": best_state,
        "encoder_config": {
            "encoder_type": args.encoder_type,
            "model_name": args.model_name,
            "frozen": args.freeze_encoder,
            "unfreeze_last_n": args.unfreeze_last_n,
        },
        **model.checkpoint_config(),
        "input_size": args.input_size,
        "piece_height": args.piece_height,
        "body_overlap_threshold": args.body_overlap_threshold,
        "body_repeat_factor": args.body_repeat_factor,
        "crop_mode": args.crop_mode,
        "lambda_base": args.lambda_base,
        "seed": args.seed,
        "source": args.source,
        "best_val_decision_accuracy": best_metric,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / "base_head.pt"
    torch.save(checkpoint, checkpoint_path)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "history": str(history_path.relative_to(_PROJECT_ROOT)),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "best_val_decision_accuracy": best_metric,
        "crop_mode": args.crop_mode,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {checkpoint_path}")


def load_rows(args: argparse.Namespace) -> list:
    if args.source == "replay":
        rows = load_replay_rows(
            clips_dir=args.clips_dir,
            eval_root=args.eval_root,
            frame_stride=args.frame_stride,
            max_frames=None,
            seed=args.seed,
            exclude_move_neighborhood=args.exclude_move_neighborhood,
        )
    elif args.source == "annotations":
        if args.annotation_root is None:
            raise ValueError("--annotation-root is required when --source=annotations")
        rows = load_annotation_rows(args.annotation_root)
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    selected = select_rows(rows, max_rows=args.max_rows, seed=args.seed)
    print(f"loaded {len(selected)} rows from source={args.source}")
    return selected


def split_rows(rows: list, *, val_fraction: float, seed: int) -> tuple[list, list]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rows_by_group: dict[str, list] = {}
    for row in rows:
        group = row.clip_path or row.source_video_id or row.row_id
        rows_by_group.setdefault(group, []).append(row)
    groups = list(rows_by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    cutoff = max(1, int(round(len(groups) * (1.0 - val_fraction))))
    cutoff = min(cutoff, len(groups) - 1)
    train_groups = set(groups[:cutoff])
    train_rows = [
        row
        for group, grouped_rows in rows_by_group.items()
        if group in train_groups
        for row in grouped_rows
    ]
    val_rows = [
        row
        for group, grouped_rows in rows_by_group.items()
        if group not in train_groups
        for row in grouped_rows
    ]
    return train_rows, val_rows


def square_targets_from_types(
    type_targets: torch.Tensor, base_targets: torch.Tensor
) -> torch.Tensor:
    predicted = torch.zeros_like(type_targets)
    keep_mask = (
        (type_targets > 0) & (type_targets < len(TYPE_CLASS_NAMES) - 1) & (base_targets > 0.5)
    )
    predicted[keep_mask] = type_targets[keep_mask]
    return predicted


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    type_criterion: nn.Module,
    base_criterion: nn.Module,
    lambda_base: float,
) -> dict[str, float]:
    model.eval()
    total = 0
    running_loss = 0.0
    type_correct = 0
    base_correct = 0
    decision_correct = 0
    with torch.no_grad():
        for images, type_targets, base_targets in loader:
            images = images.to(device)
            type_targets = type_targets.to(device)
            base_targets = base_targets.to(device)
            type_logits, base_logits = model(images)
            loss_type = type_criterion(type_logits, type_targets)
            loss_base = base_criterion(base_logits.squeeze(-1), base_targets)
            loss = loss_type + lambda_base * loss_base
            running_loss += float(loss.item()) * int(images.shape[0])
            total += int(images.shape[0])
            type_predictions = type_logits.argmax(dim=-1)
            base_predictions = (torch.sigmoid(base_logits).squeeze(-1) > 0.5).to(base_targets.dtype)
            decision_predictions = infer_square_labels(type_logits, base_logits)
            decision_targets = square_targets_from_types(type_targets, base_targets)
            type_correct += int((type_predictions == type_targets).sum().item())
            base_correct += int((base_predictions == base_targets).sum().item())
            decision_correct += int((decision_predictions == decision_targets).sum().item())
    return {
        "loss": running_loss / max(total, 1),
        "type_accuracy": type_correct / max(total, 1),
        "base_accuracy": base_correct / max(total, 1),
        "decision_accuracy": decision_correct / max(total, 1),
    }


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the base-head square classifier study.")
    parser.add_argument("--source", choices=("replay", "annotations"), default="replay")
    parser.add_argument(
        "--clips-dir", type=Path, default=_PROJECT_ROOT / "data" / "argus" / "train_real"
    )
    parser.add_argument(
        "--eval-root", type=Path, default=_PROJECT_ROOT / "data" / "physical" / "val"
    )
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--exclude-move-neighborhood", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--encoder-type", type=str, default="dinov2")
    parser.add_argument("--model-name", type=str, default=None)
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    parser.add_argument("--train-encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--piece-height", type=float, default=2.0)
    parser.add_argument("--crop-mode", choices=CROP_MODES, default=PROJECTED_CROP_MODE)
    parser.add_argument("--body-overlap-threshold", type=float, default=0.08)
    parser.add_argument("--body-repeat-factor", type=int, default=6)
    parser.add_argument("--lambda-base", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    main()
