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
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from data import DetrBoardDataset, load_annotation_rows, load_replay_rows, select_rows
from model import build_model, compute_losses, decode_predictions

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

    train_dataset = DetrBoardDataset(rows=train_rows, image_size=args.image_size)
    val_dataset = DetrBoardDataset(rows=val_rows, image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = build_model(
        encoder_type=args.encoder_type,
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
        num_queries=args.num_queries,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    ).to(device)
    if args.unfreeze_last_n > 0:
        model.vision_encoder.unfreeze_last_n_layers(args.unfreeze_last_n)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.1,
    )

    history: list[dict[str, float]] = []
    best_metric = (-1.0, -1.0)
    best_state: dict[str, torch.Tensor] | None = None

    try:
        initial_metrics = evaluate(
            model,
            val_loader,
            device=device,
            lambda_square=args.lambda_square,
            lambda_presence=args.lambda_presence,
        )
        best_metric = (
            initial_metrics["placed_board_exact_match"],
            initial_metrics["per_square_accuracy"],
        )
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            running_samples = 0
            for images, targets in train_loader:
                images = images.to(device)
                prepared_targets = [
                    {
                        "piece_types": target["piece_types"].to(device),
                        "square_indices": target["square_indices"].to(device),
                    }
                    for target in targets
                ]
                outputs = model(images)
                loss, loss_terms = compute_losses(
                    outputs,
                    prepared_targets,
                    lambda_square=args.lambda_square,
                    lambda_presence=args.lambda_presence,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * int(images.shape[0])
                running_samples += int(images.shape[0])
            scheduler.step()

            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                lambda_square=args.lambda_square,
                lambda_presence=args.lambda_presence,
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": running_loss / max(running_samples, 1),
                    **val_metrics,
                }
            )
            print(
                f"epoch {epoch}: "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_loss={val_metrics['loss_total']:.4f} "
                f"val_board_exact={val_metrics['placed_board_exact_match']:.4f} "
                f"val_square_acc={val_metrics['per_square_accuracy']:.4f}"
            )
            current_metric = (
                val_metrics["placed_board_exact_match"],
                val_metrics["per_square_accuracy"],
            )
            if current_metric > best_metric:
                best_metric = current_metric
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
        "architecture": "study_detr_minimal",
        "state_dict": best_state,
        "encoder_config": {
            "encoder_type": args.encoder_type,
            "model_name": args.model_name,
            "frozen": args.freeze_encoder,
            "unfreeze_last_n": args.unfreeze_last_n,
        },
        **model.checkpoint_config(),
        "image_size": args.image_size,
        "lambda_square": args.lambda_square,
        "lambda_presence": args.lambda_presence,
        "seed": args.seed,
        "source": args.source,
        "best_val_board_exact_match": best_metric[0],
        "best_val_per_square_accuracy": best_metric[1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / "detr_minimal.pt"
    torch.save(checkpoint, checkpoint_path)
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
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


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def evaluate(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    lambda_square: float,
    lambda_presence: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    board_total = 0
    board_exact = 0
    square_total = 0
    square_correct = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            prepared_targets = [
                {
                    "piece_types": target["piece_types"].to(device),
                    "square_indices": target["square_indices"].to(device),
                }
                for target in targets
            ]
            outputs = model(images)
            loss, loss_terms = compute_losses(
                outputs,
                prepared_targets,
                lambda_square=lambda_square,
                lambda_presence=lambda_presence,
            )
            total_loss += float(loss.item()) * int(images.shape[0])
            decoded = decode_predictions(outputs)
            for prediction, target in zip(decoded, targets):
                gt_labels = [0] * 64
                for piece_type, square_index in zip(
                    target["piece_types"].tolist(), target["square_indices"].tolist()
                ):
                    from data import square_output_index_to_board_index

                    board_index = square_output_index_to_board_index(int(square_index))
                    if board_index is not None:
                        gt_labels[board_index] = int(piece_type)
                predicted_labels = prediction["board_labels"]
                board_total += 1
                if tuple(gt_labels) == tuple(predicted_labels):
                    board_exact += 1
                square_total += 64
                square_correct += sum(int(p == g) for p, g in zip(predicted_labels, gt_labels))
    return {
        "loss_total": total_loss / max(board_total, 1),
        "placed_board_exact_match": board_exact / max(board_total, 1),
        "per_square_accuracy": square_correct / max(square_total, 1),
    }


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the minimal DETR study.")
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
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lambda-square", type=float, default=2.0)
    parser.add_argument("--lambda-presence", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    main()
