#!/usr/bin/env python3
"""Train a direct physical move model on synthetic + pseudo-real clips."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.move_data import build_real_move_window_clips

from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset
from argus.data.transforms import TemporalAugmentation, ValidationTransform
from argus.device import resolve_device
from argus.model.argus_model import ArgusModel
from argus.training.trainer import Trainer, compute_num_optimizer_steps

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYNTHETIC_ROOT = _PROJECT_ROOT / "data" / "argus" / "training_dataset"
_DEFAULT_REAL_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_move_model"
_DEFAULT_DINO_MODEL = "facebook/dinov2-base"


class MoveLossMaskDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        *,
        loss_mode: str,
        move_frame_loss_weight: float,
        no_move_loss_weight: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.loss_mode = loss_mode
        self.move_frame_loss_weight = float(move_frame_loss_weight)
        self.no_move_loss_weight = float(no_move_loss_weight)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = dict(self.base_dataset[index])
        move_mask = sample["move_mask"].to(torch.bool)
        if self.loss_mode == "move_frames":
            sample["move_loss_mask"] = move_mask.clone()
            return sample

        sample["move_loss_mask"] = torch.ones_like(move_mask, dtype=torch.bool)
        loss_weights = torch.full_like(
            sample["detect_targets"],
            fill_value=self.no_move_loss_weight,
            dtype=torch.float32,
        )
        loss_weights = torch.where(
            move_mask,
            torch.full_like(loss_weights, self.move_frame_loss_weight),
            loss_weights,
        )
        sample["move_loss_weights"] = loss_weights
        return sample


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = TemporalAugmentation(
        color_jitter=args.augment,
        random_erasing=args.augment,
        normalize=True,
    )
    val_transform = ValidationTransform()

    train_parts: list[Dataset] = []
    val_dataset: Dataset | None = None

    if args.use_synthetic:
        train_parts.append(
            MoveLossMaskDataset(
                ArgusDataset(
                    args.synthetic_root / "train",
                    clip_length=args.clip_length,
                    transform=train_transform,
                ),
                loss_mode=args.train_move_loss_mode,
                move_frame_loss_weight=args.move_frame_loss_weight,
                no_move_loss_weight=args.no_move_loss_weight,
            )
        )
        if not args.real_val_source_videos:
            val_dataset = MoveLossMaskDataset(
                ArgusDataset(
                    args.synthetic_root / "val",
                    clip_length=args.clip_length,
                    transform=val_transform,
                ),
                loss_mode=args.train_move_loss_mode,
                move_frame_loss_weight=args.move_frame_loss_weight,
                no_move_loss_weight=args.no_move_loss_weight,
            )

    real_val_source_video_ids = {
        value.strip() for value in args.real_val_source_videos.split(",") if value.strip()
    }
    if args.use_real:
        real_train_clips, real_train_meta = build_real_move_window_clips(
            clips_dir=args.real_clips_dir,
            image_size=args.image_size,
            clip_length=args.clip_length,
            negative_window_stride=args.real_negative_window_stride,
            max_negative_windows_per_clip=args.real_max_negative_windows_per_clip,
            selection_source_video_ids=(
                None if not real_val_source_video_ids else real_val_source_video_ids
            ),
            exclude_selection_source_video_ids=True,
        )
        if real_train_clips:
            train_parts.append(
                MoveLossMaskDataset(
                    ArgusInMemoryDataset(
                        clips=real_train_clips,
                        clip_length=args.clip_length,
                        transform=train_transform,
                    ),
                    loss_mode=args.train_move_loss_mode,
                    move_frame_loss_weight=args.move_frame_loss_weight,
                    no_move_loss_weight=args.no_move_loss_weight,
                )
            )
        real_val_clips, real_val_meta = build_real_move_window_clips(
            clips_dir=args.real_clips_dir,
            image_size=args.image_size,
            clip_length=args.clip_length,
            negative_window_stride=args.real_negative_window_stride,
            max_negative_windows_per_clip=args.real_max_negative_windows_per_clip,
            selection_source_video_ids=(
                None if not real_val_source_video_ids else real_val_source_video_ids
            ),
            exclude_selection_source_video_ids=False,
        )
        if real_val_clips:
            val_dataset = MoveLossMaskDataset(
                ArgusInMemoryDataset(
                    clips=real_val_clips,
                    clip_length=args.clip_length,
                    transform=val_transform,
                ),
                loss_mode=args.train_move_loss_mode,
                move_frame_loss_weight=args.move_frame_loss_weight,
                no_move_loss_weight=args.no_move_loss_weight,
            )
        (output_dir / "real_train_windows.json").write_text(
            json.dumps([meta.__dict__ for meta in real_train_meta], indent=2)
        )
        (output_dir / "real_val_windows.json").write_text(
            json.dumps([meta.__dict__ for meta in real_val_meta], indent=2)
        )

    if not train_parts:
        raise ValueError("No training data selected")
    if val_dataset is None:
        raise ValueError("No validation dataset selected")

    train_dataset: Dataset
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=physical_move_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=physical_move_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    model = ArgusModel(
        vision_encoder_name=args.model_name,
        vision_encoder_type="dinov2",
        frozen_vision=True,
        vision_feature_layer_indices=parse_feature_layer_indices(args.dino_feature_layer_indices),
        temporal_d_model=args.temporal_d_model,
        temporal_n_layers=args.temporal_layers,
        temporal_d_state=args.temporal_d_state,
        temporal_expand=args.temporal_expand,
        move_vocab_size=1970,
        pooling_type="mean",
        square_pool_size=8,
        square_head_enabled=args.square_loss_weight > 0.0,
        use_detector=False,
    )

    total_steps = (
        compute_num_optimizer_steps(
            len(train_loader),
            args.gradient_accumulation,
        )
        * args.epochs
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,
        clip_grad_norm=args.clip_grad_norm,
        gradient_accumulation=args.gradient_accumulation,
        precision=args.precision,
        loss_weights={
            "move": 1.0,
            "detect": args.detect_loss_weight,
            "square": args.square_loss_weight,
        },
        output_dir=str(output_dir),
        save_every=1,
        use_wandb=False,
        device=device,
    )

    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(epoch)
        logger.info(
            "epoch %d train move=%.4f detect=%.4f move_acc=%.4f",
            epoch,
            train_metrics.get("move", 0.0),
            train_metrics.get("detect", 0.0),
            train_metrics.get("move_acc", 0.0),
        )
        val_metrics = trainer.validate()
        logger.info(
            "epoch %d val move_acc=%.4f detect_recall=%.4f detect_precision=%.4f detect_f1=%.4f",
            epoch,
            val_metrics.get("move_accuracy", 0.0),
            val_metrics.get("move_detection_recall", 0.0),
            val_metrics.get("move_detection_precision", 0.0),
            val_metrics.get("move_detection_f1", 0.0),
        )
        trainer.save_checkpoint(epoch, val_metrics)

        selection_score = (
            val_metrics.get("move_accuracy", 0.0)
            + val_metrics.get("move_detection_recall", 0.0)
            + val_metrics.get("move_detection_f1", 0.0)
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training produced no best checkpoint")

    best_checkpoint = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "model_config": model.model_config,
        "metrics": best_metrics,
        "selection_score": best_score,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    best_path = output_dir / "best.pt"
    torch.save(best_checkpoint, best_path)
    logger.info("Saved best checkpoint to %s", best_path)

    summary = {
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "train_move_loss_mode": args.train_move_loss_mode,
        "move_frame_loss_weight": args.move_frame_loss_weight,
        "no_move_loss_weight": args.no_move_loss_weight,
        "square_loss_weight": args.square_loss_weight,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "checkpoint": str(best_path.relative_to(_PROJECT_ROOT)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train direct physical move model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--detect-loss-weight", type=float, default=0.5)
    parser.add_argument("--square-loss-weight", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--train-move-loss-mode",
        choices=("move_frames", "all_frames"),
        default="move_frames",
    )
    parser.add_argument("--move-frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-move-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--synthetic-root", type=Path, default=_DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--use-real", action="store_true")
    parser.add_argument("--real-clips-dir", type=Path, default=_DEFAULT_REAL_CLIPS_DIR)
    parser.add_argument("--real-val-source-videos", type=str, default="psrPAoHr4wA")
    parser.add_argument("--real-negative-window-stride", type=int, default=8)
    parser.add_argument("--real-max-negative-windows-per-clip", type=int, default=4)
    parser.add_argument("--model-name", type=str, default=_DEFAULT_DINO_MODEL)
    parser.add_argument("--dino-feature-layer-indices", type=str, default="8,10,11")
    parser.add_argument("--temporal-d-model", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-d-state", type=int, default=128)
    parser.add_argument("--temporal-expand", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def physical_move_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    collated = {
        "frames": torch.stack([sample["frames"] for sample in batch]),
        "move_targets": torch.stack([sample["move_targets"] for sample in batch]),
        "detect_targets": torch.stack([sample["detect_targets"] for sample in batch]),
        "legal_masks": torch.stack([sample["legal_masks"] for sample in batch]),
        "move_mask": torch.stack([sample["move_mask"] for sample in batch]),
    }
    if all("move_loss_mask" in sample for sample in batch):
        collated["move_loss_mask"] = torch.stack([sample["move_loss_mask"] for sample in batch])
    if all("move_loss_weights" in sample for sample in batch):
        collated["move_loss_weights"] = torch.stack(
            [sample["move_loss_weights"] for sample in batch]
        )
    if all("square_targets" in sample for sample in batch):
        collated["square_targets"] = torch.stack([sample["square_targets"] for sample in batch])
    return collated


def parse_feature_layer_indices(raw_value: str) -> list[int]:
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("feature layer indices must not be empty")
    return [int(value) for value in values]


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


if __name__ == "__main__":
    main()
