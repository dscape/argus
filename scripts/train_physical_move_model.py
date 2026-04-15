#!/usr/bin/env python3
"""Train a direct physical move model on synthetic + pseudo-real clips."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.replay import build_replay_board
from pipeline.physical.joint_board_reader import (
    argus_overrides_from_joint_board_reader_checkpoint,
    argus_square_reader_state_dict_from_joint_board_reader_checkpoint,
)
from pipeline.physical.move_data import (
    build_real_move_window_clips,
    load_eval_move_sequences,
    load_real_move_sequences,
)
from pipeline.physical.real_board_data import load_real_board_rows
from pipeline.physical.square_probe import evaluate_probe
from pipeline.shared import SQUARE_CLASS_NAMES, board_to_class_ids
from scripts.eval_physical_board_tracker import (
    FramePrediction,
    IdentityProbe,
    compute_tracker_sequence_metrics,
)
from scripts.eval_physical_move_model import decode_sequence_with_segmental_decoder

from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset
from argus.data.transforms import TemporalAugmentation, ValidationTransform
from argus.device import resolve_device
from argus.model.argus_model import ArgusModel
from argus.model.vision_encoder import default_model_name_for_encoder_type
from argus.training.trainer import Trainer, compute_num_optimizer_steps

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYNTHETIC_ROOT = _PROJECT_ROOT / "data" / "argus" / "training_dataset"
_DEFAULT_REAL_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_move_model"


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
        if "move_loss_mask" in sample:
            if self.loss_mode == "all_frames" and "move_loss_weights" not in sample:
                move_mask = sample["move_loss_mask"].to(torch.bool)
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


class RepeatDataset(Dataset):
    def __init__(self, base_dataset: Dataset, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats must be > 0, got {repeats}")
        self.base_dataset = base_dataset
        self.repeats = int(repeats)

    def __len__(self) -> int:
        return len(self.base_dataset) * self.repeats

    def __getitem__(self, index: int) -> Any:
        return self.base_dataset[index % len(self.base_dataset)]


def main() -> None:
    args = build_parser().parse_args()
    args.model_name = args.model_name or default_model_name_for_encoder_type(
        args.vision_encoder_type
    )
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
    train_component_sizes: dict[str, int] = {}
    train_component_repeats: dict[str, int] = {}
    real_train_source_video_ids: list[str] = []
    val_dataset: Dataset | None = None

    real_val_source_video_id_list = resolve_real_val_source_video_ids(args) if args.use_real else []
    real_val_source_video_ids = set(real_val_source_video_id_list)
    if real_val_source_video_id_list:
        logger.info(
            "Using internal real-val source videos for selection/validation: %s",
            ",".join(real_val_source_video_id_list),
        )

    if args.use_synthetic:
        synthetic_train_dataset = MoveLossMaskDataset(
            ArgusDataset(
                args.synthetic_root / "train",
                clip_length=args.clip_length,
                transform=train_transform,
            ),
            loss_mode=args.train_move_loss_mode,
            move_frame_loss_weight=args.move_frame_loss_weight,
            no_move_loss_weight=args.no_move_loss_weight,
        )
        train_component_sizes["synthetic"] = len(synthetic_train_dataset)
        train_component_repeats["synthetic"] = args.synthetic_train_repeat
        train_parts.append(
            RepeatDataset(synthetic_train_dataset, repeats=args.synthetic_train_repeat)
        )
        if args.observation_mode == "native_oblique":
            logger.info(
                "native_oblique training with synthetic data keeps synthetic clips in their "
                "existing synthetic view/domain; only real clips use source-video native crops"
            )
        if not real_val_source_video_ids:
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

    if args.use_real:
        real_train_clips, real_train_meta = build_real_move_window_clips(
            clips_dir=args.real_clips_dir,
            image_size=args.image_size,
            clip_length=args.clip_length,
            negative_window_stride=args.real_negative_window_stride,
            max_negative_windows_per_clip=args.real_max_negative_windows_per_clip,
            positive_window_repeat=args.real_move_positive_window_repeat,
            selection_source_video_ids=(
                None if not real_val_source_video_ids else real_val_source_video_ids
            ),
            exclude_selection_source_video_ids=True,
            observation_mode=args.observation_mode,
            move_target_pre_frames=args.move_target_pre_frames,
            detect_target_radius=args.detect_target_radius,
            detect_target_decay=args.detect_target_decay,
            oblique_crop_margin=args.oblique_crop_margin,
        )
        if real_train_clips:
            real_train_dataset = MoveLossMaskDataset(
                ArgusInMemoryDataset(
                    clips=real_train_clips,
                    clip_length=args.clip_length,
                    transform=train_transform,
                ),
                loss_mode=args.train_move_loss_mode,
                move_frame_loss_weight=args.move_frame_loss_weight,
                no_move_loss_weight=args.no_move_loss_weight,
            )
            train_component_sizes["real"] = len(real_train_dataset)
            train_component_repeats["real"] = args.real_train_repeat
            train_parts.append(RepeatDataset(real_train_dataset, repeats=args.real_train_repeat))
            real_train_source_video_ids = sorted(
                {
                    meta.source_video_id
                    for meta in real_train_meta
                    if meta.source_video_id is not None
                }
            )
        real_val_clips = []
        real_val_meta = []
        if real_val_source_video_ids:
            real_val_clips, real_val_meta = build_real_move_window_clips(
                clips_dir=args.real_clips_dir,
                image_size=args.image_size,
                clip_length=args.clip_length,
                negative_window_stride=args.real_negative_window_stride,
                max_negative_windows_per_clip=args.real_max_negative_windows_per_clip,
                selection_source_video_ids=real_val_source_video_ids,
                exclude_selection_source_video_ids=False,
                observation_mode=args.observation_mode,
                move_target_pre_frames=args.move_target_pre_frames,
                detect_target_radius=args.detect_target_radius,
                detect_target_decay=args.detect_target_decay,
                oblique_crop_margin=args.oblique_crop_margin,
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

    selection_sequences = None
    selection_sequence_source: str | None = None
    if args.selection_mode == "decoded_segmental":
        selection_sequence_source = resolve_selection_sequence_source(
            args.selection_sequence_source,
            use_real=args.use_real,
            real_val_source_video_ids=real_val_source_video_ids,
        )
        if selection_sequence_source == "real_val":
            selection_sequences = load_real_move_sequences(
                clips_dir=args.real_clips_dir,
                image_size=args.image_size,
                selection_source_video_ids=(
                    None if not real_val_source_video_ids else real_val_source_video_ids
                ),
                exclude_selection_source_video_ids=False,
                observation_mode=args.observation_mode,
                oblique_crop_margin=args.oblique_crop_margin,
            )
        else:
            selection_sequences = load_eval_move_sequences(
                image_size=args.image_size,
                observation_mode=args.observation_mode,
                oblique_crop_margin=args.oblique_crop_margin,
            )
        if args.selection_max_clips > 0:
            selection_sequences = selection_sequences[: args.selection_max_clips]
        logger.info(
            "Loaded %d %s sequence(s) for decoded checkpoint selection",
            len(selection_sequences),
            selection_sequence_source,
        )

    if (
        args.initialize_move_model_checkpoint is not None
        and args.initialize_square_reader_checkpoint is not None
    ):
        raise ValueError(
            "--initialize-move-model-checkpoint and --initialize-square-reader-checkpoint "
            "are mutually exclusive"
        )

    joint_board_reader_checkpoint = None
    initial_model_checkpoint = None
    if args.initialize_move_model_checkpoint is not None:
        initial_model_checkpoint = torch.load(
            args.initialize_move_model_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model_kwargs = normalized_checkpoint_model_config(
            initial_model_checkpoint.get("model_config")
        )
    else:
        model_kwargs = {
            "vision_encoder_name": args.model_name,
            "vision_encoder_type": args.vision_encoder_type,
            "frozen_vision": True,
            "vision_feature_layer_indices": parse_feature_layer_indices(args.feature_layer_indices),
            "temporal_d_model": args.temporal_d_model,
            "temporal_n_layers": args.temporal_layers,
            "temporal_d_state": args.temporal_d_state,
            "temporal_expand": args.temporal_expand,
            "move_vocab_size": 1970,
            "pooling_type": args.pooling_type,
            "square_pool_size": 8,
            "square_head_enabled": args.square_loss_weight > 0.0,
            "square_head_type": args.square_head_type,
            "square_head_hidden_dim": args.square_head_hidden_dim,
            "square_head_transformer_layers": args.square_head_transformer_layers,
            "square_head_transformer_heads": args.square_head_transformer_heads,
            "square_head_transformer_ff_dim": args.square_head_transformer_ff_dim,
            "square_head_dropout": args.square_head_dropout,
            "square_token_mode": args.square_token_mode,
            "square_query_num_heads": args.square_query_num_heads,
            "square_query_dropout": args.square_query_dropout,
            "square_query_mlp_ratio": args.square_query_mlp_ratio,
            "use_detector": False,
        }
        if args.initialize_square_reader_checkpoint is not None:
            joint_board_reader_checkpoint = torch.load(
                args.initialize_square_reader_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            model_kwargs.update(
                argus_overrides_from_joint_board_reader_checkpoint(joint_board_reader_checkpoint)
            )

    model = ArgusModel(**model_kwargs)
    initialized_square_reader_adaptation = resolve_square_reader_adaptation_mode(args)
    if initial_model_checkpoint is not None:
        model.load_state_dict(initial_model_checkpoint["model_state_dict"])
        configure_square_reader_adaptation(model, initialized_square_reader_adaptation)
    elif joint_board_reader_checkpoint is not None:
        translated_state_dict = argus_square_reader_state_dict_from_joint_board_reader_checkpoint(
            joint_board_reader_checkpoint
        )
        load_result = model.load_state_dict(translated_state_dict, strict=False)
        unexpected_keys = sorted(load_result.unexpected_keys)
        if unexpected_keys:
            raise ValueError(
                f"Unexpected square-reader initialization keys: {', '.join(unexpected_keys)}"
            )
        missing_square_keys = sorted(
            key
            for key in load_result.missing_keys
            if key.startswith("square_tokenizer") or key.startswith("square_head")
        )
        if missing_square_keys:
            raise ValueError(
                f"Square-reader initialization missed model keys: {', '.join(missing_square_keys)}"
            )
        configure_square_reader_adaptation(model, initialized_square_reader_adaptation)

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

    best_selection_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_selection_metrics: dict[str, float] | None = None
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

        selection_metrics = dict(val_metrics)
        selection_key = raw_validation_selection_key(val_metrics)
        if (
            args.selection_mode == "decoded_segmental"
            and selection_sequences is not None
            and (epoch % args.selection_every == 0 or epoch == args.epochs)
        ):
            selection_metrics = evaluate_decoded_selection_metrics(
                model=model,
                sequences=selection_sequences,
                transform=val_transform,
                device=device,
                clip_length=args.clip_length,
                beam_size=args.selection_beam_size,
                top_move_candidates=args.selection_beam_top_moves,
                board_weight=args.selection_beam_board_weight,
                move_weight=args.selection_beam_move_weight,
                detect_weight=args.selection_beam_detect_weight,
                move_score_margin=args.selection_beam_move_score_margin,
                detect_peak_threshold=args.selection_segmental_detect_peak_threshold,
                board_change_peak_threshold=args.selection_segmental_board_change_peak_threshold,
                min_event_separation=args.selection_segmental_min_event_separation,
                max_event_proposals=args.selection_segmental_max_event_proposals,
                state_aware_proposal_passes=args.selection_segmental_state_aware_proposal_passes,
                board_drop_worst_frames=selection_segmental_board_drop_worst_frames(args),
                output_path=output_dir / f"selection_epoch{epoch:04d}.json",
            )
            selection_key = decoded_selection_key(selection_metrics)
            logger.info(
                "epoch %d selection board_exact=%.4f macro=%.4f recall=%.4f false_change=%.4f",
                epoch,
                selection_metrics.get("board_exact_match", 0.0),
                selection_metrics.get("macro_f1", 0.0),
                selection_metrics.get("move_detection_recall", 0.0),
                selection_metrics.get("static_frame_false_change_rate", 0.0),
            )

        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_selection_metrics = dict(selection_metrics)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_metrics is None or best_selection_metrics is None:
        raise RuntimeError("Training produced no best checkpoint")

    best_checkpoint = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "model_config": model.model_config,
        "metrics": best_metrics,
        "selection_metrics": best_selection_metrics,
        "selection_key": list(best_selection_key) if best_selection_key is not None else None,
        "selection_mode": args.selection_mode,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    best_path = output_dir / "best.pt"
    torch.save(best_checkpoint, best_path)
    logger.info("Saved best checkpoint to %s", best_path)

    summary = {
        "train_dataset_size": len(train_dataset),
        "train_component_sizes": train_component_sizes,
        "train_component_repeats": train_component_repeats,
        "real_train_source_video_ids": real_train_source_video_ids,
        "real_selection_source_video_ids": real_val_source_video_id_list,
        "val_dataset_size": len(val_dataset),
        "observation_mode": args.observation_mode,
        "vision_encoder_type": model.model_config["vision_encoder_type"],
        "vision_encoder_name": model.model_config["vision_encoder_name"],
        "vision_feature_layer_indices": model.model_config.get("vision_feature_layer_indices"),
        "square_token_mode": model.model_config["square_token_mode"],
        "square_head_type": model.model_config.get("square_head_type"),
        "initialize_move_model_checkpoint": None
        if args.initialize_move_model_checkpoint is None
        else str(args.initialize_move_model_checkpoint),
        "initialize_square_reader_checkpoint": None
        if args.initialize_square_reader_checkpoint is None
        else str(args.initialize_square_reader_checkpoint),
        "freeze_initialized_square_reader": args.freeze_initialized_square_reader,
        "initialized_square_reader_adaptation": initialized_square_reader_adaptation,
        "train_move_loss_mode": args.train_move_loss_mode,
        "move_target_pre_frames": args.move_target_pre_frames,
        "detect_target_radius": args.detect_target_radius,
        "detect_target_decay": args.detect_target_decay,
        "move_frame_loss_weight": args.move_frame_loss_weight,
        "no_move_loss_weight": args.no_move_loss_weight,
        "square_loss_weight": args.square_loss_weight,
        "real_negative_window_stride": args.real_negative_window_stride,
        "real_max_negative_windows_per_clip": args.real_max_negative_windows_per_clip,
        "real_move_positive_window_repeat": args.real_move_positive_window_repeat,
        "selection_mode": args.selection_mode,
        "selection_sequence_source": selection_sequence_source,
        "selection_every": args.selection_every,
        "selection_max_clips": args.selection_max_clips,
        "selection_segmental_board_drop_worst_frames": selection_segmental_board_drop_worst_frames(
            args
        ),
        "best_epoch": best_epoch,
        "best_selection_key": list(best_selection_key) if best_selection_key is not None else None,
        "best_metrics": best_metrics,
        "best_selection_metrics": best_selection_metrics,
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
    parser.add_argument("--square-loss-weight", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--observation-mode",
        choices=("rectified", "oblique", "native_oblique"),
        default="oblique",
    )
    parser.add_argument(
        "--train-move-loss-mode",
        choices=("move_frames", "all_frames"),
        default="move_frames",
    )
    parser.add_argument("--move-target-pre-frames", type=int, default=2)
    parser.add_argument("--detect-target-radius", type=int, default=1)
    parser.add_argument("--detect-target-decay", type=float, default=0.5)
    parser.add_argument("--move-frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-move-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--synthetic-root", type=Path, default=_DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--synthetic-train-repeat", type=int, default=1)
    parser.add_argument("--use-real", action="store_true")
    parser.add_argument("--real-clips-dir", type=Path, default=_DEFAULT_REAL_CLIPS_DIR)
    parser.add_argument("--real-train-repeat", type=int, default=1)
    parser.add_argument("--real-val-source-videos", type=str, default="")
    parser.add_argument("--real-val-source-video-count", type=int, default=2)
    parser.add_argument("--real-negative-window-stride", type=int, default=8)
    parser.add_argument("--real-max-negative-windows-per-clip", type=int, default=4)
    parser.add_argument("--real-move-positive-window-repeat", type=int, default=1)
    parser.add_argument("--oblique-crop-margin", type=float, default=0.18)
    parser.add_argument(
        "--vision-encoder-type",
        choices=("dinov2", "siglip", "siglip2", "yolo"),
        default="siglip",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--feature-layer-indices", type=str, default="")
    parser.add_argument(
        "--pooling-type",
        choices=("mean", "square_attention"),
        default="square_attention",
    )
    parser.add_argument(
        "--square-token-mode",
        choices=("pooled", "oblique_square_queries"),
        default="oblique_square_queries",
    )
    parser.add_argument(
        "--square-head-type",
        choices=("simple_mlp", "linear", "pos_mlp", "transformer"),
        default="simple_mlp",
    )
    parser.add_argument("--square-head-hidden-dim", type=int, default=512)
    parser.add_argument("--square-head-transformer-layers", type=int, default=2)
    parser.add_argument("--square-head-transformer-heads", type=int, default=8)
    parser.add_argument("--square-head-transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--square-head-dropout", type=float, default=0.1)
    parser.add_argument("--square-query-num-heads", type=int, default=8)
    parser.add_argument("--square-query-dropout", type=float, default=0.0)
    parser.add_argument("--square-query-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--initialize-move-model-checkpoint", type=Path, default=None)
    parser.add_argument("--initialize-square-reader-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-initialized-square-reader", action="store_true")
    parser.add_argument(
        "--initialized-square-reader-adaptation",
        choices=("full", "head_only", "frozen"),
        default=None,
    )
    parser.add_argument("--temporal-d-model", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-d-state", type=int, default=128)
    parser.add_argument("--temporal-expand", type=int, default=2)
    parser.add_argument(
        "--selection-mode",
        choices=("raw_val", "decoded_segmental"),
        default="raw_val",
    )
    parser.add_argument(
        "--selection-sequence-source",
        choices=("auto", "heldout_eval", "real_val"),
        default="auto",
    )
    parser.add_argument("--selection-every", type=int, default=1)
    parser.add_argument("--selection-max-clips", type=int, default=0)
    parser.add_argument("--selection-beam-size", type=int, default=8)
    parser.add_argument("--selection-beam-top-moves", type=int, default=16)
    parser.add_argument("--selection-beam-board-weight", type=float, default=1.0)
    parser.add_argument("--selection-beam-move-weight", type=float, default=0.2)
    parser.add_argument("--selection-beam-detect-weight", type=float, default=0.1)
    parser.add_argument("--selection-beam-move-score-margin", type=float, default=2.0)
    parser.add_argument("--selection-segmental-detect-peak-threshold", type=float, default=0.1)
    parser.add_argument(
        "--selection-segmental-board-change-peak-threshold", type=float, default=0.03125
    )
    parser.add_argument("--selection-segmental-min-event-separation", type=int, default=16)
    parser.add_argument("--selection-segmental-max-event-proposals", type=int, default=20)
    parser.add_argument("--selection-segmental-state-aware-proposal-passes", type=int, default=0)
    parser.add_argument("--selection-segmental-board-drop-worst-frames", type=int, default=-1)
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
    if all("board_corners" in sample for sample in batch):
        collated["board_corners"] = torch.stack([sample["board_corners"] for sample in batch])
    return collated


def load_transformers_config(model_name: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoConfig.from_pretrained(model_name)


def normalized_checkpoint_model_config(model_config: dict[str, Any] | None) -> dict[str, Any]:
    if model_config is None:
        raise ValueError("Checkpoint is missing model_config; cannot initialize move model")

    normalized = dict(model_config)
    encoder_type = str(normalized.get("vision_encoder_type", "")).lower()
    model_name = normalized.get("vision_encoder_name")
    if encoder_type == "siglip2" and isinstance(model_name, str):
        model_type = getattr(load_transformers_config(model_name), "model_type", None)
        if model_type in {"siglip", "siglip_vision_model"}:
            logger.warning(
                "Legacy checkpoint requests encoder_type='siglip2' for %s, but its config "
                "model_type is %r; remapping to encoder_type='siglip' for compatibility",
                model_name,
                model_type,
            )
            normalized["vision_encoder_type"] = "siglip"
    return normalized


def parse_source_video_ids(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def select_real_val_source_video_ids(
    rows: list[Any],
    *,
    requested_source_video_ids: list[str],
    requested_count: int,
) -> list[str]:
    source_video_to_clips: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_video_id = getattr(row, "source_video_id", None)
        clip_path = getattr(row, "clip_path", None)
        if not isinstance(source_video_id, str) or not source_video_id:
            continue
        if not isinstance(clip_path, str) or not clip_path:
            continue
        source_video_to_clips[source_video_id].add(clip_path)

    available_source_video_ids = sorted(source_video_to_clips)
    if requested_count < 0:
        raise ValueError(f"real_val_source_video_count must be >= 0, got {requested_count}")
    if requested_source_video_ids:
        missing_source_video_ids = sorted(
            set(requested_source_video_ids) - set(available_source_video_ids)
        )
        if missing_source_video_ids:
            raise ValueError(
                f"Unknown real selection source videos: {','.join(missing_source_video_ids)}"
            )
        return sorted(set(requested_source_video_ids))
    if requested_count == 0:
        return []
    if requested_count >= len(available_source_video_ids):
        raise ValueError(
            "real_val_source_video_count must be smaller than the number of available source "
            f"videos ({len(available_source_video_ids)})"
        )

    ranked_source_video_ids = sorted(
        source_video_to_clips,
        key=lambda source_video_id: (
            -len(source_video_to_clips[source_video_id]),
            source_video_id,
        ),
    )
    return sorted(ranked_source_video_ids[:requested_count])


def resolve_real_val_source_video_ids(args: argparse.Namespace) -> list[str]:
    requested_source_video_ids = parse_source_video_ids(args.real_val_source_videos)
    if not args.use_real:
        return []
    rows = load_real_board_rows(
        clips_dir=args.real_clips_dir,
        frame_stride=1,
        max_frames=None,
    )
    if not rows:
        return []
    return select_real_val_source_video_ids(
        rows,
        requested_source_video_ids=requested_source_video_ids,
        requested_count=args.real_val_source_video_count,
    )


def resolve_selection_sequence_source(
    requested_source: str,
    *,
    use_real: bool,
    real_val_source_video_ids: set[str],
) -> str:
    if requested_source == "auto":
        return "real_val" if use_real and real_val_source_video_ids else "heldout_eval"
    if requested_source == "real_val":
        if not use_real:
            raise ValueError("--selection-sequence-source real_val requires --use-real")
        if not real_val_source_video_ids:
            raise ValueError(
                "--selection-sequence-source real_val requires a non-empty real-val split"
            )
    return requested_source


def resolve_square_reader_adaptation_mode(args: argparse.Namespace) -> str:
    if args.initialized_square_reader_adaptation is not None:
        return str(args.initialized_square_reader_adaptation)
    return "frozen" if args.freeze_initialized_square_reader else "full"


def configure_square_reader_adaptation(model: ArgusModel, mode: str) -> None:
    if mode == "full":
        return
    if mode == "head_only":
        freeze_module(model.square_tokenizer)
        return
    if mode == "frozen":
        freeze_module(model.square_tokenizer)
        freeze_module(model.square_head)
        return
    raise ValueError(f"Unsupported square-reader adaptation mode: {mode}")


def freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def raw_validation_selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        float(metrics.get("move_accuracy", 0.0)),
        float(metrics.get("move_detection_recall", 0.0)),
        float(metrics.get("move_detection_f1", 0.0)),
        -float(metrics.get("val_loss", 0.0)),
    )


def decoded_selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        float(metrics.get("board_exact_match", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        float(metrics.get("move_detection_recall", 0.0)),
        -float(metrics.get("static_frame_false_change_rate", 0.0)),
    )


def selection_segmental_board_drop_worst_frames(args: argparse.Namespace) -> int:
    if args.selection_segmental_board_drop_worst_frames >= 0:
        return int(args.selection_segmental_board_drop_worst_frames)
    return 2 if args.observation_mode == "native_oblique" else 0


def evaluate_decoded_selection_metrics(
    *,
    model: ArgusModel,
    sequences: list[Any],
    transform: ValidationTransform,
    device: str,
    clip_length: int,
    beam_size: int,
    top_move_candidates: int,
    board_weight: float,
    move_weight: float,
    detect_weight: float,
    move_score_margin: float,
    detect_peak_threshold: float,
    board_change_peak_threshold: float,
    min_event_separation: int,
    max_event_proposals: int,
    state_aware_proposal_passes: int,
    board_drop_worst_frames: int,
    output_path: Path,
) -> dict[str, float]:
    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    square_predictions: list[int] = []
    square_targets: list[int] = []
    board_annotation_ids: list[str] = []

    with torch.no_grad():
        for sequence in sequences:
            sequence_predictions, _diagnostics = decode_sequence_with_segmental_decoder(
                model=model,
                sequence=sequence,
                transform=transform,
                device=torch.device(device),
                beam_size=beam_size,
                top_move_candidates=top_move_candidates,
                top_board_candidates=0,
                board_weight=board_weight,
                move_weight=move_weight,
                detect_weight=detect_weight,
                move_score_margin=move_score_margin,
                detect_peak_threshold=detect_peak_threshold,
                board_change_peak_threshold=board_change_peak_threshold,
                min_event_separation=min_event_separation,
                secondary_min_event_separation=None,
                secondary_peak_ratio=0.8,
                state_aware_proposal_passes=state_aware_proposal_passes,
                anomaly_change_evidence_threshold=0.25,
                anomaly_settled_gain_threshold=0.0,
                segment_board_drop_worst_frames=board_drop_worst_frames,
                event_window_radius=1,
                max_event_proposals=max_event_proposals,
                diagnostic_settled_horizon=8,
            )
            for frame_offset, (board, accepted_move_uci) in enumerate(sequence_predictions):
                predicted_labels = tuple(board_to_class_ids(build_replay_board(board.board_fen())))
                target_labels = sequence.labels[frame_offset]
                annotation_id = (
                    f"{Path(sequence.clip_path).stem}_frame"
                    f"{sequence.frame_indices[frame_offset]:04d}"
                )
                predictions_by_clip[sequence.clip_path].append(
                    FramePrediction(
                        annotation_id=annotation_id,
                        frame_index=int(sequence.frame_indices[frame_offset]),
                        target_labels=target_labels,
                        predicted_labels=predicted_labels,
                        move_uci=accepted_move_uci,
                    )
                )
                square_predictions.extend(predicted_labels)
                square_targets.extend(target_labels)
                board_annotation_ids.extend([annotation_id] * 64)

    logits = torch.zeros((len(square_predictions), len(SQUARE_CLASS_NAMES)), dtype=torch.float32)
    for index, class_id in enumerate(square_predictions):
        logits[index, class_id] = 1.0
    metrics = evaluate_probe(
        IdentityProbe(),
        logits,
        torch.tensor(square_targets, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )
    move_recall, static_false_change_rate, diagnostics = compute_tracker_sequence_metrics(
        predictions_by_clip,
        tolerance=1,
    )
    report = {
        "metrics": metrics.to_dict(),
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": static_false_change_rate,
        "sequence_diagnostics": diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return {
        "board_exact_match": float(report["metrics"]["board_exact_match"]),
        "non_empty_accuracy": float(report["metrics"]["non_empty_accuracy"]),
        "macro_f1": float(report["metrics"]["macro_f1"]),
        "move_detection_recall": float(move_recall),
        "static_frame_false_change_rate": float(static_false_change_rate),
    }


def parse_feature_layer_indices(raw_value: str) -> list[int] | None:
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not values:
        return None
    return [int(value) for value in values]


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


if __name__ == "__main__":
    main()
