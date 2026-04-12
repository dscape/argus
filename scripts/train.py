#!/usr/bin/env python3
"""Training entry point for Argus."""

# Python 3.14 compatibility: Hydra's LazyCompletionHelp breaks argparse's new
# _check_help validation. Patch argparse to tolerate non-string help objects.
import argparse
import logging
import sys

_orig_check_help = getattr(argparse.ArgumentParser, "_check_help", None)
if _orig_check_help is not None:

    def _patched_check_help(self, action):  # type: ignore[no-untyped-def]
        if not isinstance(action.help, str):
            return
        _orig_check_help(self, action)

    argparse.ArgumentParser._check_help = _patched_check_help  # type: ignore[attr-defined]

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402

from argus.data.collate import argus_collate_fn  # noqa: E402
from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset  # noqa: E402
from argus.data.transforms import ValidationTransform  # noqa: E402
from argus.device import resolve_device  # noqa: E402
from argus.model.argus_model import ArgusModel  # noqa: E402
from argus.training.trainer import Trainer, compute_num_optimizer_steps  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _compute_split_lengths(total_clips: int, requested_val_clips: int) -> tuple[int, int]:
    if total_clips <= 0:
        raise ValueError("Dataset is empty")
    if total_clips == 1:
        return 1, 0

    val_clips = min(max(requested_val_clips, 0), total_clips - 1)
    train_clips = total_clips - val_clips
    return train_clips, val_clips


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Training config: {cfg.training.name}")
    device = resolve_device()
    logger.info(f"Using device: {device}")

    clip_length = cfg.data.get("clip_length", 16)
    seed = cfg.get("seed", 42)
    data_dir = cfg.data.get("data_dir", None)
    transform = ValidationTransform()

    if data_dir is not None:
        # Load pre-generated data from disk (use make datagen first)
        from pathlib import Path

        train_dir = Path(data_dir) / "train"
        val_dir = Path(data_dir) / "val"
        if not train_dir.exists():
            # Flat directory: split by ratio
            logger.info(f"Loading data from {data_dir}...")
            full_ds = ArgusDataset(data_dir, clip_length=clip_length, transform=transform)
            requested_val = int(cfg.data.get("num_val_clips", 100))
            num_train, num_val = _compute_split_lengths(len(full_ds), requested_val)
            logger.info(
                "Using deterministic random split for flat clip directory: %d train / %d val",
                num_train,
                num_val,
            )
            train_ds, val_ds = random_split(
                full_ds,
                [num_train, num_val],
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            # Separate train/val directories
            logger.info(f"Loading train from {train_dir}, val from {val_dir}...")
            train_ds = ArgusDataset(train_dir, clip_length=clip_length, transform=transform)
            val_ds = ArgusDataset(val_dir, clip_length=clip_length, transform=transform)
    else:
        # Generate synthetic data on-the-fly
        from argus.datagen.synth import generate_dataset

        num_train_clips = cfg.data.get("num_train_clips", 1000)
        num_val_clips = cfg.data.get("num_val_clips", 100)
        total_clips = num_train_clips + num_val_clips
        image_size = cfg.data.get("image_size", 224)

        logger.info(
            f"Generating {total_clips} synthetic clips "
            f"({num_train_clips} train, {num_val_clips} val)..."
        )
        clips = generate_dataset(
            num_clips=total_clips,
            clip_length=clip_length,
            image_size=image_size,
            frames_per_move=cfg.data.get("frames_per_move", 4),
            augment=cfg.data.get("augment", True),
            occlusion_prob=cfg.data.get("occlusion_prob", 0.2),
            illegal_clip_prob=cfg.data.get("illegal_clip_prob", 0.0),
            min_moves=cfg.data.get("min_moves", 10),
            max_moves=cfg.data.get("max_moves", 80),
            game_source=cfg.data.get("game_source", "random"),
            pgn_path=cfg.data.get("pgn_path", None),
            min_elo=cfg.data.get("min_elo", 1500),
            seed=seed,
        )
        logger.info(f"Generated {len(clips)} clips")

        dataset = ArgusInMemoryDataset(clips=clips, clip_length=clip_length, transform=transform)
        train_ds, val_ds = random_split(dataset, [num_train_clips, num_val_clips])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=argus_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=argus_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    vision_encoder_cfg = cfg.model.vision_encoder
    pooling_cfg = cfg.model.get("pooling", {})
    square_head_cfg = cfg.model.get("square_head", {})
    feature_layer_indices = vision_encoder_cfg.get("feature_layer_indices")
    model = ArgusModel(
        vision_encoder_name=vision_encoder_cfg.model_name,
        vision_encoder_type=vision_encoder_cfg.get("type", "dinov2"),
        vision_embed_dim=vision_encoder_cfg.get("embed_dim"),
        vision_feature_layer_indices=list(feature_layer_indices) if feature_layer_indices else None,
        vision_output_grid_size=vision_encoder_cfg.get("output_grid_size", 14),
        frozen_vision=vision_encoder_cfg.frozen,
        temporal_d_model=cfg.model.temporal.d_model,
        temporal_n_layers=cfg.model.temporal.n_layers,
        temporal_d_state=cfg.model.temporal.d_state,
        temporal_expand=cfg.model.temporal.expand,
        move_vocab_size=cfg.model.move_head.vocab_size,
        pooling_type=pooling_cfg.get("type", "mean"),
        square_pool_size=pooling_cfg.get("square_size", 8),
        square_head_enabled=square_head_cfg.get("enabled", False),
        square_vocab_size=square_head_cfg.get("num_classes", 13),
        use_detector=False,
    )

    unfreeze_last_n = int(vision_encoder_cfg.get("unfreeze_last_n", 0))
    if unfreeze_last_n > 0:
        model.vision_encoder.unfreeze_last_n_layers(unfreeze_last_n)
        logger.info("Unfroze last %d vision encoder layer(s)", unfreeze_last_n)

    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Initialized model weights from %s", init_checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    steps_per_epoch = compute_num_optimizer_steps(
        len(train_loader),
        cfg.training.gradient_accumulation,
    )
    total_steps = steps_per_epoch * cfg.training.epochs
    logger.info(
        "Optimizer steps: %d per epoch, %d total",
        steps_per_epoch,
        total_steps,
    )
    if total_steps < 200:
        logger.warning(
            "Only %d optimizer steps configured; "
            "treat this as a debugging probe, not a convergence run",
            total_steps,
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_steps=cfg.training.scheduler.warmup_steps,
        total_steps=total_steps,
        min_lr=cfg.training.scheduler.min_lr,
        clip_grad_norm=cfg.training.clip_grad_norm,
        gradient_accumulation=cfg.training.gradient_accumulation,
        precision=cfg.training.precision,
        loss_weights=dict(cfg.training.loss_weights),
        output_dir=cfg.get("output_dir", "outputs"),
        save_every=cfg.training.checkpoint.save_every,
        use_wandb=cfg.training.wandb.get("enabled", False),
        device=device,
    )

    trainer.fit(epochs=cfg.training.epochs)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
