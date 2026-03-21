#!/usr/bin/env python3
"""Training entry point for Argus."""

import logging
import sys

# Python 3.14 compatibility: Hydra's LazyCompletionHelp breaks argparse's new
# _check_help validation. Patch argparse to tolerate non-string help objects.
import argparse
_orig_check_help = getattr(argparse.ArgumentParser, "_check_help", None)
if _orig_check_help is not None:
    def _patched_check_help(self, action):  # type: ignore[no-untyped-def]
        if not isinstance(action.help, str):
            return
        _orig_check_help(self, action)
    argparse.ArgumentParser._check_help = _patched_check_help  # type: ignore[attr-defined]

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from argus.data.collate import argus_collate_fn
from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset
from argus.model.argus_model import ArgusModel
from argus.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Training config: {cfg.training.name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    clip_length = cfg.data.get("clip_length", 16)
    seed = cfg.get("seed", 42)
    data_dir = cfg.data.get("data_dir", None)

    if data_dir is not None:
        # Load pre-generated data from disk (use make datagen first)
        from pathlib import Path
        train_dir = Path(data_dir) / "train"
        val_dir = Path(data_dir) / "val"
        if not train_dir.exists():
            # Flat directory: split by ratio
            logger.info(f"Loading data from {data_dir}...")
            full_ds = ArgusDataset(data_dir, clip_length=clip_length)
            num_val = cfg.data.get("num_val_clips", 100)
            num_train = len(full_ds) - num_val
            train_ds, val_ds = random_split(full_ds, [num_train, num_val])
        else:
            # Separate train/val directories
            logger.info(f"Loading train from {train_dir}, val from {val_dir}...")
            train_ds = ArgusDataset(train_dir, clip_length=clip_length)
            val_ds = ArgusDataset(val_dir, clip_length=clip_length)
    else:
        # Generate synthetic data on-the-fly
        from argus.datagen.synth import generate_dataset
        num_train_clips = cfg.data.get("num_train_clips", 1000)
        num_val_clips = cfg.data.get("num_val_clips", 100)
        total_clips = num_train_clips + num_val_clips
        image_size = cfg.data.get("image_size", 224)

        logger.info(f"Generating {total_clips} synthetic clips ({num_train_clips} train, {num_val_clips} val)...")
        clips = generate_dataset(
            num_clips=total_clips,
            clip_length=clip_length,
            image_size=image_size,
            seed=seed,
        )
        logger.info(f"Generated {len(clips)} clips")

        dataset = ArgusInMemoryDataset(clips=clips, clip_length=clip_length)
        train_ds, val_ds = random_split(dataset, [num_train_clips, num_val_clips])

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=argus_collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=argus_collate_fn, num_workers=0, pin_memory=True)

    model = ArgusModel(
        vision_encoder_name=cfg.model.vision_encoder.model_name,
        vision_embed_dim=cfg.model.vision_encoder.embed_dim,
        frozen_vision=cfg.model.vision_encoder.frozen,
        temporal_d_model=cfg.model.temporal.d_model,
        temporal_n_layers=cfg.model.temporal.n_layers,
        temporal_d_state=cfg.model.temporal.d_state,
        temporal_expand=cfg.model.temporal.expand,
        move_vocab_size=cfg.model.move_head.vocab_size,
        use_detector=False,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    total_steps = len(train_loader) * cfg.training.epochs // cfg.training.gradient_accumulation

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay,
        warmup_steps=cfg.training.scheduler.warmup_steps, total_steps=total_steps,
        min_lr=cfg.training.scheduler.min_lr, clip_grad_norm=cfg.training.clip_grad_norm,
        gradient_accumulation=cfg.training.gradient_accumulation, precision=cfg.training.precision,
        loss_weights=dict(cfg.training.loss_weights),
        output_dir=cfg.get("output_dir", "outputs"),
        save_every=cfg.training.checkpoint.save_every,
        use_wandb=cfg.training.wandb.get("enabled", False), device=device,
    )

    trainer.fit(epochs=cfg.training.epochs)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
