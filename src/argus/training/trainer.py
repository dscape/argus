"""Main training loop for Argus."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from argus.eval.metrics import compute_move_metrics
from argus.model.argus_model import ArgusModel
from argus.model.losses import ArgusLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for Argus.

    Handles the training loop, validation, checkpointing,
    and metric logging.
    """

    def __init__(
        self,
        model: ArgusModel,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader | None = None,  # type: ignore[type-arg]
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        total_steps: int | None = None,
        min_lr: float = 1e-6,
        clip_grad_norm: float = 1.0,
        gradient_accumulation: int = 1,
        precision: str = "bf16",
        loss_weights: dict[str, float] | None = None,
        output_dir: str = "outputs",
        save_every: int = 5,
        use_wandb: bool = False,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clip_grad_norm = clip_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.precision = precision
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.use_wandb = use_wandb
        self.device = torch.device(device)

        # Loss
        lw = loss_weights or {}
        self.criterion = ArgusLoss(
            w_move=lw.get("move", 1.0),
            w_detect=lw.get("detect", 0.5),
            w_bbox=lw.get("bbox", 0.0),
            w_identity=lw.get("identity", 0.0),
        )

        # Optimizer (separate LR for vision encoder if needed)
        param_groups = self._build_param_groups(lr, weight_decay)
        self.optimizer = AdamW(param_groups)

        # Scheduler
        if total_steps is None:
            total_steps = len(train_loader) * 50  # Default to 50 epochs
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Mixed precision
        self.use_amp = precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = GradScaler(device, enabled=(precision == "fp16"))

        # Logging
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not available, disabling")
                self.use_wandb = False

        self.global_step = 0

    def _build_param_groups(
        self,
        lr: float,
        weight_decay: float,
    ) -> list[dict]:  # type: ignore[type-arg]
        """Separate param groups for vision encoder vs rest."""
        vision_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "vision_encoder" in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
        if vision_params:
            groups.append(
                {"params": vision_params, "lr": lr * 0.1, "weight_decay": weight_decay}
            )
        return groups

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "total": [],
            "move": [],
            "detect": [],
        }
        epoch_metrics: dict[str, list[float]] = {
            "move_acc": [],
        }

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            frames = batch["frames"].to(self.device)
            move_targets = batch["move_targets"].to(self.device)
            detect_targets = batch["detect_targets"].to(self.device)
            legal_masks = batch["legal_masks"].to(self.device)
            move_mask = batch["move_mask"].to(self.device)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(crops=frames, legal_masks=legal_masks)

                # Remove the board dimension for single-board
                move_logits = output.move_logits.squeeze(2)  # (B, T, VOCAB_SIZE)
                detect_logits = output.detect_logits.squeeze(2)  # (B, T)

                losses = self.criterion(
                    move_logits=move_logits,
                    detect_logits=detect_logits,
                    move_targets=move_targets,
                    detect_targets=detect_targets,
                    move_mask=move_mask,
                )

            loss = losses["total"] / self.gradient_accumulation
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Record losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key].append(losses[key].item())

            # Compute metrics on this batch
            with torch.no_grad():
                preds = output.move_probs.squeeze(2).argmax(dim=-1)
                metrics = compute_move_metrics(
                    preds, move_targets, detect_logits, detect_targets, move_mask
                )
                epoch_metrics["move_acc"].append(metrics.get("move_accuracy", 0.0))

            # Logging
            if self.use_wandb and self.global_step % 10 == 0:
                log_dict = {f"train/{k}": v[-1] for k, v in epoch_losses.items() if v}
                log_dict["train/move_acc"] = metrics.get("move_accuracy", 0.0)
                log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]
                self._wandb.log(log_dict, step=self.global_step)

        return {
            k: sum(v) / max(len(v), 1) for k, v in {**epoch_losses, **epoch_metrics}.items()
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_detect_logits: list[torch.Tensor] = []
        all_detect_targets: list[torch.Tensor] = []
        all_move_masks: list[torch.Tensor] = []
        total_loss = 0.0
        count = 0

        for batch in self.val_loader:
            frames = batch["frames"].to(self.device)
            move_targets = batch["move_targets"].to(self.device)
            detect_targets = batch["detect_targets"].to(self.device)
            legal_masks = batch["legal_masks"].to(self.device)
            move_mask = batch["move_mask"].to(self.device)

            output = self.model(crops=frames, legal_masks=legal_masks)

            move_logits = output.move_logits.squeeze(2)
            detect_logits = output.detect_logits.squeeze(2)

            losses = self.criterion(
                move_logits=move_logits,
                detect_logits=detect_logits,
                move_targets=move_targets,
                detect_targets=detect_targets,
                move_mask=move_mask,
            )
            total_loss += losses["total"].item()
            count += 1

            preds = output.move_probs.squeeze(2).argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(move_targets.cpu())
            all_detect_logits.append(detect_logits.cpu())
            all_detect_targets.append(detect_targets.cpu())
            all_move_masks.append(move_mask.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        d_logits = torch.cat(all_detect_logits)
        d_targets = torch.cat(all_detect_targets)
        m_masks = torch.cat(all_move_masks)

        metrics = compute_move_metrics(preds, targets, d_logits, d_targets, m_masks)
        metrics["val_loss"] = total_loss / max(count, 1)

        if self.use_wandb:
            self._wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=self.global_step)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, float] | None = None) -> Path:
        """Save model checkpoint."""
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch:04d}.pt"
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if metrics:
            checkpoint["metrics"] = metrics
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint.get("epoch", 0)

    def fit(self, epochs: int) -> None:
        """Full training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            elapsed = time.time() - t0

            log_parts = [f"Epoch {epoch}/{epochs} ({elapsed:.1f}s)"]
            for k, v in train_metrics.items():
                log_parts.append(f"{k}={v:.4f}")
            logger.info(" | ".join(log_parts))

            # Validate
            if self.val_loader and epoch % 2 == 0:
                val_metrics = self.validate()
                val_parts = [f"  Val"]
                for k, v in val_metrics.items():
                    val_parts.append(f"{k}={v:.4f}")
                logger.info(" | ".join(val_parts))

            # Checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, train_metrics)
