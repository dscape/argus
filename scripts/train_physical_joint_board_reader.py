#!/usr/bin/env python3
"""Train a joint oblique whole-board reader on frozen dense patch tokens."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.annotation_dataset import rectify_board_image
from pipeline.physical.joint_board_reader import (
    JointBoardReaderConfig,
    evaluate_joint_board_reader,
    extract_patch_token_features,
    train_joint_board_reader,
)
from pipeline.physical.oblique_board_data import (
    PhysicalEvalObliqueBoardDataset,
    PhysicalRealObliqueBoardDataset,
    PhysicalSyntheticWarpedObliqueBoardDataset,
)
from pipeline.physical.oblique_square_context import PhysicalRealObliqueBoardRow
from pipeline.physical.real_board_data import (
    PhysicalRealBoardRow,
    _frame_tensor_to_rgb,
    load_real_board_rows,
)
from pipeline.physical.square_classifier import read_board_logits_batch_from_frames
from pipeline.physical.square_probe import ProbeMetrics

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_joint_board_reader"


def main() -> None:
    args = build_parser().parse_args()
    args.model_name = args.model_name or default_model_name_for_encoder_type(args.encoder_type)
    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = VisionEncoder(
        encoder_type=args.encoder_type,
        model_name=args.model_name,
        frozen=True,
        feature_layer_indices=parse_feature_layer_indices(args.feature_layer_indices),
    ).to(device)
    encoder.eval()

    real_train_dataset, selection_dataset, train_source_ids, selection_source_ids = (
        build_real_datasets(
            clips_dir=args.real_train_clips_dir,
            image_size=args.input_size,
            frame_stride=args.real_train_frame_stride,
            max_train_frames=args.real_train_max_frames,
            seed=args.seed,
            exclude_move_neighborhood=args.real_train_exclude_move_neighborhood,
            selection_source_video_ids=parse_source_video_ids(args.real_val_source_videos),
            selection_source_video_count=args.real_val_source_video_count,
        )
    )
    synthetic_train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    if args.synthetic_train_positions > 0:
        synthetic_train_dataset = PhysicalSyntheticWarpedObliqueBoardDataset(
            num_positions=args.synthetic_train_positions,
            image_size=args.input_size,
            seed=args.seed,
            augment=args.synthetic_augment,
            min_moves=args.synthetic_min_moves,
            max_moves=args.synthetic_max_moves,
            min_ply=args.synthetic_min_ply,
        )

    train_parts: list[Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    if synthetic_train_dataset is not None:
        train_parts.append(synthetic_train_dataset)
    train_parts.append(real_train_dataset)
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)
    eval_dataset = PhysicalEvalObliqueBoardDataset(image_size=args.input_size)

    eval_rows = getattr(eval_dataset, "rows")
    eval_annotation_ids = [row.annotation_id for row in eval_rows]
    eval_source_ids = [row.source_video_id or "unknown" for row in eval_rows]

    logger.info("Extracting train patch tokens")
    train_patch_tokens, train_labels, train_corners = extract_patch_token_features(
        train_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.encoder_batch_size,
        storage_dtype=torch.float16,
    )
    logger.info("Extracting selection patch tokens")
    selection_patch_tokens, selection_labels, selection_corners = extract_patch_token_features(
        selection_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.encoder_batch_size,
        storage_dtype=torch.float16,
    )
    logger.info("Extracting held-out eval patch tokens")
    eval_patch_tokens, eval_labels, eval_corners = extract_patch_token_features(
        eval_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.encoder_batch_size,
        storage_dtype=torch.float16,
    )

    class_weights = build_class_weights(train_labels, mode=args.class_weighting)
    board_weights = build_board_weights(
        synthetic_count=0 if synthetic_train_dataset is None else len(synthetic_train_dataset),
        real_count=len(real_train_dataset),
        real_loss_weight=args.real_loss_weight,
    )
    train_teacher_logits = None
    train_distill_mask = None
    if args.teacher_distillation_weight > 0.0:
        logger.info("Extracting teacher logits from %s", args.teacher_weights_path)
        real_teacher_logits = build_teacher_logits_for_real_rows(
            rows=list(real_train_dataset.rows),
            image_size=args.teacher_input_size,
            device=args.device,
            weights_path=args.teacher_weights_path,
            batch_size=args.teacher_batch_size,
        )
        synthetic_count = 0 if synthetic_train_dataset is None else len(synthetic_train_dataset)
        train_teacher_logits = torch.zeros(
            train_labels.shape[0],
            train_labels.shape[1],
            13,
            dtype=torch.float32,
        )
        train_distill_mask = torch.zeros(train_labels.shape[0], dtype=torch.float32)
        train_teacher_logits[synthetic_count:] = real_teacher_logits
        train_distill_mask[synthetic_count:] = 1.0
    config = JointBoardReaderConfig(
        input_size=args.input_size,
        num_classes=13,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
    )
    model, best_selection_score = train_joint_board_reader(
        train_patch_tokens,
        train_labels,
        train_corners,
        selection_patch_tokens,
        selection_labels,
        selection_corners,
        config=config,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.train_batch_size,
        device=device,
        class_weights=class_weights,
        board_weights=board_weights,
        train_teacher_logits=train_teacher_logits,
        train_distill_mask=train_distill_mask,
        distillation_weight=args.teacher_distillation_weight,
        selection_metric=args.selection_metric,
    )

    selection_annotation_ids = annotation_ids_for_dataset(selection_dataset)
    selection_metrics = evaluate_joint_board_reader(
        model,
        selection_patch_tokens,
        selection_labels,
        selection_corners,
        device=device,
        annotation_ids=selection_annotation_ids,
    )
    eval_metrics = evaluate_joint_board_reader(
        model,
        eval_patch_tokens,
        eval_labels,
        eval_corners,
        device=device,
        annotation_ids=eval_annotation_ids,
    )
    per_source_metrics = evaluate_per_source_metrics(
        model,
        eval_patch_tokens,
        eval_labels,
        eval_corners,
        annotation_ids=eval_annotation_ids,
        source_video_ids=eval_source_ids,
        device=device,
    )

    checkpoint = {
        "architecture": "joint_oblique_board_reader",
        "state_dict": model.state_dict(),
        "reader_config": model.checkpoint_config(),
        "encoder_config": {
            "encoder_type": args.encoder_type,
            "model_name": args.model_name,
            "feature_layer_indices": parse_feature_layer_indices(args.feature_layer_indices),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_score": best_selection_score,
        "selection_metrics": selection_metrics.to_dict(),
        "eval_metrics": eval_metrics.to_dict(),
        "per_source_eval_metrics": {k: v.to_dict() for k, v in per_source_metrics.items()},
    }
    checkpoint_path = output_dir / "joint_board_reader.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "encoder_type": args.encoder_type,
        "model_name": args.model_name,
        "feature_layer_indices": parse_feature_layer_indices(args.feature_layer_indices),
        "train_frames": len(train_dataset),
        "synthetic_train_frames": 0
        if synthetic_train_dataset is None
        else len(synthetic_train_dataset),
        "real_train_frames": len(real_train_dataset),
        "selection_frames": len(selection_dataset),
        "eval_frames": len(eval_dataset),
        "train_source_video_ids": train_source_ids,
        "selection_source_video_ids": selection_source_ids,
        "selection_score": best_selection_score,
        "selection_metric": args.selection_metric,
        "teacher_distillation_weight": args.teacher_distillation_weight,
        "teacher_weights_path": None
        if args.teacher_weights_path is None
        else str(args.teacher_weights_path),
        "selection_metrics": selection_metrics.to_dict(),
        "eval_metrics": eval_metrics.to_dict(),
        "per_source_eval_metrics": {k: v.to_dict() for k, v in per_source_metrics.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("Selection metrics: %s", json.dumps(selection_metrics.to_dict(), sort_keys=True))
    logger.info("Eval metrics: %s", json.dumps(eval_metrics.to_dict(), sort_keys=True))
    logger.info(
        "Per-source eval metrics: %s",
        json.dumps({k: v.to_dict() for k, v in per_source_metrics.items()}, sort_keys=True),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train joint oblique whole-board reader")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--head-type", choices=("linear", "pos_mlp", "transformer"), default="pos_mlp"
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--class-weighting", choices=("none", "max_ratio"), default="max_ratio")
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "non_empty_accuracy", "macro_f1", "non_empty_plus_macro"),
        default="non_empty_plus_macro",
    )
    parser.add_argument("--synthetic-train-positions", type=int, default=0)
    parser.add_argument("--synthetic-augment", action="store_true")
    parser.add_argument("--synthetic-min-moves", type=int, default=12)
    parser.add_argument("--synthetic-max-moves", type=int, default=80)
    parser.add_argument("--synthetic-min-ply", type=int, default=8)
    parser.add_argument("--real-train-clips-dir", type=Path, default=Path("data/argus/train_real"))
    parser.add_argument("--real-train-frame-stride", type=int, default=1)
    parser.add_argument("--real-train-max-frames", type=int, default=0)
    parser.add_argument("--real-train-exclude-move-neighborhood", type=int, default=-1)
    parser.add_argument("--real-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-weights-path", type=Path, default=Path("weights/physical/v7r6.pt")
    )
    parser.add_argument("--teacher-input-size", type=int, default=224)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--real-val-source-video-count", type=int, default=0)
    parser.add_argument("--real-val-source-videos", type=str, default="psrPAoHr4wA")
    parser.add_argument(
        "--encoder-type", choices=("dinov2", "siglip", "siglip2", "yolo"), default="dinov2"
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--feature-layer-indices", type=str, default="8,10,11")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def parse_feature_layer_indices(raw_value: str) -> list[int] | None:
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not values:
        return None
    return [int(value) for value in values]


def parse_source_video_ids(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def build_real_datasets(
    *,
    clips_dir: Path,
    image_size: int,
    frame_stride: int,
    max_train_frames: int,
    seed: int,
    exclude_move_neighborhood: int,
    selection_source_video_ids: list[str],
    selection_source_video_count: int,
) -> tuple[
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    list[str],
    list[str],
]:
    rows = load_real_board_rows(
        clips_dir=clips_dir,
        frame_stride=frame_stride,
        max_frames=None,
        seed=seed,
        exclude_move_neighborhood=exclude_move_neighborhood,
    )
    if not rows:
        raise ValueError("No pseudo-real rows available")

    available_source_ids = sorted({row.source_video_id for row in rows if row.source_video_id})
    if selection_source_video_ids:
        selection_ids = sorted(set(selection_source_video_ids))
    elif selection_source_video_count > 0:
        if selection_source_video_count >= len(available_source_ids):
            raise ValueError(
                "real_val_source_video_count must be smaller than available pseudo-real videos"
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(available_source_ids), generator=generator).tolist()
        selection_ids = sorted(
            available_source_ids[index] for index in indices[:selection_source_video_count]
        )
    else:
        raise ValueError("Select at least one pseudo-real source video for checkpoint selection")

    selection_id_set = set(selection_ids)
    train_rows = [row for row in rows if row.source_video_id not in selection_id_set]
    selection_rows = [row for row in rows if row.source_video_id in selection_id_set]
    if max_train_frames > 0 and len(train_rows) > max_train_frames:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_rows), generator=generator).tolist()[:max_train_frames]
        train_rows = [train_rows[index] for index in indices]
    if not train_rows or not selection_rows:
        raise ValueError("Pseudo-real train/selection split produced an empty dataset")

    train_dataset = PhysicalRealObliqueBoardDataset(
        rows=to_oblique_rows(train_rows),
        image_size=image_size,
    )
    selection_dataset = PhysicalRealObliqueBoardDataset(
        rows=to_oblique_rows(selection_rows),
        image_size=image_size,
    )
    train_source_ids = sorted({row.source_video_id for row in train_rows if row.source_video_id})
    return train_dataset, selection_dataset, train_source_ids, selection_ids


def to_oblique_rows(rows: list[PhysicalRealBoardRow]) -> list[PhysicalRealObliqueBoardRow]:
    return [
        PhysicalRealObliqueBoardRow(
            clip_path=row.clip_path,
            frame_index=row.frame_index,
            source_video_id=row.source_video_id,
            source_channel_handle=row.source_channel_handle,
            corners=row.corners,
            labels=row.labels,
        )
        for row in rows
    ]


def build_teacher_logits_for_real_rows(
    *,
    rows: list[PhysicalRealObliqueBoardRow],
    image_size: int,
    device: str,
    weights_path: Path,
    batch_size: int,
) -> torch.Tensor:
    clip_cache: dict[Path, dict[str, Any]] = {}
    rectified_images = []
    for row in rows:
        clip_path = (_PROJECT_ROOT / row.clip_path).resolve()
        clip = clip_cache.get(clip_path)
        if clip is None:
            loaded = torch.load(clip_path, map_location="cpu", weights_only=False)
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid clip payload: {row.clip_path}")
            clip_cache[clip_path] = loaded
            clip = loaded
        frames = clip.get("frames")
        if not isinstance(frames, torch.Tensor):
            raise ValueError(f"Clip has no frames tensor: {row.clip_path}")
        image_rgb = _frame_tensor_to_rgb(frames[row.frame_index])
        rectified_rgb = rectify_board_image(
            image_rgb,
            [list(point) for point in row.corners],
            output_size=image_size,
        )
        rectified_images.append(rectified_rgb[:, :, ::-1].copy())

    teacher_logits = read_board_logits_batch_from_frames(
        rectified_images,
        device=device,
        weights_path=weights_path,
        batch_size=batch_size,
    )
    if teacher_logits is None:
        raise ValueError(f"Failed to load teacher logits from {weights_path}")
    return torch.stack(teacher_logits, dim=0).to(torch.float32)


def build_class_weights(labels: torch.Tensor, *, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels.reshape(-1), minlength=13).float()
    return counts.max() / counts.clamp_min(1.0)


def build_board_weights(
    *,
    synthetic_count: int,
    real_count: int,
    real_loss_weight: float,
) -> torch.Tensor | None:
    if synthetic_count == 0 and real_loss_weight == 1.0:
        return None
    total = synthetic_count + real_count
    weights = torch.ones(total, dtype=torch.float32)
    if real_count > 0:
        weights[synthetic_count:] = real_loss_weight
    return weights


def annotation_ids_for_dataset(dataset: Dataset[Any]) -> list[str]:
    rows = getattr(dataset, "rows", None)
    if rows is None:
        raise ValueError("Dataset does not expose rows metadata")
    annotation_ids: list[str] = []
    for row in rows:
        clip_stem = Path(row.clip_path).stem
        annotation_ids.append(f"{clip_stem}_frame{int(row.frame_index):04d}")
    return annotation_ids


def evaluate_per_source_metrics(
    model,
    patch_tokens: torch.Tensor,
    labels: torch.Tensor,
    corners: torch.Tensor,
    *,
    annotation_ids: list[str],
    source_video_ids: list[str],
    device: torch.device,
) -> dict[str, ProbeMetrics]:
    grouped_indices: dict[str, list[int]] = {}
    for index, source_video_id in enumerate(source_video_ids):
        grouped_indices.setdefault(source_video_id, []).append(index)

    metrics_by_source: dict[str, ProbeMetrics] = {}
    for source_video_id, indices in grouped_indices.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        source_annotation_ids = [annotation_ids[index] for index in indices]
        metrics_by_source[source_video_id] = evaluate_joint_board_reader(
            model,
            patch_tokens.index_select(0, index_tensor),
            labels.index_select(0, index_tensor),
            corners.index_select(0, index_tensor),
            device=device,
            annotation_ids=source_annotation_ids,
        )
    return dict(sorted(metrics_by_source.items()))


if __name__ == "__main__":
    main()
