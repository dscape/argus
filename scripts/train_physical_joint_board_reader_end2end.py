#!/usr/bin/env python3
"""Train an end-to-end oblique whole-board reader with partial encoder unfreezing."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.end_to_end_joint_board_reader import (
    EndToEndJointBoardReaderConfig,
    EndToEndPhysicalJointBoardReader,
    evaluate_end_to_end_joint_board_reader,
    train_end_to_end_joint_board_reader,
)
from pipeline.physical.oblique_board_data import (
    PhysicalEvalObliqueBoardDataset,
    PhysicalRealObliqueBoardDataset,
    PhysicalSyntheticWarpedObliqueBoardDataset,
)
from pipeline.physical.oblique_square_context import PhysicalRealObliqueBoardRow
from pipeline.physical.real_board_data import PhysicalRealBoardRow, load_real_board_rows

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_joint_board_reader_end2end"


def main() -> None:
    args = build_parser().parse_args()
    args.model_name = args.model_name or default_model_name_for_encoder_type(args.encoder_type)
    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        real_train_dataset,
        selection_dataset,
        train_source_ids,
        selection_source_ids,
    ) = build_real_datasets(
        clips_dir=args.real_train_clips_dir,
        image_size=args.input_size,
        frame_stride=args.real_train_frame_stride,
        max_train_frames=args.real_train_max_frames,
        seed=args.seed,
        exclude_move_neighborhood=args.real_train_exclude_move_neighborhood,
        selection_source_video_ids=parse_source_video_ids(args.real_val_source_videos),
        selection_source_video_count=args.real_val_source_video_count,
    )
    train_parts: list[Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    if args.synthetic_train_positions > 0:
        train_parts.append(
            PhysicalSyntheticWarpedObliqueBoardDataset(
                num_positions=args.synthetic_train_positions,
                image_size=args.input_size,
                seed=args.seed,
                augment=args.synthetic_augment,
                min_moves=args.synthetic_min_moves,
                max_moves=args.synthetic_max_moves,
                min_ply=args.synthetic_min_ply,
            )
        )
    train_parts.append(real_train_dataset)
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)

    eval_dataset = PhysicalEvalObliqueBoardDataset(image_size=args.input_size)
    selection_annotation_ids = annotation_ids_for_dataset(selection_dataset)
    eval_annotation_ids = [row.annotation_id for row in eval_dataset.rows]
    eval_source_ids = [row.source_video_id or "unknown" for row in eval_dataset.rows]

    encoder = VisionEncoder(
        encoder_type=args.encoder_type,
        model_name=args.model_name,
        frozen=True,
        feature_layer_indices=parse_feature_layer_indices(args.feature_layer_indices),
    )
    if args.unfreeze_last_n_layers > 0:
        encoder.unfreeze_last_n_layers(args.unfreeze_last_n_layers)
    model = EndToEndPhysicalJointBoardReader(
        vision_encoder=encoder,
        config=EndToEndJointBoardReaderConfig(
            input_size=args.input_size,
            num_classes=13,
            square_query_num_heads=args.square_query_num_heads,
            square_query_dropout=args.square_query_dropout,
            square_query_mlp_ratio=args.square_query_mlp_ratio,
            head_type=args.head_type,
            hidden_dim=args.hidden_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_ff_dim=args.transformer_ff_dim,
            dropout=args.dropout,
        ),
    )
    class_weights = None
    best_model, best_selection_score = train_end_to_end_joint_board_reader(
        model,
        train_dataset,
        selection_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr_scale=args.encoder_lr_scale,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        selection_metric=args.selection_metric,
    )

    selection_metrics = evaluate_end_to_end_joint_board_reader(
        best_model,
        selection_dataset,
        device=device,
        annotation_ids=selection_annotation_ids,
        batch_size=args.batch_size,
    )
    eval_metrics = evaluate_end_to_end_joint_board_reader(
        best_model,
        eval_dataset,
        device=device,
        annotation_ids=eval_annotation_ids,
        batch_size=args.batch_size,
    )
    per_source_eval_metrics = evaluate_per_source_metrics(
        best_model,
        eval_dataset,
        annotation_ids=eval_annotation_ids,
        source_video_ids=eval_source_ids,
        device=device,
        batch_size=args.batch_size,
    )

    checkpoint = {
        "architecture": "end_to_end_joint_oblique_board_reader",
        "state_dict": best_model.state_dict(),
        "encoder_config": {
            "encoder_type": args.encoder_type,
            "model_name": args.model_name,
            "feature_layer_indices": parse_feature_layer_indices(args.feature_layer_indices),
            "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        },
        **best_model.checkpoint_config(),
        "selection_score": best_selection_score,
        "selection_metrics": selection_metrics.to_dict(),
        "eval_metrics": eval_metrics.to_dict(),
        "per_source_eval_metrics": {k: v.to_dict() for k, v in per_source_eval_metrics.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / "joint_board_reader_end2end.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "train_frames": len(train_dataset),
        "selection_frames": len(selection_dataset),
        "eval_frames": len(eval_dataset),
        "train_source_video_ids": train_source_ids,
        "selection_source_video_ids": selection_source_ids,
        "selection_score": best_selection_score,
        "selection_metric": args.selection_metric,
        "selection_metrics": selection_metrics.to_dict(),
        "eval_metrics": eval_metrics.to_dict(),
        "per_source_eval_metrics": {k: v.to_dict() for k, v in per_source_eval_metrics.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("Eval metrics: %s", json.dumps(eval_metrics.to_dict(), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train end-to-end oblique joint board reader")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-type",
        choices=("siglip2", "siglip", "dinov2"),
        default="siglip",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--feature-layer-indices", type=str, default="")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=2)
    parser.add_argument("--square-query-num-heads", type=int, default=8)
    parser.add_argument("--square-query-dropout", type=float, default=0.0)
    parser.add_argument("--square-query-mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--head-type",
        choices=("linear", "pos_mlp", "transformer"),
        default="pos_mlp",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument("--real-train-max-frames", type=int, default=512)
    parser.add_argument("--real-train-exclude-move-neighborhood", type=int, default=-1)
    parser.add_argument("--real-val-source-video-count", type=int, default=0)
    parser.add_argument("--real-val-source-videos", type=str, default="psrPAoHr4wA")
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


def annotation_ids_for_dataset(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> list[str]:
    rows = getattr(dataset, "rows")
    return [f"{Path(row.clip_path).stem}_frame{int(row.frame_index):04d}" for row in rows]


def evaluate_per_source_metrics(
    model: EndToEndPhysicalJointBoardReader,
    dataset: PhysicalEvalObliqueBoardDataset,
    *,
    annotation_ids: list[str],
    source_video_ids: list[str],
    device: torch.device,
    batch_size: int,
):
    metrics = {}
    for source_video_id in sorted(set(source_video_ids)):
        rows = [
            row for row in dataset.rows if (row.source_video_id or "unknown") == source_video_id
        ]
        source_dataset = PhysicalEvalObliqueBoardDataset(
            image_size=dataset.image_size,
            rows=rows,
        )
        source_annotation_ids = [
            annotation_id
            for annotation_id, sid in zip(annotation_ids, source_video_ids)
            if sid == source_video_id
        ]
        metrics[source_video_id] = evaluate_end_to_end_joint_board_reader(
            model,
            source_dataset,
            device=device,
            annotation_ids=source_annotation_ids,
            batch_size=batch_size,
        )
    return metrics


if __name__ == "__main__":
    main()
