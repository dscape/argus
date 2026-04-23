#!/usr/bin/env python3
"""Train a frozen-feature board-context probe for physical per-square state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.board_probe.board_data import INPUT_SIZE as DEFAULT_INPUT_SIZE
from pipeline.physical.board_probe.board_data import (
    PhysicalEvalBoardDataset,
    PhysicalManualTrainBoardDataset,
    PhysicalSyntheticClipBoardDataset,
)
from pipeline.physical.board_probe.decoder import production_decoder_config
from pipeline.physical.board_probe.probe import (
    board_probe_config_from_checkpoint,
    evaluate_board_probe,
    extract_square_token_features,
    save_board_probe_checkpoint,
    train_board_probe,
)
from pipeline.physical.board_probe.square_probe import ProbeMetrics, load_probe_checkpoint
from pipeline.physical.shared.real_board_data import (
    PhysicalRealBoardDataset,
    PhysicalRealBoardRow,
    load_real_board_rows,
)
from pipeline.physical.shared.training_receipts import write_training_receipt
from pipeline.shared import SQUARE_CLASS_NAMES

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_board_probe"
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_MODEL_CODE_VERSION = "v8"
_BOARD_INPUT_MODE = "piece_projection_board"
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_kwargs = build_encoder_kwargs(args)
    logger.info(
        "Loading frozen %s encoder %s on %s",
        args.encoder_type,
        encoder_kwargs["model_name"],
        device,
    )
    encoder = VisionEncoder(**encoder_kwargs).to(device)
    encoder.eval()

    synthetic_train_dataset: Dataset[Any] | None = None
    if args.synthetic_train_positions > 0:
        synthetic_train_dataset = build_synthetic_dataset(
            synthetic_clips_dir=args.synthetic_clips_dir,
            board_input_mode=args.board_input_mode,
            num_positions=args.synthetic_train_positions,
            image_size=args.input_size,
            seed=args.seed,
        )
    val_dataset: Dataset[Any] | None = None
    if args.synthetic_val_positions > 0:
        val_dataset = build_synthetic_dataset(
            synthetic_clips_dir=args.synthetic_clips_dir,
            board_input_mode=args.board_input_mode,
            num_positions=args.synthetic_val_positions,
            image_size=args.input_size,
            seed=args.seed + 1,
        )
    eval_dataset = build_eval_dataset(
        board_input_mode=args.board_input_mode,
        image_size=args.input_size,
    )
    eval_annotation_ids = [row.annotation_id for row in eval_dataset.rows]

    real_train_dataset: Dataset[Any] | None = None
    real_selection_dataset: Dataset[Any] | None = None
    real_train_source_video_ids: list[str] = []
    real_selection_source_video_ids: list[str] = []
    manual_train_dataset: Dataset[Any] | None = None
    train_parts: list[Dataset[Any]] = []
    if synthetic_train_dataset is not None:
        train_parts.append(synthetic_train_dataset)
    if args.real_train_max_frames > 0:
        (
            real_train_dataset,
            real_selection_dataset,
            real_train_source_video_ids,
            real_selection_source_video_ids,
        ) = build_real_train_and_selection_datasets(
            clips_dir=args.real_train_clips_dir,
            board_input_mode=args.board_input_mode,
            image_size=args.input_size,
            frame_stride=args.real_train_frame_stride,
            max_train_frames=args.real_train_max_frames,
            seed=args.seed,
            exclude_move_neighborhood=args.real_train_exclude_move_neighborhood,
            real_val_source_video_count=args.real_val_source_video_count,
            real_val_source_videos=parse_source_video_ids(args.real_val_source_videos),
        )
        if real_train_dataset is not None and len(real_train_dataset) > 0:
            train_parts.append(real_train_dataset)
    if args.manual_train_max_boards > 0:
        manual_train_dataset = build_manual_train_dataset(
            annotation_root=args.manual_train_root,
            board_input_mode=args.board_input_mode,
            image_size=args.input_size,
            max_boards=args.manual_train_max_boards,
            seed=args.seed,
        )
        if manual_train_dataset is not None and len(manual_train_dataset) > 0:
            train_parts.append(manual_train_dataset)
    if not train_parts:
        raise ValueError("No training datasets selected; add synthetic or real training data")

    train_dataset: Dataset[Any]
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)

    synthetic_train_positions = (
        0 if synthetic_train_dataset is None else len(synthetic_train_dataset)
    )
    synthetic_val_positions = 0 if val_dataset is None else len(val_dataset)
    real_train_positions = 0 if real_train_dataset is None else len(real_train_dataset)
    real_selection_positions = 0 if real_selection_dataset is None else len(real_selection_dataset)
    manual_train_positions = 0 if manual_train_dataset is None else len(manual_train_dataset)
    if real_selection_positions > 0:
        selection_dataset_name = "real_selection"
    elif synthetic_val_positions > 0:
        selection_dataset_name = "synthetic_val"
    else:
        raise ValueError("Selection requires synthetic val data or a pseudo-real selection split")
    selection_metric = resolve_selection_metric(
        args.selection_metric,
        has_real_selection=real_selection_positions > 0,
    )

    if args.save_samples > 0:
        if synthetic_train_dataset is not None:
            save_board_samples(
                synthetic_train_dataset,
                output_dir,
                count=args.save_samples,
                prefix="synthetic_train",
            )
        if real_train_dataset is not None and len(real_train_dataset) > 0:
            save_board_samples(
                real_train_dataset,
                output_dir,
                count=args.save_samples,
                prefix="real_train",
            )
        if manual_train_dataset is not None and len(manual_train_dataset) > 0:
            save_board_samples(
                manual_train_dataset,
                output_dir,
                count=args.save_samples,
                prefix="manual_train",
            )
        if real_selection_dataset is not None and len(real_selection_dataset) > 0:
            save_board_samples(
                real_selection_dataset,
                output_dir,
                count=args.save_samples,
                prefix="real_selection",
            )
        if val_dataset is not None:
            save_board_samples(
                val_dataset,
                output_dir,
                count=args.save_samples,
                prefix="synthetic_val",
            )
        save_board_samples(
            eval_dataset,
            output_dir,
            count=args.save_samples,
            prefix="real_eval",
        )

    logger.info("Extracting train square-token features")
    train_tokens, train_labels = extract_square_token_features(
        train_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    real_selection_tokens: torch.Tensor | None = None
    real_selection_labels: torch.Tensor | None = None
    if real_selection_dataset is not None and len(real_selection_dataset) > 0:
        logger.info("Extracting pseudo-real selection square-token features")
        real_selection_tokens, real_selection_labels = extract_square_token_features(
            real_selection_dataset,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
        )

    val_tokens: torch.Tensor
    val_labels: torch.Tensor
    if val_dataset is not None:
        logger.info("Extracting synthetic val square-token features")
        val_tokens, val_labels = extract_square_token_features(
            val_dataset,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
        )
    elif real_selection_tokens is not None and real_selection_labels is not None:
        logger.info("Reusing pseudo-real selection features as validation features")
        val_tokens, val_labels = real_selection_tokens, real_selection_labels
    else:
        raise ValueError("Validation requires synthetic val data or pseudo-real selection data")
    logger.info("Extracting held-out real board features")
    eval_tokens, eval_labels = extract_square_token_features(
        eval_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    class_weights = build_class_weights(train_labels, mode=args.class_weighting)
    board_weights = build_board_weights(
        synthetic_count=synthetic_train_positions,
        real_count=real_train_positions,
        manual_count=manual_train_positions,
        real_loss_weight=args.real_loss_weight,
        manual_loss_weight=args.manual_train_loss_weight,
    )
    initial_probe_state = load_initial_probe_state(args.initialize_checkpoint)
    probe, best_selection_score = train_board_probe(
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        board_weights=board_weights,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        selection_square_tokens=real_selection_tokens,
        selection_labels=real_selection_labels,
        selection_metric=selection_metric,
        initial_state_dict=initial_probe_state,
    )

    train_metrics = evaluate_board_probe(probe, train_tokens, train_labels, device=device)
    synth_val_metrics = (
        None
        if val_dataset is None
        else evaluate_board_probe(probe, val_tokens, val_labels, device=device)
    )
    real_selection_metrics = None
    if real_selection_tokens is not None and real_selection_labels is not None:
        real_selection_metrics = evaluate_board_probe(
            probe,
            real_selection_tokens,
            real_selection_labels,
            device=device,
        )
    real_eval_metrics = evaluate_board_probe(
        probe,
        eval_tokens,
        eval_labels,
        device=device,
        annotation_ids=eval_annotation_ids,
    )

    write_training_receipt(
        output_dir,
        kind="board_probe",
        entries=_collect_probe_receipt_entries(
            real_train_dataset=real_train_dataset,
            manual_train_dataset=manual_train_dataset,
        ),
    )

    checkpoint_path = output_dir / "board_probe.pt"
    save_board_probe_checkpoint(
        checkpoint_path,
        probe=probe.cpu(),
        model_name=str(encoder_kwargs["model_name"]),
        input_size=args.input_size,
        metadata={
            "encoder_type": args.encoder_type,
            "feature_layer_indices": encoder_kwargs.get("feature_layer_indices"),
            "output_grid_size": encoder_kwargs.get("output_grid_size"),
            "board_input_mode": args.board_input_mode,
            "synthetic_source": relative_to_project(args.synthetic_clips_dir),
            "synthetic_clips_dir": relative_to_project(args.synthetic_clips_dir),
            "synthetic_train_positions": synthetic_train_positions,
            "real_train_positions": real_train_positions,
            "manual_train_positions": manual_train_positions,
            "synthetic_val_positions": synthetic_val_positions,
            "class_weighting": args.class_weighting,
            "real_train_exclude_move_neighborhood": args.real_train_exclude_move_neighborhood,
            "real_loss_weight": args.real_loss_weight,
            "manual_train_root": (
                relative_to_project(args.manual_train_root)
                if args.manual_train_root is not None
                else None
            ),
            "manual_train_loss_weight": args.manual_train_loss_weight,
            "head_type": args.head_type,
            "hidden_dim": args.hidden_dim,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "transformer_ff_dim": args.transformer_ff_dim,
            "dropout": args.dropout,
            "selection_dataset": selection_dataset_name,
            "selection_metric": selection_metric,
            "best_selection_score": best_selection_score,
            "real_selection_positions": real_selection_positions,
            "real_train_source_video_ids": real_train_source_video_ids,
            "real_selection_source_video_ids": real_selection_source_video_ids,
            "initialize_checkpoint": (
                None
                if args.initialize_checkpoint is None
                else relative_to_project(args.initialize_checkpoint)
            ),
        },
    )

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "encoder_type": args.encoder_type,
        "model_name": str(encoder_kwargs["model_name"]),
        "feature_layer_indices": encoder_kwargs.get("feature_layer_indices"),
        "output_grid_size": encoder_kwargs.get("output_grid_size"),
        "board_input_mode": args.board_input_mode,
        "device": str(device),
        "input_size": args.input_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "synthetic_source": relative_to_project(args.synthetic_clips_dir),
        "synthetic_clips_dir": relative_to_project(args.synthetic_clips_dir),
        "class_weighting": args.class_weighting,
        "real_loss_weight": args.real_loss_weight,
        "head_type": args.head_type,
        "hidden_dim": args.hidden_dim,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_ff_dim": args.transformer_ff_dim,
        "dropout": args.dropout,
        "synthetic_train_positions": synthetic_train_positions,
        "real_train_positions": real_train_positions,
        "real_selection_positions": real_selection_positions,
        "manual_train_positions": manual_train_positions,
        "synthetic_val_positions": synthetic_val_positions,
        "real_train_exclude_move_neighborhood": args.real_train_exclude_move_neighborhood,
        "manual_train_root": (
            relative_to_project(args.manual_train_root)
            if args.manual_train_root is not None
            else None
        ),
        "manual_train_loss_weight": args.manual_train_loss_weight,
        "selection_dataset": selection_dataset_name,
        "selection_metric": selection_metric,
        "best_selection_score": best_selection_score,
        "initialize_checkpoint": (
            None
            if args.initialize_checkpoint is None
            else relative_to_project(args.initialize_checkpoint)
        ),
        "real_train_source_video_ids": real_train_source_video_ids,
        "real_selection_source_video_ids": real_selection_source_video_ids,
        "real_eval_positions": len(eval_dataset),
        "train_label_histogram": class_histogram(train_labels.reshape(-1)),
        "synthetic_val_label_histogram": (
            None if val_dataset is None else class_histogram(val_labels.reshape(-1))
        ),
        "real_selection_label_histogram": (
            None
            if real_selection_labels is None
            else class_histogram(real_selection_labels.reshape(-1))
        ),
        "real_eval_label_histogram": class_histogram(eval_labels.reshape(-1)),
        "probe_config": board_probe_config_from_checkpoint(
            {"probe_config": probe.checkpoint_config()}
        ),
        "train_metrics": train_metrics.to_dict(),
        "synthetic_val_metrics": (
            None if synth_val_metrics is None else synth_val_metrics.to_dict()
        ),
        "real_selection_metrics": (
            None if real_selection_metrics is None else real_selection_metrics.to_dict()
        ),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "checkpoint": relative_to_project(checkpoint_path),
    }
    write_json(output_dir / "metrics.json", report)

    summary_lines = [
        "# Physical board-context probe",
        "",
        f"- encoder: `{args.encoder_type}`",
        f"- model: `{str(encoder_kwargs['model_name'])}`",
        f"- synthetic clips dir: `{relative_to_project(args.synthetic_clips_dir)}`",
        f"- board input mode: `{args.board_input_mode}`",
        f"- input size: `{args.input_size}`",
        f"- head type: `{args.head_type}`",
        f"- real loss weight: `{args.real_loss_weight}`",
        f"- class weighting: `{args.class_weighting}`",
        f"- real move exclusion: `{args.real_train_exclude_move_neighborhood}`",
        f"- selection dataset: `{selection_dataset_name}`",
        f"- selection metric: `{selection_metric}`",
        (
            f"- initialize checkpoint: `{relative_to_project(args.initialize_checkpoint)}`"
            if args.initialize_checkpoint is not None
            else "- initialize checkpoint: `none`"
        ),
        f"- best selection score: `{best_selection_score:.4f}`",
        f"- synthetic train positions: `{synthetic_train_positions}`",
        f"- real train positions: `{real_train_positions}`",
        f"- real selection positions: `{real_selection_positions}`",
        f"- manual train positions: `{manual_train_positions}`",
        f"- synthetic val positions: `{synthetic_val_positions}`",
        f"- real eval positions: `{len(eval_dataset)}`",
        f"- train square accuracy: `{train_metrics.accuracy:.4f}`",
        f"- real eval square accuracy: `{real_eval_metrics.accuracy:.4f}`",
        f"- real eval non-empty accuracy: `{real_eval_metrics.non_empty_accuracy:.4f}`",
        f"- real eval macro F1: `{real_eval_metrics.macro_f1:.4f}`",
        f"- real eval board exact match: `{(real_eval_metrics.board_exact_match or 0.0):.4f}`",
        "",
    ]
    if synth_val_metrics is not None:
        summary_lines.extend(
            [
                f"- synthetic val square accuracy: `{synth_val_metrics.accuracy:.4f}`",
                "",
            ]
        )
    if real_selection_metrics is not None:
        summary_lines.extend(
            [
                f"- real selection source videos: `{','.join(real_selection_source_video_ids)}`",
                f"- real selection square accuracy: `{real_selection_metrics.accuracy:.4f}`",
                (
                    "- real selection non-empty accuracy: "
                    f"`{real_selection_metrics.non_empty_accuracy:.4f}`"
                ),
                f"- real selection macro F1: `{real_selection_metrics.macro_f1:.4f}`",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    if args.promote_to_weights:
        promote_to_runtime_weights(
            checkpoint_path=checkpoint_path,
            encoder_kwargs=encoder_kwargs,
            args=args,
            probe_config=probe.checkpoint_config(),
            real_eval_metrics=real_eval_metrics,
            best_selection_score=best_selection_score,
            selection_dataset_name=selection_dataset_name,
            selection_metric=selection_metric,
            held_out_eval_size=len(eval_dataset),
            real_train_positions=real_train_positions,
            real_selection_positions=real_selection_positions,
            manual_train_positions=manual_train_positions,
        )

    logger.info("Train square accuracy: %.4f", train_metrics.accuracy)
    if synth_val_metrics is not None:
        logger.info("Synthetic val square accuracy: %.4f", synth_val_metrics.accuracy)
    if real_selection_metrics is not None:
        logger.info("Real selection square accuracy: %.4f", real_selection_metrics.accuracy)
        logger.info(
            "Real selection non-empty accuracy: %.4f",
            real_selection_metrics.non_empty_accuracy,
        )
        logger.info("Real selection macro F1: %.4f", real_selection_metrics.macro_f1)
    logger.info("Real eval square accuracy: %.4f", real_eval_metrics.accuracy)
    logger.info("Real eval non-empty accuracy: %.4f", real_eval_metrics.non_empty_accuracy)
    logger.info("Real eval macro F1: %.4f", real_eval_metrics.macro_f1)
    logger.info("Real eval board exact match: %.4f", real_eval_metrics.board_exact_match or 0.0)
    logger.info("Saved report to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train physical board-context square probe")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--board-input-mode",
        type=str,
        choices=[_BOARD_INPUT_MODE],
        default=_BOARD_INPUT_MODE,
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["dinov2", "siglip", "siglip2", "yolo"],
        default="dinov2",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dino-feature-layer-indices", type=str, default="")
    parser.add_argument("--yolo-feature-layer-indices", type=str, default="16,19,22")
    parser.add_argument("--yolo-output-grid-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--synthetic-clips-dir", type=Path, default=Path("data/argus/train"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--head-type",
        type=str,
        choices=["linear", "pos_mlp", "transformer"],
        default="linear",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--synthetic-train-positions", type=int, default=1200)
    parser.add_argument("--synthetic-val-positions", type=int, default=300)
    parser.add_argument("--real-train-clips-dir", type=Path, default=Path("data/argus/train_real"))
    parser.add_argument("--real-train-max-frames", type=int, default=0)
    parser.add_argument("--real-train-frame-stride", type=int, default=4)
    parser.add_argument("--real-train-exclude-move-neighborhood", type=int, default=-1)
    parser.add_argument("--real-val-source-video-count", type=int, default=0)
    parser.add_argument("--real-val-source-videos", type=str, default="")
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=[
            "auto",
            "accuracy",
            "non_empty_accuracy",
            "macro_f1",
            "non_empty_plus_macro",
        ],
        default="auto",
    )
    parser.add_argument(
        "--manual-train-root",
        type=Path,
        default=Path("data/physical/train_manual"),
    )
    parser.add_argument("--manual-train-max-boards", type=int, default=0)
    parser.add_argument("--real-loss-weight", type=float, default=1.0)
    parser.add_argument("--manual-train-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--class-weighting",
        type=str,
        choices=["none", "max_ratio"],
        default="none",
    )
    parser.add_argument("--save-samples", type=int, default=0)
    parser.add_argument("--initialize-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--promote-to-weights",
        action="store_true",
        help="Copy the trained board probe to weights/physical/ for runtime use",
    )
    return parser


def load_initial_probe_state(
    checkpoint_path: Path | None,
) -> dict[str, torch.Tensor] | None:
    if checkpoint_path is None:
        return None
    checkpoint = load_probe_checkpoint(checkpoint_path)
    architecture = str(checkpoint.get("architecture", "board_probe"))
    if architecture != "board_probe":
        raise ValueError(f"Expected board_probe checkpoint, got {architecture}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint is missing state_dict: {checkpoint_path}")
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def _collect_probe_receipt_entries(
    *,
    real_train_dataset: Dataset[Any] | None,
    manual_train_dataset: Dataset[Any] | None,
) -> list[dict[str, str | int | None]]:
    entries: list[dict[str, str | int | None]] = []
    seen_clips: set[str] = set()
    rows = list(getattr(real_train_dataset, "rows", []) or [])
    for row in rows:
        clip_path = getattr(row, "clip_path", None)
        if not isinstance(clip_path, str) or clip_path in seen_clips:
            continue
        seen_clips.add(clip_path)
        entries.append(
            {
                "source": "real_train",
                "clip_path": clip_path,
                "source_channel_handle": getattr(row, "source_channel_handle", None),
                "source_video_id": getattr(row, "source_video_id", None),
            }
        )
    manual_rows = list(getattr(manual_train_dataset, "rows", []) or [])
    for row in manual_rows:
        annotation_id = getattr(row, "annotation_id", None)
        if not isinstance(annotation_id, str):
            continue
        entries.append(
            {
                "source": "manual_train",
                "annotation_id": annotation_id,
                "clip_path": getattr(row, "clip_path", None),
                "source_video_id": getattr(row, "source_video_id", None),
            }
        )
    return entries


def build_encoder_kwargs(args: argparse.Namespace) -> dict[str, object]:
    model_name = args.model_name or default_model_name_for_encoder_type(args.encoder_type)
    encoder_kwargs: dict[str, object] = {
        "model_name": model_name,
        "frozen": True,
        "encoder_type": args.encoder_type,
    }
    if (
        args.encoder_type in {"dinov2", "siglip", "siglip2"}
        and args.dino_feature_layer_indices.strip()
    ):
        encoder_kwargs["feature_layer_indices"] = parse_feature_layer_indices(
            args.dino_feature_layer_indices
        )
    if args.encoder_type == "yolo":
        encoder_kwargs["feature_layer_indices"] = parse_feature_layer_indices(
            args.yolo_feature_layer_indices
        )
        encoder_kwargs["output_grid_size"] = args.yolo_output_grid_size
    return encoder_kwargs


def parse_feature_layer_indices(raw_value: str) -> list[int]:
    indices = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not indices:
        raise ValueError("yolo feature layer indices must not be empty")
    return [int(index) for index in indices]


def build_synthetic_dataset(
    *,
    synthetic_clips_dir: Path,
    board_input_mode: str,
    num_positions: int,
    image_size: int,
    seed: int,
) -> Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if board_input_mode != _BOARD_INPUT_MODE:
        raise ValueError(f"Unsupported board_input_mode: {board_input_mode}")
    return PhysicalSyntheticClipBoardDataset(
        clips_dir=synthetic_clips_dir,
        num_positions=num_positions,
        image_size=image_size,
        seed=seed,
    )


def build_eval_dataset(
    *,
    board_input_mode: str,
    image_size: int,
) -> Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if board_input_mode != _BOARD_INPUT_MODE:
        raise ValueError(f"Unsupported board_input_mode: {board_input_mode}")
    return PhysicalEvalBoardDataset(image_size=image_size)


def parse_source_video_ids(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def resolve_selection_metric(raw_metric: str, *, has_real_selection: bool) -> str:
    if raw_metric == "auto":
        return "non_empty_plus_macro" if has_real_selection else "accuracy"
    return raw_metric


def dataset_from_real_rows(
    *,
    board_input_mode: str,
    clips_dir: Path,
    image_size: int,
    rows: list[PhysicalRealBoardRow],
) -> Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    if not rows:
        return None
    if board_input_mode != _BOARD_INPUT_MODE:
        raise ValueError(f"Unsupported board_input_mode: {board_input_mode}")
    return PhysicalRealBoardDataset(
        clips_dir=clips_dir,
        image_size=image_size,
        rows=rows,
    )


def build_real_train_and_selection_datasets(
    *,
    clips_dir: Path,
    board_input_mode: str,
    image_size: int,
    frame_stride: int,
    max_train_frames: int,
    seed: int,
    exclude_move_neighborhood: int,
    real_val_source_video_count: int,
    real_val_source_videos: list[str],
) -> tuple[
    Dataset[Any] | None,
    Dataset[Any] | None,
    list[str],
    list[str],
]:
    all_rows = load_real_board_rows(
        clips_dir=clips_dir,
        frame_stride=frame_stride,
        max_frames=None,
        seed=seed,
        exclude_move_neighborhood=exclude_move_neighborhood,
    )
    if not all_rows:
        return None, None, [], []

    selection_source_video_ids = select_real_val_source_video_ids(
        all_rows,
        requested_source_video_ids=real_val_source_videos,
        requested_count=real_val_source_video_count,
        seed=seed,
    )
    train_rows, selection_rows = split_real_rows_by_source_video_ids(
        all_rows,
        selection_source_video_ids=selection_source_video_ids,
    )
    sampled_train_rows = sample_real_rows(train_rows, max_frames=max_train_frames, seed=seed)
    real_train_dataset = dataset_from_real_rows(
        board_input_mode=board_input_mode,
        clips_dir=clips_dir,
        image_size=image_size,
        rows=sampled_train_rows,
    )
    real_selection_dataset = dataset_from_real_rows(
        board_input_mode=board_input_mode,
        clips_dir=clips_dir,
        image_size=image_size,
        rows=selection_rows,
    )
    real_train_source_video_ids = sorted(
        {row.source_video_id for row in sampled_train_rows if row.source_video_id is not None}
    )
    return (
        real_train_dataset,
        real_selection_dataset,
        real_train_source_video_ids,
        selection_source_video_ids,
    )


def select_real_val_source_video_ids(
    rows: list[PhysicalRealBoardRow],
    *,
    requested_source_video_ids: list[str],
    requested_count: int,
    seed: int,
) -> list[str]:
    available_source_video_ids = sorted(
        {row.source_video_id for row in rows if row.source_video_id is not None}
    )
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
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(available_source_video_ids), generator=rng).tolist()
    return sorted(available_source_video_ids[index] for index in indices[:requested_count])


def split_real_rows_by_source_video_ids(
    rows: list[PhysicalRealBoardRow],
    *,
    selection_source_video_ids: list[str],
) -> tuple[list[PhysicalRealBoardRow], list[PhysicalRealBoardRow]]:
    selection_source_video_id_set = set(selection_source_video_ids)
    train_rows: list[PhysicalRealBoardRow] = []
    selection_rows: list[PhysicalRealBoardRow] = []
    for row in rows:
        if row.source_video_id in selection_source_video_id_set:
            selection_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, selection_rows


def sample_real_rows(
    rows: list[PhysicalRealBoardRow],
    *,
    max_frames: int,
    seed: int,
) -> list[PhysicalRealBoardRow]:
    if max_frames <= 0 or len(rows) <= max_frames:
        return list(rows)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(rows), generator=generator).tolist()[:max_frames]
    return [rows[index] for index in indices]


def build_class_weights(labels: torch.Tensor, *, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "max_ratio":
        raise ValueError(f"Unsupported class weighting mode: {mode}")
    class_counts = torch.bincount(labels.reshape(-1), minlength=len(SQUARE_CLASS_NAMES)).float()
    return class_counts.max() / class_counts.clamp_min(1.0)


def build_board_weights(
    *,
    synthetic_count: int,
    real_count: int,
    manual_count: int,
    real_loss_weight: float,
    manual_loss_weight: float,
) -> torch.Tensor | None:
    total_count = synthetic_count + real_count + manual_count
    if total_count == synthetic_count:
        return None
    if real_loss_weight <= 0.0:
        raise ValueError(f"real_loss_weight must be > 0, got {real_loss_weight}")
    if manual_loss_weight <= 0.0:
        raise ValueError(f"manual_train_loss_weight must be > 0, got {manual_loss_weight}")
    weights = torch.ones(total_count, dtype=torch.float32)
    real_start = synthetic_count
    real_end = real_start + real_count
    if real_count > 0:
        weights[real_start:real_end] = real_loss_weight
    if manual_count > 0:
        weights[real_end:] = manual_loss_weight
    return weights


def build_manual_train_dataset(
    *,
    annotation_root: Path,
    board_input_mode: str,
    image_size: int,
    max_boards: int,
    seed: int,
) -> Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    if max_boards <= 0 or not annotation_root.exists():
        return None
    if board_input_mode != _BOARD_INPUT_MODE:
        raise ValueError(f"Unsupported board_input_mode: {board_input_mode}")
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        PhysicalManualTrainBoardDataset(
            annotation_root=annotation_root,
            image_size=image_size,
        )
    )
    rows = getattr(dataset, "rows")
    if len(rows) <= max_boards:
        return dataset
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(rows), generator=rng).tolist()[:max_boards]
    sampled_rows = [rows[index] for index in indices]
    return PhysicalManualTrainBoardDataset(
        annotation_root=annotation_root,
        image_size=image_size,
        rows=sampled_rows,
    )


def class_histogram(labels: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(labels.to(torch.long), minlength=len(SQUARE_CLASS_NAMES))
    return {
        class_name: int(counts[class_index].item())
        for class_index, class_name in enumerate(SQUARE_CLASS_NAMES)
    }


def save_board_samples(
    dataset: Dataset[Any],
    output_dir: Path,
    *,
    count: int,
    prefix: str,
) -> None:
    sample_dir = output_dir / "samples" / prefix
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    row_metadata = getattr(dataset, "rows", None)

    for index in range(min(count, len(dataset))):
        sample = dataset[index]
        if len(sample) == 2:
            image_tensor, labels = sample
        elif len(sample) == 3:
            image_tensor, labels, _corners = sample
        else:
            raise ValueError(f"Unsupported sample shape: {len(sample)} items")
        image_path = sample_dir / f"{index:03d}.png"
        tensor_to_rgb_image(image_tensor).save(image_path)
        payload: dict[str, object] = {
            "index": index,
            "image": relative_to_project(image_path),
            "label_histogram": class_histogram(labels.reshape(-1)),
        }
        if isinstance(row_metadata, list) and index < len(row_metadata):
            row = row_metadata[index]
            if hasattr(row, "annotation_id"):
                payload["annotation_id"] = getattr(row, "annotation_id")
            if hasattr(row, "source_video_id"):
                payload["source_video_id"] = getattr(row, "source_video_id")
            if hasattr(row, "board_path"):
                payload["board_path"] = getattr(row, "board_path")
            if hasattr(row, "clip_path"):
                payload["clip_path"] = getattr(row, "clip_path")
            if hasattr(row, "frame_index"):
                payload["frame_index"] = getattr(row, "frame_index")
            if hasattr(row, "source_channel_handle"):
                payload["source_channel_handle"] = getattr(row, "source_channel_handle")
        manifest.append(payload)

    write_json(sample_dir / "manifest.json", manifest)


def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().to(torch.float32)
    if tensor.ndim == 4:
        tiles = [tensor_to_rgb_image(square_tensor) for square_tensor in tensor]
        tile_width = max(tile.width for tile in tiles)
        tile_height = max(tile.height for tile in tiles)
        sheet = Image.new("RGB", (tile_width * 8, tile_height * 8), color=(245, 245, 245))
        for index, tile in enumerate(tiles[:64]):
            row = index // 8
            col = index % 8
            resized_tile = tile.resize((tile_width, tile_height))
            sheet.paste(resized_tile, (col * tile_width, row * tile_height))
        return sheet

    rgb = (tensor * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0.0, 1.0)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def promote_to_runtime_weights(
    *,
    checkpoint_path: Path,
    encoder_kwargs: dict[str, object],
    args: argparse.Namespace,
    probe_config: dict[str, object],
    real_eval_metrics: ProbeMetrics,
    best_selection_score: float,
    selection_dataset_name: str,
    selection_metric: str,
    held_out_eval_size: int,
    real_train_positions: int,
    real_selection_positions: int,
    manual_train_positions: int,
) -> None:
    weights_dir = _DEFAULT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)
    revision, version = next_version(weights_dir)
    versioned_path = weights_dir / f"{version}.pt"
    best_path = weights_dir / "best.pt"
    versioned_path.write_bytes(checkpoint_path.read_bytes())
    best_path.write_bytes(checkpoint_path.read_bytes())
    metadata = {
        "code_version": _MODEL_CODE_VERSION,
        "revision": revision,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": encoder_kwargs["model_name"],
        "encoder_type": args.encoder_type,
        "feature_layer_indices": encoder_kwargs.get("feature_layer_indices"),
        "output_grid_size": encoder_kwargs.get("output_grid_size"),
        "board_input_mode": args.board_input_mode,
        "input_size": args.input_size,
        "selection_dataset": selection_dataset_name,
        "selection_metric": selection_metric,
        "best_selection_score": round(best_selection_score, 4),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "probe_config": board_probe_config_from_checkpoint({"probe_config": probe_config}),
        "sources": {
            "synthetic_train_positions": args.synthetic_train_positions,
            "synthetic_val_positions": args.synthetic_val_positions,
            "real_train_positions": real_train_positions,
            "real_selection_positions": real_selection_positions,
            "manual_train_positions": manual_train_positions,
            "held_out_eval_size": held_out_eval_size,
        },
        "recommended_temporal_smoothing": {
            "mode": "adaptive_ema",
            "low_alpha": 0.02,
            "high_alpha": 0.12,
            "change_threshold": 8,
        },
        "runtime_format": "pytorch",
        "architecture": "board_probe",
        "runtime_constraints": "back_rank_pawns_and_exactly_one_king_per_color",
        "recommended_tracker": production_decoder_config(),
    }
    write_json(weights_dir / "metadata.json", metadata)
    logger.info("Promoted runtime weights to %s", best_path)


def next_version(weights_dir: Path) -> tuple[int, str]:
    metadata_path = weights_dir / "metadata.json"
    revision = 1
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("code_version") == _MODEL_CODE_VERSION:
            revision = int(metadata.get("revision", 0)) + 1
    return revision, f"{_MODEL_CODE_VERSION}r{revision}"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(_PROJECT_ROOT.resolve()))


if __name__ == "__main__":
    main()
