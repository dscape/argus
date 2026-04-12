#!/usr/bin/env python3
"""Train a frozen-feature board-context probe for physical per-square state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.board_data import (
    INPUT_SIZE as DEFAULT_INPUT_SIZE,
)
from pipeline.physical.board_data import (
    PhysicalEvalBoardDataset,
    PhysicalSyntheticBoardDataset,
    PhysicalSyntheticRenderedBoardDataset,
)
from pipeline.physical.board_probe import (
    evaluate_board_probe,
    extract_square_token_features,
    save_board_probe_checkpoint,
    train_board_probe,
)
from pipeline.shared import SQUARE_CLASS_NAMES

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_board_probe"
_DEFAULT_DINO_MODEL = "facebook/dinov2-base"
_DEFAULT_YOLO_MODEL = "weights/yolo_base/yolo11n.pt"
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

    train_dataset = build_synthetic_dataset(
        synthetic_source=args.synthetic_source,
        num_positions=args.synthetic_train_positions,
        image_size=args.input_size,
        seed=args.seed,
        augment=args.augment,
        min_moves=args.synthetic_min_moves,
        max_moves=args.synthetic_max_moves,
        min_ply=args.synthetic_min_ply,
    )
    val_dataset = build_synthetic_dataset(
        synthetic_source=args.synthetic_source,
        num_positions=args.synthetic_val_positions,
        image_size=args.input_size,
        seed=args.seed + 1,
        augment=False,
        min_moves=args.synthetic_min_moves,
        max_moves=args.synthetic_max_moves,
        min_ply=args.synthetic_min_ply,
    )
    eval_dataset = PhysicalEvalBoardDataset(image_size=args.input_size)
    eval_annotation_ids = [row.annotation_id for row in eval_dataset.rows]

    if args.save_samples > 0:
        save_board_samples(
            train_dataset,
            output_dir,
            count=args.save_samples,
            prefix="synthetic_train",
        )
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
    logger.info("Extracting synthetic val square-token features")
    val_tokens, val_labels = extract_square_token_features(
        val_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting held-out real board features")
    eval_tokens, eval_labels = extract_square_token_features(
        eval_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    class_weights = build_class_weights(train_labels, mode=args.class_weighting)
    probe, best_synth_val_accuracy = train_board_probe(
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
    )

    train_metrics = evaluate_board_probe(probe, train_tokens, train_labels, device=device)
    synth_val_metrics = evaluate_board_probe(probe, val_tokens, val_labels, device=device)
    real_eval_metrics = evaluate_board_probe(
        probe,
        eval_tokens,
        eval_labels,
        device=device,
        annotation_ids=eval_annotation_ids,
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
            "synthetic_source": args.synthetic_source,
            "synthetic_train_positions": args.synthetic_train_positions,
            "synthetic_val_positions": args.synthetic_val_positions,
            "synthetic_min_moves": args.synthetic_min_moves,
            "synthetic_max_moves": args.synthetic_max_moves,
            "synthetic_min_ply": args.synthetic_min_ply,
            "augment": args.augment,
            "class_weighting": args.class_weighting,
            "best_synth_val_accuracy": best_synth_val_accuracy,
        },
    )

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "encoder_type": args.encoder_type,
        "model_name": str(encoder_kwargs["model_name"]),
        "feature_layer_indices": encoder_kwargs.get("feature_layer_indices"),
        "output_grid_size": encoder_kwargs.get("output_grid_size"),
        "device": str(device),
        "input_size": args.input_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "synthetic_source": args.synthetic_source,
        "augment": args.augment,
        "class_weighting": args.class_weighting,
        "synthetic_train_positions": args.synthetic_train_positions,
        "synthetic_val_positions": args.synthetic_val_positions,
        "synthetic_min_moves": args.synthetic_min_moves,
        "synthetic_max_moves": args.synthetic_max_moves,
        "synthetic_min_ply": args.synthetic_min_ply,
        "real_eval_positions": len(eval_dataset),
        "train_label_histogram": class_histogram(train_labels.reshape(-1)),
        "synthetic_val_label_histogram": class_histogram(val_labels.reshape(-1)),
        "real_eval_label_histogram": class_histogram(eval_labels.reshape(-1)),
        "best_synthetic_val_accuracy": best_synth_val_accuracy,
        "train_metrics": train_metrics.to_dict(),
        "synthetic_val_metrics": synth_val_metrics.to_dict(),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "checkpoint": relative_to_project(checkpoint_path),
    }
    write_json(output_dir / "metrics.json", report)

    summary_lines = [
        "# Physical board-context probe",
        "",
        f"- encoder: `{args.encoder_type}`",
        f"- model: `{str(encoder_kwargs['model_name'])}`",
        f"- synthetic source: `{args.synthetic_source}`",
        f"- input size: `{args.input_size}`",
        f"- train augmentation: `{args.augment}`",
        f"- class weighting: `{args.class_weighting}`",
        f"- synthetic train positions: `{args.synthetic_train_positions}`",
        f"- synthetic val positions: `{args.synthetic_val_positions}`",
        f"- real eval positions: `{len(eval_dataset)}`",
        f"- train square accuracy: `{train_metrics.accuracy:.4f}`",
        f"- synthetic val square accuracy: `{synth_val_metrics.accuracy:.4f}`",
        f"- real eval square accuracy: `{real_eval_metrics.accuracy:.4f}`",
        f"- real eval non-empty accuracy: `{real_eval_metrics.non_empty_accuracy:.4f}`",
        f"- real eval macro F1: `{real_eval_metrics.macro_f1:.4f}`",
        f"- real eval board exact match: `{(real_eval_metrics.board_exact_match or 0.0):.4f}`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    logger.info("Train square accuracy: %.4f", train_metrics.accuracy)
    logger.info("Synthetic val square accuracy: %.4f", synth_val_metrics.accuracy)
    logger.info("Real eval square accuracy: %.4f", real_eval_metrics.accuracy)
    logger.info("Real eval non-empty accuracy: %.4f", real_eval_metrics.non_empty_accuracy)
    logger.info("Real eval macro F1: %.4f", real_eval_metrics.macro_f1)
    logger.info("Real eval board exact match: %.4f", real_eval_metrics.board_exact_match or 0.0)
    logger.info("Saved report to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train physical board-context square probe")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["dinov2", "yolo"],
        default="dinov2",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--yolo-feature-layer-indices", type=str, default="16,19,22")
    parser.add_argument("--yolo-output-grid-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--synthetic-source",
        type=str,
        choices=["topdown", "rendered"],
        default="topdown",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-train-positions", type=int, default=1200)
    parser.add_argument("--synthetic-val-positions", type=int, default=300)
    parser.add_argument("--synthetic-min-moves", type=int, default=12)
    parser.add_argument("--synthetic-max-moves", type=int, default=80)
    parser.add_argument("--synthetic-min-ply", type=int, default=8)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--class-weighting",
        type=str,
        choices=["none", "max_ratio"],
        default="none",
    )
    parser.add_argument("--save-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def build_encoder_kwargs(args: argparse.Namespace) -> dict[str, object]:
    model_name = args.model_name or default_model_name(args.encoder_type)
    encoder_kwargs: dict[str, object] = {
        "model_name": model_name,
        "frozen": True,
        "encoder_type": args.encoder_type,
    }
    if args.encoder_type == "yolo":
        encoder_kwargs["feature_layer_indices"] = parse_feature_layer_indices(
            args.yolo_feature_layer_indices
        )
        encoder_kwargs["output_grid_size"] = args.yolo_output_grid_size
    return encoder_kwargs


def default_model_name(encoder_type: str) -> str:
    if encoder_type == "yolo":
        return _DEFAULT_YOLO_MODEL
    return _DEFAULT_DINO_MODEL


def parse_feature_layer_indices(raw_value: str) -> list[int]:
    indices = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not indices:
        raise ValueError("yolo feature layer indices must not be empty")
    return [int(index) for index in indices]


def build_synthetic_dataset(
    *,
    synthetic_source: str,
    num_positions: int,
    image_size: int,
    seed: int,
    augment: bool,
    min_moves: int,
    max_moves: int,
    min_ply: int,
) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    if synthetic_source == "topdown":
        return PhysicalSyntheticBoardDataset(
            num_positions=num_positions,
            image_size=image_size,
            seed=seed,
            augment=augment,
            min_moves=min_moves,
            max_moves=max_moves,
            min_ply=min_ply,
        )
    if synthetic_source == "rendered":
        return PhysicalSyntheticRenderedBoardDataset(
            num_positions=num_positions,
            image_size=image_size,
            seed=seed,
            augment=augment,
            min_moves=min_moves,
            max_moves=max_moves,
        )
    raise ValueError(f"Unsupported synthetic source: {synthetic_source}")


def build_class_weights(labels: torch.Tensor, *, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "max_ratio":
        raise ValueError(f"Unsupported class weighting mode: {mode}")
    class_counts = torch.bincount(labels.reshape(-1), minlength=len(SQUARE_CLASS_NAMES)).float()
    return class_counts.max() / class_counts.clamp_min(1.0)


def class_histogram(labels: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(labels.to(torch.long), minlength=len(SQUARE_CLASS_NAMES))
    return {
        class_name: int(counts[class_index].item())
        for class_index, class_name in enumerate(SQUARE_CLASS_NAMES)
    }


def save_board_samples(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
    *,
    count: int,
    prefix: str,
) -> None:
    sample_dir = output_dir / "samples" / prefix
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    row_metadata = dataset.rows if isinstance(dataset, PhysicalEvalBoardDataset) else None

    for index in range(min(count, len(dataset))):
        image_tensor, labels = dataset[index]
        image_path = sample_dir / f"{index:03d}.png"
        tensor_to_rgb_image(image_tensor).save(image_path)
        payload: dict[str, object] = {
            "index": index,
            "image": relative_to_project(image_path),
            "label_histogram": class_histogram(labels.reshape(-1)),
        }
        if row_metadata is not None:
            row = row_metadata[index]
            payload["annotation_id"] = row.annotation_id
            payload["source_video_id"] = row.source_video_id
            payload["board_path"] = row.board_path
        manifest.append(payload)

    write_json(sample_dir / "manifest.json", manifest)


def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().to(torch.float32)
    rgb = (tensor * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0.0, 1.0)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(_PROJECT_ROOT.resolve()))


if __name__ == "__main__":
    main()
