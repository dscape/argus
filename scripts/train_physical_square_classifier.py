#!/usr/bin/env python3
"""Train a frozen-feature linear probe for physical square classification."""

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

from pipeline.physical.square_data import (
    CLASS_NAMES,
    INPUT_SIZE,
    PhysicalEvalSquareDataset,
    PhysicalSyntheticSquareDataset,
    load_eval_rows,
)
from pipeline.physical.square_probe import (
    ProbeMetrics,
    evaluate_probe,
    extract_features,
    save_probe_checkpoint,
    train_linear_probe,
)

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_square_probe"
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "physical"
_DEFAULT_DINO_MODEL = "facebook/dinov2-base"
_DEFAULT_YOLO_MODEL = "weights/yolo_base/yolo11n.pt"
_MODEL_CODE_VERSION = "v1"
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

    logger.info("Building synthetic physical datasets at %dx%d", INPUT_SIZE, INPUT_SIZE)
    train_dataset = PhysicalSyntheticSquareDataset(
        num_samples_per_class=args.synthetic_train_samples,
        image_size=INPUT_SIZE,
        seed=args.seed,
        augment=args.augment,
    )
    val_dataset = PhysicalSyntheticSquareDataset(
        num_samples_per_class=args.synthetic_val_samples,
        image_size=INPUT_SIZE,
        seed=args.seed + 1,
        augment=False,
    )

    eval_rows = load_eval_rows()
    eval_dataset = PhysicalEvalSquareDataset(
        rows=eval_rows,
        image_size=INPUT_SIZE,
        max_per_class=(args.eval_max_per_class or None),
        seed=args.seed,
    )
    eval_annotation_ids = [row.annotation_id for row in eval_dataset.rows]

    if args.save_samples > 0:
        save_square_samples(
            train_dataset,
            output_dir,
            count=args.save_samples,
            prefix="synthetic_train",
        )
        save_square_samples(
            val_dataset,
            output_dir,
            count=args.save_samples,
            prefix="synthetic_val",
        )
        save_square_samples(
            eval_dataset,
            output_dir,
            count=args.save_samples,
            prefix="real_eval",
        )

    logger.info("Extracting train features")
    train_features, train_labels = extract_features(
        train_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting synthetic val features")
    val_features, val_labels = extract_features(
        val_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Extracting held-out real eval features")
    eval_features, eval_labels = extract_features(
        eval_dataset,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    logger.info("Training linear probe")
    probe, best_synth_val_accuracy = train_linear_probe(
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    train_metrics = evaluate_probe(
        probe,
        train_features,
        train_labels,
        device=device,
    )
    synth_val_metrics = evaluate_probe(
        probe,
        val_features,
        val_labels,
        device=device,
    )
    real_eval_metrics = evaluate_probe(
        probe,
        eval_features,
        eval_labels,
        device=device,
        board_annotation_ids=eval_annotation_ids,
    )

    checkpoint_path = output_dir / "linear_probe.pt"
    save_probe_checkpoint(
        checkpoint_path,
        probe=probe.cpu(),
        model_name=str(encoder_kwargs["model_name"]),
        input_size=INPUT_SIZE,
        metadata={
            "encoder_type": args.encoder_type,
            "feature_layer_indices": encoder_kwargs.get("feature_layer_indices"),
            "output_grid_size": encoder_kwargs.get("output_grid_size"),
            "synthetic_train_samples": args.synthetic_train_samples,
            "synthetic_val_samples": args.synthetic_val_samples,
            "augment": args.augment,
            "eval_max_per_class": args.eval_max_per_class,
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
        "input_size": INPUT_SIZE,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "augment": args.augment,
        "synthetic_train_size": len(train_dataset),
        "synthetic_val_size": len(val_dataset),
        "real_eval_size": len(eval_dataset),
        "train_label_histogram": class_histogram(train_labels),
        "synthetic_val_label_histogram": class_histogram(val_labels),
        "real_eval_label_histogram": class_histogram(eval_labels),
        "best_synthetic_val_accuracy": best_synth_val_accuracy,
        "train_metrics": train_metrics.to_dict(),
        "synthetic_val_metrics": synth_val_metrics.to_dict(),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "checkpoint": relative_to_project(checkpoint_path),
    }
    write_json(output_dir / "metrics.json", report)

    summary_lines = [
        "# Physical square linear probe",
        "",
        f"- encoder: `{args.encoder_type}`",
        f"- model: `{str(encoder_kwargs['model_name'])}`",
        f"- train augmentation: `{args.augment}`",
        f"- synthetic train size: `{len(train_dataset)}`",
        f"- synthetic val size: `{len(val_dataset)}`",
        f"- real eval size: `{len(eval_dataset)}`",
        f"- train square accuracy: `{train_metrics.accuracy:.4f}`",
        f"- synthetic val square accuracy: `{synth_val_metrics.accuracy:.4f}`",
        f"- real eval square accuracy: `{real_eval_metrics.accuracy:.4f}`",
        f"- real eval non-empty accuracy: `{real_eval_metrics.non_empty_accuracy:.4f}`",
        f"- real eval macro F1: `{real_eval_metrics.macro_f1:.4f}`",
        f"- real eval board exact match: `{(real_eval_metrics.board_exact_match or 0.0):.4f}`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    if args.promote_to_weights:
        promote_to_runtime_weights(
            checkpoint_path=checkpoint_path,
            encoder_kwargs=encoder_kwargs,
            args=args,
            real_eval_metrics=real_eval_metrics,
            best_synth_val_accuracy=best_synth_val_accuracy,
            held_out_eval_size=len(eval_dataset),
        )

    logger.info("Train square accuracy: %.4f", train_metrics.accuracy)
    logger.info("Synthetic val accuracy: %.4f", synth_val_metrics.accuracy)
    logger.info("Real eval square accuracy: %.4f", real_eval_metrics.accuracy)
    logger.info("Real eval non-empty accuracy: %.4f", real_eval_metrics.non_empty_accuracy)
    logger.info("Real eval macro F1: %.4f", real_eval_metrics.macro_f1)
    logger.info("Real eval board exact match: %.4f", real_eval_metrics.board_exact_match or 0.0)
    logger.info("Saved report to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train physical square linear probe")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["dinov2", "yolo"],
        default="dinov2",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--yolo-feature-layer-indices", type=str, default="16,19,22")
    parser.add_argument("--yolo-output-grid-size", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-train-samples", type=int, default=600)
    parser.add_argument("--synthetic-val-samples", type=int, default=150)
    parser.add_argument("--eval-max-per-class", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--save-samples", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for metrics/checkpoints (default: outputs/physical_square_probe/<timestamp>)"
        ),
    )
    parser.add_argument(
        "--promote-to-weights",
        action="store_true",
        help="Copy the trained linear head to weights/physical/ for runtime use",
    )
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


def class_histogram(labels: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(labels.to(torch.long), minlength=len(CLASS_NAMES))
    return {
        class_name: int(counts[class_index].item())
        for class_index, class_name in enumerate(CLASS_NAMES)
    }


def save_square_samples(
    dataset: Dataset[tuple[torch.Tensor, int]],
    output_dir: Path,
    *,
    count: int,
    prefix: str,
) -> None:
    sample_dir = output_dir / "samples" / prefix
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    row_metadata = dataset.rows if isinstance(dataset, PhysicalEvalSquareDataset) else None

    for index in range(min(count, len(dataset))):
        image_tensor, label = dataset[index]
        image_path = sample_dir / f"{index:03d}.png"
        tensor_to_rgb_image(image_tensor).save(image_path)
        payload: dict[str, object] = {
            "index": index,
            "image": relative_to_project(image_path),
            "label_index": int(label),
            "label_name": CLASS_NAMES[int(label)],
        }
        if row_metadata is not None:
            row = row_metadata[index]
            payload["annotation_id"] = row.annotation_id
            payload["source_video_id"] = row.source_video_id
            payload["crop_path"] = row.crop_path
            payload["square_index"] = row.square_index
        manifest.append(payload)

    write_json(sample_dir / "manifest.json", manifest)


def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().to(torch.float32)
    rgb = (tensor * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0.0, 1.0)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def promote_to_runtime_weights(
    *,
    checkpoint_path: Path,
    encoder_kwargs: dict[str, object],
    args: argparse.Namespace,
    real_eval_metrics: ProbeMetrics,
    best_synth_val_accuracy: float,
    held_out_eval_size: int,
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
        "input_size": INPUT_SIZE,
        "best_synthetic_val_accuracy": round(best_synth_val_accuracy, 4),
        "real_eval_metrics": real_eval_metrics.to_dict(),
        "sources": {
            "synthetic_train_samples": args.synthetic_train_samples,
            "synthetic_val_samples": args.synthetic_val_samples,
            "held_out_eval_size": held_out_eval_size,
        },
        "runtime_format": "pytorch",
        "architecture": f"{args.encoder_type}_linear_probe",
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
