#!/usr/bin/env python3
"""Train a real-data physical square classifier and evaluate board reconstruction."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.replay import build_replay_board
from pipeline.overlay.square_classifier_model import (
    INPUT_SIZE,
    MODEL_CODE_VERSION,
    TinySquareClassifier,
)
from pipeline.physical.board_data import PhysicalEvalBoardDataset
from pipeline.physical.square_data import (
    CLASS_NAMES,
    augment_physical_square_image,
    split_rectified_board_into_squares,
)
from pipeline.physical.square_probe import evaluate_probe
from pipeline.shared import (
    LookaheadLegalMoveStateTracker,
    SegmentalLegalSequenceDecoder,
    board_to_class_ids,
    constrained_board_class_ids,
)
from scripts.eval_physical_board_tracker import (
    FramePrediction,
    IdentityProbe,
    compute_tracker_sequence_metrics,
)

from argus.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAIN_ROOT = _PROJECT_ROOT / "data" / "physical" / "train"
_DEFAULT_VAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_square_tiny_classifier"
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class SquareRow:
    annotation_id: str
    clip_path: str
    crop_path: str
    frame_index: int
    label_index: int
    label_name: str
    source_video_id: str | None
    square_index: int
    square_name: str


class PhysicalSquareCropDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        rows: list[SquareRow],
        *,
        augment: bool,
        seed: int,
        imagenet_normalize: bool,
        context_margin: float,
        board_path_by_annotation: dict[str, str],
    ) -> None:
        self.rows = rows
        self.augment = augment
        self.seed = int(seed)
        self.imagenet_normalize = imagenet_normalize
        self.context_margin = float(context_margin)
        self.board_path_by_annotation = board_path_by_annotation
        self._board_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = self._load_square_image(row)
        if self.augment:
            rng = random.Random(self.seed + index * 1_000_003)
            image = augment_physical_square_image(image, rng)
        return (
            preprocess_square_crop(image, imagenet_normalize=self.imagenet_normalize),
            row.label_index,
        )

    def _load_square_image(self, row: SquareRow) -> np.ndarray:
        if self.context_margin <= 0.0:
            image = cv2.imread(str(_PROJECT_ROOT / row.crop_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load square crop: {row.crop_path}")
            return image
        board_path = self.board_path_by_annotation.get(row.annotation_id)
        if board_path is None:
            raise ValueError(f"Missing board image for annotation: {row.annotation_id}")
        board_image = self._board_cache.get(board_path)
        if board_image is None:
            board_image = cv2.imread(str(_PROJECT_ROOT / board_path), cv2.IMREAD_COLOR)
            if board_image is None:
                raise ValueError(f"Failed to load board image: {board_path}")
            self._board_cache[board_path] = board_image
        return extract_square_crop_with_margin(
            board_image,
            square_index=row.square_index,
            margin_fraction=self.context_margin,
        )


class BoardSquareDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        square_crops: list[np.ndarray],
        *,
        imagenet_normalize: bool,
    ) -> None:
        self.square_crops = square_crops
        self.imagenet_normalize = imagenet_normalize

    def __len__(self) -> int:
        return len(self.square_crops)

    def __getitem__(self, index: int) -> torch.Tensor:
        return preprocess_square_crop(
            self.square_crops[index],
            imagenet_normalize=self.imagenet_normalize,
        )


def extract_square_crop_with_margin(
    board_bgr: np.ndarray,
    *,
    square_index: int,
    margin_fraction: float,
) -> np.ndarray:
    height, width = board_bgr.shape[:2]
    if height != width:
        raise ValueError(f"Board must be square, got {width}x{height}")
    square_size = height // 8
    row = square_index // 8
    col = square_index % 8
    margin = int(round(square_size * margin_fraction))
    y1 = max(0, row * square_size - margin)
    x1 = max(0, col * square_size - margin)
    y2 = min(height, (row + 1) * square_size + margin)
    x2 = min(width, (col + 1) * square_size + margin)
    return board_bgr[y1:y2, x1:x2].copy()


def preprocess_square_crop(
    image_bgr: np.ndarray,
    *,
    imagenet_normalize: bool,
) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != INPUT_SIZE or rgb.shape[1] != INPUT_SIZE:
        interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= INPUT_SIZE else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=interpolation)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    if imagenet_normalize:
        return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor


def load_square_rows(root: Path) -> list[SquareRow]:
    manifest_path = root / "square_manifest.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"Missing square manifest: {manifest_path}")
    rows: list[SquareRow] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            SquareRow(
                annotation_id=str(payload["annotation_id"]),
                clip_path=str(payload["clip_path"]),
                crop_path=str(payload["crop_path"]),
                frame_index=int(payload["frame_index"]),
                label_index=int(payload["label_index"]),
                label_name=str(payload["label_name"]),
                source_video_id=(
                    str(payload["source_video_id"])
                    if payload.get("source_video_id") is not None
                    else None
                ),
                square_index=int(payload["square_index"]),
                square_name=str(payload["square_name"]),
            )
        )
    return rows


def load_board_path_by_annotation(root: Path) -> dict[str, str]:
    manifest_path = root / "board_annotations.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"Missing board annotation manifest: {manifest_path}")
    mapping: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        mapping[str(payload["annotation_id"])] = str(payload["rectified_board_path"])
    return mapping


def build_model(model_name: str) -> tuple[torch.nn.Module, bool]:
    if model_name == "tiny":
        return TinySquareClassifier(), False
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model, True
    raise ValueError(f"Unsupported model: {model_name}")


def build_train_loader(
    rows: list[SquareRow],
    *,
    batch_size: int,
    augment: bool,
    seed: int,
    imagenet_normalize: bool,
    context_margin: float,
    board_path_by_annotation: dict[str, str],
) -> DataLoader[tuple[torch.Tensor, int]]:
    dataset = PhysicalSquareCropDataset(
        rows,
        augment=augment,
        seed=seed,
        imagenet_normalize=imagenet_normalize,
        context_margin=context_margin,
        board_path_by_annotation=board_path_by_annotation,
    )
    label_counts = Counter(row.label_index for row in rows)
    sample_weights = torch.tensor(
        [1.0 / max(label_counts[row.label_index], 1) for row in rows],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(rows), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )


def build_eval_loader(
    rows: list[SquareRow],
    *,
    batch_size: int,
    imagenet_normalize: bool,
    context_margin: float,
    board_path_by_annotation: dict[str, str],
) -> DataLoader[tuple[torch.Tensor, int]]:
    return DataLoader(
        PhysicalSquareCropDataset(
            rows,
            augment=False,
            seed=0,
            imagenet_normalize=imagenet_normalize,
            context_margin=context_margin,
            board_path_by_annotation=board_path_by_annotation,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def evaluate_square_loader(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    per_class_correct = [0] * len(CLASS_NAMES)
    per_class_total = [0] * len(CLASS_NAMES)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()
            for pred, target in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_class_total[target] += 1
                if pred == target:
                    per_class_correct[target] += 1
    class_accuracy = {
        CLASS_NAMES[index]: (per_class_correct[index] / per_class_total[index])
        if per_class_total[index] > 0
        else 0.0
        for index in range(len(CLASS_NAMES))
    }
    return {
        "accuracy": correct / max(total, 1),
        "class_accuracy": class_accuracy,
        "total": total,
    }


def train_model(
    *,
    model_name: str,
    train_rows: list[SquareRow],
    val_rows: list[SquareRow],
    train_board_path_by_annotation: dict[str, str],
    val_board_path_by_annotation: dict[str, str],
    context_margin: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    augment: bool,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any], bool]:
    model, imagenet_normalize = build_model(model_name)
    train_loader = build_train_loader(
        train_rows,
        batch_size=batch_size,
        augment=augment,
        seed=seed,
        imagenet_normalize=imagenet_normalize,
        context_margin=context_margin,
        board_path_by_annotation=train_board_path_by_annotation,
    )
    val_loader = build_eval_loader(
        val_rows,
        batch_size=batch_size,
        imagenet_normalize=imagenet_normalize,
        context_margin=context_margin,
        board_path_by_annotation=val_board_path_by_annotation,
    )

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * images.shape[0]
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        scheduler.step()

        train_accuracy = total_correct / max(total, 1)
        val_metrics = evaluate_square_loader(model, val_loader, device=device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total, 1),
                "train_accuracy": train_accuracy,
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )
        logger.info(
            "epoch %d/%d train_loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch,
            epochs,
            total_loss / max(total, 1),
            train_accuracy,
            float(val_metrics["accuracy"]),
        )
        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    best_model, _ = build_model(model_name)
    best_model.load_state_dict(best_state)
    return (
        best_model,
        {
            "best_val_accuracy": best_val_accuracy,
            "history": history,
        },
        imagenet_normalize,
    )


def predict_board_logits_batch(
    model: torch.nn.Module,
    board_images: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    imagenet_normalize: bool,
    context_margin: float,
) -> list[torch.Tensor]:
    if not board_images:
        return []
    square_crops = [
        crop
        for image in board_images
        for crop in (
            split_rectified_board_into_squares(image)
            if context_margin <= 0.0
            else [
                extract_square_crop_with_margin(
                    image,
                    square_index=square_index,
                    margin_fraction=context_margin,
                )
                for square_index in range(64)
            ]
        )
    ]
    loader = DataLoader(
        BoardSquareDataset(square_crops, imagenet_normalize=imagenet_normalize),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    logits_batches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits_batches.append(model(batch.to(device)).cpu())
    all_logits = torch.cat(logits_batches, dim=0)
    return [all_logits[index : index + 64] for index in range(0, all_logits.shape[0], 64)]


def load_rectified_board_images(rows: list[Any]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for row in rows:
        image = cv2.imread(str(_PROJECT_ROOT / row.board_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load board image: {row.board_path}")
        images.append(image)
    return images


def initial_board_state_for_clip(clip_path: str) -> tuple[str, str | None]:
    clip = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    initial_board_fen = clip.get("initial_board_fen") if isinstance(clip, dict) else None
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    raw_side = clip.get("initial_side_to_move") if isinstance(clip, dict) else None
    return initial_board_fen, raw_side if isinstance(raw_side, str) else None


def evaluate_board_variants(
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    imagenet_normalize: bool,
    context_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = PhysicalEvalBoardDataset()
    rows = sorted(
        dataset.rows,
        key=lambda row: (
            row.clip_path or row.annotation_id,
            -1 if row.frame_index is None else row.frame_index,
            row.annotation_id,
        ),
    )
    images = load_rectified_board_images(rows)
    board_logits = predict_board_logits_batch(
        model,
        images,
        device=device,
        batch_size=batch_size,
        imagenet_normalize=imagenet_normalize,
        context_margin=context_margin,
    )

    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    stateless_predictions: list[int] = []
    stateless_targets: list[int] = []
    stateless_annotation_ids: list[str] = []
    rows_by_clip: dict[str, list[tuple[Any, torch.Tensor]]] = defaultdict(list)

    for row, logits in zip(rows, board_logits):
        rows_by_clip[row.clip_path or row.annotation_id].append((row, logits))
        class_ids = constrained_board_class_ids(logits)
        stateless_predictions.extend(class_ids.tolist())
        stateless_targets.extend(int(value) for value in row.labels)
        stateless_annotation_ids.extend([row.annotation_id] * 64)

    stateless_logits = torch.zeros(
        (len(stateless_predictions), len(CLASS_NAMES)),
        dtype=torch.float32,
    )
    for index, class_id in enumerate(stateless_predictions):
        stateless_logits[index, class_id] = 1.0
    stateless_metrics = evaluate_probe(
        IdentityProbe(),
        stateless_logits,
        torch.tensor(stateless_targets, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=stateless_annotation_ids,
    ).to_dict()

    variants: list[dict[str, Any]] = [
        {
            "name": "stateless_runtime",
            "board_exact_match": float(stateless_metrics["board_exact_match"] or 0.0),
            "square_accuracy": float(stateless_metrics["accuracy"]),
            "non_empty_accuracy": float(stateless_metrics["non_empty_accuracy"]),
            "macro_f1": float(stateless_metrics["macro_f1"]),
            "move_detection_recall": 0.0,
            "false_change_rate": 0.0,
        }
    ]

    for variant_name, decoder_kind, decoder_kwargs in [
        ("lookahead_w3_m8", "lookahead", {"lookahead_window": 3, "move_score_margin": 8.0}),
        ("lookahead_w4_m10", "lookahead", {"lookahead_window": 4, "move_score_margin": 10.0}),
        (
            "segmental_boardonly",
            "segmental",
            {
                "beam_size": 8,
                "top_move_candidates": 16,
                "top_board_candidates": 0,
                "board_weight": 1.0,
                "move_weight": 0.0,
                "detect_weight": 0.0,
                "move_score_margin": 0.0,
                "detect_peak_threshold": 0.1,
                "board_change_peak_threshold": 2.0 / 64.0,
                "min_event_separation": 16,
                "secondary_min_event_separation": None,
                "secondary_peak_ratio": 0.8,
                "state_aware_proposal_passes": 0,
                "anomaly_change_evidence_threshold": 0.25,
                "anomaly_settled_gain_threshold": 0.0,
                "segment_board_drop_worst_frames": 0,
                "event_window_radius": 1,
                "max_event_proposals": 20,
                "diagnostic_settled_horizon": 8,
            },
        ),
    ]:
        predictions_by_clip.clear()
        square_predictions: list[int] = []
        square_targets: list[int] = []
        board_annotation_ids: list[str] = []
        for clip_path, clip_rows in rows_by_clip.items():
            clip_rows.sort(
                key=lambda item: (
                    -1 if item[0].frame_index is None else item[0].frame_index,
                    item[0].annotation_id,
                )
            )
            initial_board_fen, initial_side_to_move = initial_board_state_for_clip(clip_path)
            clip_sequence_logits = [logits for _row, logits in clip_rows]
            if decoder_kind == "lookahead":
                results = list(
                    LookaheadLegalMoveStateTracker(
                        initial_board_fen,
                        initial_side_to_move=initial_side_to_move,
                        lookahead_window=int(decoder_kwargs["lookahead_window"]),
                        move_score_margin=float(decoder_kwargs["move_score_margin"]),
                    ).decode(clip_sequence_logits)
                )
            else:
                results = list(
                    SegmentalLegalSequenceDecoder(
                        initial_board_fen,
                        initial_side_to_move=initial_side_to_move,
                        **decoder_kwargs,
                    )
                    .decode(clip_sequence_logits)
                    .frames
                )
            for (row, _logits), result in zip(clip_rows, results):
                predicted_labels = tuple(board_to_class_ids(build_replay_board(result.fen)))
                target_labels = tuple(int(value) for value in row.labels)
                predictions_by_clip[clip_path].append(
                    FramePrediction(
                        annotation_id=row.annotation_id,
                        frame_index=int(row.frame_index or 0),
                        target_labels=target_labels,
                        predicted_labels=predicted_labels,
                        move_uci=result.move_uci,
                    )
                )
                square_predictions.extend(predicted_labels)
                square_targets.extend(target_labels)
                board_annotation_ids.extend([row.annotation_id] * 64)
        logits = torch.zeros((len(square_predictions), len(CLASS_NAMES)), dtype=torch.float32)
        for index, class_id in enumerate(square_predictions):
            logits[index, class_id] = 1.0
        metrics = evaluate_probe(
            IdentityProbe(),
            logits,
            torch.tensor(square_targets, dtype=torch.long),
            device=torch.device("cpu"),
            board_annotation_ids=board_annotation_ids,
        ).to_dict()
        move_recall, false_change_rate, _diagnostics = compute_tracker_sequence_metrics(
            predictions_by_clip,
            tolerance=1,
        )
        variants.append(
            {
                "name": variant_name,
                "board_exact_match": float(metrics["board_exact_match"] or 0.0),
                "square_accuracy": float(metrics["accuracy"]),
                "non_empty_accuracy": float(metrics["non_empty_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "move_detection_recall": float(move_recall),
                "false_change_rate": float(false_change_rate),
            }
        )

    return variants, {
        "evaluated_boards": len(rows),
        "val_source_video_ids": sorted(
            {row.source_video_id for row in rows if row.source_video_id}
        ),
    }


def write_summary_md(
    path: Path,
    *,
    model_name: str,
    context_margin: float,
    train_rows: list[SquareRow],
    val_rows: list[SquareRow],
    train_square_metrics: dict[str, Any],
    val_square_metrics: dict[str, Any],
    board_variants: list[dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# Physical square classifier",
        "",
        f"- model: `{model_name}`",
        f"- context margin: `{context_margin}`",
        f"- train squares: `{len(train_rows)}`",
        f"- val squares: `{len(val_rows)}`",
        f"- checkpoint: `{checkpoint_path.relative_to(_PROJECT_ROOT)}`",
        f"- train square accuracy: `{train_square_metrics['accuracy']:.4f}`",
        f"- val square accuracy: `{val_square_metrics['accuracy']:.4f}`",
        "",
        "## Board / tracker variants",
        "",
    ]
    for variant in board_variants:
        lines.extend(
            [
                f"### {variant['name']}",
                f"- board exact: `{variant['board_exact_match']:.4f}`",
                f"- square accuracy: `{variant['square_accuracy']:.4f}`",
                f"- non-empty accuracy: `{variant['non_empty_accuracy']:.4f}`",
                f"- macro-F1: `{variant['macro_f1']:.4f}`",
                f"- move recall: `{variant['move_detection_recall']:.4f}`",
                f"- false-change: `{variant['false_change_rate']:.4f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a real-data physical square classifier")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-root", type=Path, default=_DEFAULT_TRAIN_ROOT)
    parser.add_argument("--val-root", type=Path, default=_DEFAULT_VAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", choices=("tiny", "resnet18"), default="tiny")
    parser.add_argument("--context-margin", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(resolve_device(args.device))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            _DEFAULT_OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_square_rows(args.train_root)
    val_rows = load_square_rows(args.val_root)
    train_board_path_by_annotation = load_board_path_by_annotation(args.train_root)
    val_board_path_by_annotation = load_board_path_by_annotation(args.val_root)
    logger.info("Loaded %d train squares and %d val squares", len(train_rows), len(val_rows))

    best_model, train_summary, imagenet_normalize = train_model(
        model_name=args.model,
        train_rows=train_rows,
        val_rows=val_rows,
        train_board_path_by_annotation=train_board_path_by_annotation,
        val_board_path_by_annotation=val_board_path_by_annotation,
        context_margin=args.context_margin,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        augment=args.augment,
        seed=args.seed,
    )
    best_model.to(device)

    train_square_metrics = evaluate_square_loader(
        best_model,
        build_eval_loader(
            train_rows,
            batch_size=args.batch_size,
            imagenet_normalize=imagenet_normalize,
            context_margin=args.context_margin,
            board_path_by_annotation=train_board_path_by_annotation,
        ),
        device=device,
    )
    val_square_metrics = evaluate_square_loader(
        best_model,
        build_eval_loader(
            val_rows,
            batch_size=args.batch_size,
            imagenet_normalize=imagenet_normalize,
            context_margin=args.context_margin,
            board_path_by_annotation=val_board_path_by_annotation,
        ),
        device=device,
    )
    board_variants, board_context = evaluate_board_variants(
        best_model,
        device=device,
        batch_size=args.batch_size,
        imagenet_normalize=imagenet_normalize,
        context_margin=args.context_margin,
    )

    checkpoint_path = output_dir / "square_classifier.pt"
    torch.save(
        {
            "architecture": f"{args.model}_square_cnn",
            "model_name": args.model,
            "input_size": INPUT_SIZE,
            "num_classes": len(CLASS_NAMES),
            "state_dict": best_model.cpu().state_dict(),
            "metadata": {
                "code_version": MODEL_CODE_VERSION,
                "class_names": list(CLASS_NAMES),
                "train_root": str(args.train_root.relative_to(_PROJECT_ROOT)),
                "val_root": str(args.val_root.relative_to(_PROJECT_ROOT)),
                "augment": args.augment,
                "context_margin": args.context_margin,
                "imagenet_normalize": imagenet_normalize,
                "seed": args.seed,
            },
        },
        checkpoint_path,
    )

    (output_dir / "summary.json").write_text(json.dumps(board_variants, indent=2, sort_keys=True))
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "device": str(device),
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "augment": args.augment,
        "context_margin": args.context_margin,
        "train_square_metrics": train_square_metrics,
        "val_square_metrics": val_square_metrics,
        "board_variants": board_variants,
        "board_context": board_context,
        "train_summary": train_summary,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    write_summary_md(
        output_dir / "summary.md",
        model_name=args.model,
        context_margin=args.context_margin,
        train_rows=train_rows,
        val_rows=val_rows,
        train_square_metrics=train_square_metrics,
        val_square_metrics=val_square_metrics,
        board_variants=board_variants,
        checkpoint_path=checkpoint_path,
    )
    logger.info("Wrote %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
