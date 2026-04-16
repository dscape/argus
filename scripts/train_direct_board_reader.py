#!/usr/bin/env python3
"""Train a full-image direct board-state reader and compare it to prior baselines."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.replay import build_replay_board
from pipeline.physical.direct_board_reader import DirectBoardReaderConfig, DirectPhysicalBoardReader
from pipeline.physical.direct_board_reader_data import DirectBoardManifestDataset
from pipeline.physical.square_probe import ProbeMetrics, evaluate_probe
from pipeline.shared import SQUARE_CLASS_NAMES, LookaheadLegalMoveStateTracker, board_to_class_ids

from argus.device import resolve_device
from argus.model.vision_encoder import VisionEncoder, default_model_name_for_encoder_type

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "direct_board_reader"
_PREVIOUS_PIPELINE_TRACKER_PATH = (
    _PROJECT_ROOT
    / "outputs"
    / "2026-04-16"
    / "tracker_sweep_rectified_realplusmanual"
    / "eval_w2_m10.json"
)
_PREVIOUS_RAW_PROBE_PATH = (
    _PROJECT_ROOT
    / "outputs"
    / "2026-04-16"
    / (
        "physical_board_probe_rectified_realplusmanual_psrholdout_stride1_rw4_mw8_"
        "posmlp_layers8_10_11"
    )
    / "metrics.json"
)


@dataclass(frozen=True)
class FramePrediction:
    annotation_id: str
    frame_index: int
    target_labels: tuple[int, ...]
    predicted_labels: tuple[int, ...]
    move_uci: str | None


class IdentityProbe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(resolve_device(args.device))
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_data_dir = args.prepared_data_dir.resolve()
    train_dataset = DirectBoardManifestDataset(
        manifest_path=prepared_data_dir / "train_manifest.jsonl",
        image_size=args.input_size,
    )
    physical_val_dataset = DirectBoardManifestDataset(
        manifest_path=prepared_data_dir / "physical_val_manifest.jsonl",
        image_size=args.input_size,
    )
    chessred_val_dataset = DirectBoardManifestDataset(
        manifest_path=prepared_data_dir / "chessred_val_manifest.jsonl",
        image_size=args.input_size,
    )

    torch.manual_seed(args.seed)
    requested_encoder_type = args.encoder_type
    resolved_encoder_type = requested_encoder_type
    resolved_model_name = args.model_name or default_model_name_for_encoder_type(
        requested_encoder_type
    )
    feature_layer_indices = parse_feature_layer_indices(
        raw_value=args.feature_layer_indices,
        encoder_type=requested_encoder_type,
    )
    encoder = VisionEncoder(
        encoder_type=requested_encoder_type,
        model_name=resolved_model_name,
        frozen=True,
        feature_layer_indices=feature_layer_indices,
    )
    model = DirectPhysicalBoardReader(
        vision_encoder=encoder,
        config=DirectBoardReaderConfig(
            input_size=args.input_size,
            num_classes=len(SQUARE_CLASS_NAMES),
            num_heads=args.num_heads,
            dropout=args.dropout,
            mlp_ratio=args.mlp_ratio,
            head_type=args.head_type,
            hidden_dim=args.hidden_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_ff_dim=args.transformer_ff_dim,
        ),
    ).to(device)

    class_weights = build_class_weights(train_dataset)
    best_model, best_selection_score = train_direct_board_reader(
        model,
        train_dataset=train_dataset,
        selection_dataset=physical_val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        class_weights=class_weights,
        selection_metric=args.selection_metric,
    )

    physical_logits, physical_labels = predict_dataset(
        best_model,
        dataset=physical_val_dataset,
        device=device,
        batch_size=args.batch_size,
    )
    chessred_logits, chessred_labels = predict_dataset(
        best_model,
        dataset=chessred_val_dataset,
        device=device,
        batch_size=args.batch_size,
    )
    train_logits, train_labels = predict_dataset(
        best_model,
        dataset=train_dataset,
        device=device,
        batch_size=args.batch_size,
    )

    train_metrics = evaluate_logits(
        train_logits,
        train_labels,
        annotation_ids=[row.example_id for row in train_dataset.rows],
    )
    physical_metrics = evaluate_logits(
        physical_logits,
        physical_labels,
        annotation_ids=[row.annotation_id or row.example_id for row in physical_val_dataset.rows],
    )
    chessred_metrics = evaluate_logits(
        chessred_logits,
        chessred_labels,
        annotation_ids=[row.example_id for row in chessred_val_dataset.rows],
    )

    physical_tracker_report = evaluate_tracker_from_logits(
        logits=physical_logits,
        rows=physical_val_dataset.rows,
        lookahead_window=args.tracker_lookahead_window,
        lookahead_margin=args.tracker_lookahead_margin,
        move_match_tolerance=args.tracker_move_match_tolerance,
    )

    previous_baselines = load_previous_baselines()
    comparison = build_comparison(
        physical_metrics=physical_metrics,
        physical_tracker_report=physical_tracker_report,
        previous_baselines=previous_baselines,
    )

    checkpoint = {
        "architecture": "direct_full_image_board_reader",
        "state_dict": best_model.state_dict(),
        "encoder_config": {
            "encoder_type": resolved_encoder_type,
            "requested_encoder_type": requested_encoder_type,
            "model_name": resolved_model_name,
            "feature_layer_indices": feature_layer_indices,
        },
        **best_model.checkpoint_config(),
        "selection_score": best_selection_score,
        "selection_metric": args.selection_metric,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / "direct_board_reader.pt"
    torch.save(checkpoint, checkpoint_path)

    prepared_summary_path = prepared_data_dir / "summary.json"
    prepared_summary = (
        json.loads(prepared_summary_path.read_text()) if prepared_summary_path.exists() else None
    )
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(_PROJECT_ROOT)),
        "prepared_data_dir": str(prepared_data_dir.relative_to(_PROJECT_ROOT)),
        "prepared_summary": prepared_summary,
        "encoder_type": resolved_encoder_type,
        "requested_encoder_type": requested_encoder_type,
        "model_name": resolved_model_name,
        "feature_layer_indices": feature_layer_indices,
        "input_size": args.input_size,
        "train_count": len(train_dataset),
        "train_counts_by_domain": dict(Counter(row.domain for row in train_dataset.rows)),
        "physical_val_count": len(physical_val_dataset),
        "chessred_val_count": len(chessred_val_dataset),
        "selection_score": best_selection_score,
        "selection_metric": args.selection_metric,
        "train_metrics": train_metrics.to_dict(),
        "physical_val_metrics": physical_metrics.to_dict(),
        "chessred_val_metrics": chessred_metrics.to_dict(),
        "physical_val_tracker_report": physical_tracker_report,
        "previous_baselines": previous_baselines,
        "comparison": comparison,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "summary.md").write_text(render_summary_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a direct full-image board reader")
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-type", choices=("dinov2", "siglip2"), default="dinov2")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--feature-layer-indices", type=str, default="")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--head-type",
        choices=("linear", "pos_mlp", "transformer"),
        default="pos_mlp",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "non_empty_accuracy", "macro_f1", "non_empty_plus_macro"),
        default="non_empty_plus_macro",
    )
    parser.add_argument("--tracker-lookahead-window", type=int, default=2)
    parser.add_argument("--tracker-lookahead-margin", type=float, default=10.0)
    parser.add_argument("--tracker-move-match-tolerance", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def parse_feature_layer_indices(*, raw_value: str, encoder_type: str) -> list[int] | None:
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if values:
        return [int(value) for value in values]
    if encoder_type == "dinov2":
        return [8, 10, 11]
    return None


def build_class_weights(dataset: DirectBoardManifestDataset) -> torch.Tensor:
    counts = torch.zeros(len(SQUARE_CLASS_NAMES), dtype=torch.long)
    for row in dataset.rows:
        labels = torch.tensor(row.labels, dtype=torch.long)
        counts += torch.bincount(labels, minlength=len(SQUARE_CLASS_NAMES))
    max_count = counts.max().clamp_min(1)
    weights = max_count.to(torch.float32) / counts.clamp_min(1).to(torch.float32)
    return weights


def train_direct_board_reader(
    model: DirectPhysicalBoardReader,
    *,
    train_dataset: DirectBoardManifestDataset,
    selection_dataset: DirectBoardManifestDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    class_weights: torch.Tensor,
    selection_metric: str,
) -> tuple[DirectPhysicalBoardReader, float]:
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="none")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    for _epoch in range(epochs):
        model.train()
        for images, labels, sample_weights in train_loader:
            logits = model(images.to(device))
            labels = labels.to(device)
            per_square_loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            per_board_loss = per_square_loss.reshape(labels.shape[0], labels.shape[1]).mean(dim=1)
            sample_weights = sample_weights.to(device)
            loss = (per_board_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        selection_logits, selection_labels = predict_dataset(
            model,
            dataset=selection_dataset,
            device=device,
            batch_size=batch_size,
        )
        selection_metrics = evaluate_logits(
            selection_logits,
            selection_labels,
            annotation_ids=[row.annotation_id or row.example_id for row in selection_dataset.rows],
        )
        score = selection_score_for_metrics(selection_metrics, selection_metric)
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")

    best_model = DirectPhysicalBoardReader(
        vision_encoder=model.vision_encoder,
        config=model.config,
    )
    best_model.load_state_dict(best_state)
    return best_model.to(device), best_score


def predict_dataset(
    model: DirectPhysicalBoardReader,
    *,
    dataset: DirectBoardManifestDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels, _sample_weights in loader:
            logits = model(images.to(device))
            logits_batches.append(logits.cpu())
            label_batches.append(labels.cpu())
    return torch.cat(logits_batches, dim=0), torch.cat(label_batches, dim=0)


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    annotation_ids: list[str],
) -> ProbeMetrics:
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    flattened_labels = labels.reshape(-1)
    board_annotation_ids = [annotation_id for annotation_id in annotation_ids for _ in range(64)]
    return evaluate_probe(
        IdentityProbe(),
        flattened_logits,
        flattened_labels,
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )


def selection_score_for_metrics(metrics: ProbeMetrics, selection_metric: str) -> float:
    if selection_metric == "accuracy":
        return metrics.accuracy
    if selection_metric == "non_empty_accuracy":
        return metrics.non_empty_accuracy
    if selection_metric == "macro_f1":
        return metrics.macro_f1
    if selection_metric == "non_empty_plus_macro":
        return (metrics.non_empty_accuracy + metrics.macro_f1) / 2.0
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def evaluate_tracker_from_logits(
    *,
    logits: torch.Tensor,
    rows: list,
    lookahead_window: int,
    lookahead_margin: float,
    move_match_tolerance: int,
) -> dict[str, object]:
    rows_by_clip: dict[str, list[tuple[object, torch.Tensor]]] = defaultdict(list)
    for row, row_logits in zip(rows, logits, strict=False):
        clip_key = row.clip_path or row.example_id
        rows_by_clip[clip_key].append((row, row_logits))

    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    square_predictions: list[int] = []
    square_targets: list[int] = []
    board_annotation_ids: list[str] = []

    for clip_key, clip_rows in rows_by_clip.items():
        if clip_key is None:
            continue
        clip_rows.sort(
            key=lambda item: (
                -1 if item[0].frame_index is None else int(item[0].frame_index),
                item[0].annotation_id or item[0].example_id,
            )
        )
        initial_board_fen, initial_side_to_move = _initial_board_state_for_row(clip_rows[0][0])
        sequence_results = LookaheadLegalMoveStateTracker(
            initial_board_fen,
            initial_side_to_move=initial_side_to_move,
            lookahead_window=lookahead_window,
            move_score_margin=lookahead_margin,
        ).decode([row_logits for _row, row_logits in clip_rows])
        for (row, _row_logits), result in zip(clip_rows, sequence_results, strict=False):
            predicted_labels = tuple(board_to_class_ids(build_replay_board(result.fen)))
            target_labels = tuple(int(value) for value in row.labels)
            annotation_id = row.annotation_id or row.example_id
            predictions_by_clip[clip_key].append(
                FramePrediction(
                    annotation_id=annotation_id,
                    frame_index=int(row.frame_index or 0),
                    target_labels=target_labels,
                    predicted_labels=predicted_labels,
                    move_uci=result.move_uci,
                )
            )
            square_predictions.extend(predicted_labels)
            square_targets.extend(target_labels)
            board_annotation_ids.extend([annotation_id] * 64)

    report_logits = torch.zeros(
        (len(square_predictions), len(SQUARE_CLASS_NAMES)),
        dtype=torch.float32,
    )
    for index, class_id in enumerate(square_predictions):
        report_logits[index, class_id] = 1.0
    tracker_metrics = evaluate_probe(
        IdentityProbe(),
        report_logits,
        torch.tensor(square_targets, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )
    move_recall, static_false_change_rate, diagnostics = compute_tracker_sequence_metrics(
        predictions_by_clip,
        tolerance=move_match_tolerance,
    )
    return {
        "lookahead_window": lookahead_window,
        "lookahead_margin": lookahead_margin,
        "move_match_tolerance": move_match_tolerance,
        "metrics": tracker_metrics.to_dict(),
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": static_false_change_rate,
        "sequence_diagnostics": diagnostics,
    }


def _initial_board_state_for_row(row: object) -> tuple[str, str | None]:
    clip_path = getattr(row, "clip_path", None)
    if not isinstance(clip_path, str):
        raise ValueError("Tracker eval requires clip_path on physical rows")
    clip = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    initial_board_fen = clip.get("initial_board_fen") if isinstance(clip, dict) else None
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    raw_side_to_move = clip.get("initial_side_to_move") if isinstance(clip, dict) else None
    side_to_move = raw_side_to_move if isinstance(raw_side_to_move, str) else None
    return initial_board_fen, side_to_move


def compute_tracker_sequence_metrics(
    predictions_by_clip: dict[str, list[FramePrediction]],
    *,
    tolerance: int,
) -> tuple[float, float, dict[str, object]]:
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")

    total_gt_moves = 0
    matched_gt_moves = 0
    total_static_frames = 0
    total_false_changes = 0
    diagnostics: dict[str, object] = {"clips": {}}

    for clip_path, frames in predictions_by_clip.items():
        gt_change_frames: list[int] = []
        predicted_change_frames: list[int] = []
        previous_target = None
        previous_prediction = None

        for frame in frames:
            if previous_target is not None:
                gt_changed = frame.target_labels != previous_target
                pred_changed = frame.predicted_labels != previous_prediction
                if gt_changed:
                    gt_change_frames.append(frame.frame_index)
                elif pred_changed:
                    total_false_changes += 1
                if not gt_changed:
                    total_static_frames += 1
                if pred_changed:
                    predicted_change_frames.append(frame.frame_index)
            previous_target = frame.target_labels
            previous_prediction = frame.predicted_labels

        matched_gt_frame_indices = _match_frames(
            gt_change_frames,
            predicted_change_frames,
            tolerance=tolerance,
        )
        total_gt_moves += len(gt_change_frames)
        matched_gt_moves += matched_gt_frame_indices
        diagnostics["clips"][clip_path] = {
            "gt_change_frames": gt_change_frames,
            "predicted_change_frames": predicted_change_frames,
            "matched_gt_changes": matched_gt_frame_indices,
        }

    move_recall = 0.0 if total_gt_moves == 0 else matched_gt_moves / total_gt_moves
    static_false_change_rate = (
        0.0 if total_static_frames == 0 else total_false_changes / total_static_frames
    )
    diagnostics["total_gt_change_frames"] = total_gt_moves
    diagnostics["matched_gt_change_frames"] = matched_gt_moves
    diagnostics["total_static_frames"] = total_static_frames
    diagnostics["false_change_frames"] = total_false_changes
    return move_recall, static_false_change_rate, diagnostics


def _match_frames(gt_frames: list[int], predicted_frames: list[int], *, tolerance: int) -> int:
    matched = 0
    used_predicted: set[int] = set()
    for gt_frame in gt_frames:
        for predicted_index, predicted_frame in enumerate(predicted_frames):
            if predicted_index in used_predicted:
                continue
            if abs(predicted_frame - gt_frame) <= tolerance:
                used_predicted.add(predicted_index)
                matched += 1
                break
    return matched


def load_previous_baselines() -> dict[str, object]:
    baselines: dict[str, object] = {}
    if _PREVIOUS_PIPELINE_TRACKER_PATH.exists():
        baselines["previous_best_pipeline_tracker"] = json.loads(
            _PREVIOUS_PIPELINE_TRACKER_PATH.read_text()
        )
    if _PREVIOUS_RAW_PROBE_PATH.exists():
        baselines["previous_best_raw_probe"] = json.loads(_PREVIOUS_RAW_PROBE_PATH.read_text())

    joint_paths = glob.glob(
        str(_PROJECT_ROOT / "outputs" / "2026-04-14" / "joint_board_reader*" / "summary.json")
    )
    best_joint = None
    best_joint_score = float("-inf")
    for path in joint_paths:
        payload = json.loads(Path(path).read_text())
        eval_metrics = payload.get("eval_metrics")
        if not isinstance(eval_metrics, dict):
            continue
        score = float(eval_metrics.get("non_empty_accuracy", 0.0)) + float(
            eval_metrics.get("macro_f1", 0.0)
        )
        if score > best_joint_score:
            best_joint_score = score
            best_joint = {"path": str(Path(path).relative_to(_PROJECT_ROOT)), **payload}
    if best_joint is not None:
        baselines["previous_best_joint_board_reader"] = best_joint
    return baselines


def build_comparison(
    *,
    physical_metrics: ProbeMetrics,
    physical_tracker_report: dict[str, object],
    previous_baselines: dict[str, object],
) -> dict[str, object]:
    comparison: dict[str, object] = {}

    previous_pipeline = previous_baselines.get("previous_best_pipeline_tracker")
    if isinstance(previous_pipeline, dict):
        previous_tracker_metrics = previous_pipeline.get("metrics")
        if isinstance(previous_tracker_metrics, dict):
            comparison["vs_previous_pipeline_tracker_board_exact"] = physical_tracker_report[
                "metrics"
            ]["board_exact_match"] - float(previous_tracker_metrics.get("board_exact_match", 0.0))

    previous_raw_probe = previous_baselines.get("previous_best_raw_probe")
    if isinstance(previous_raw_probe, dict):
        previous_raw_probe_metrics = previous_raw_probe.get("real_eval_metrics")
        if isinstance(previous_raw_probe_metrics, dict):
            comparison["vs_previous_raw_probe_non_empty_accuracy"] = (
                physical_metrics.non_empty_accuracy
                - float(previous_raw_probe_metrics.get("non_empty_accuracy", 0.0))
            )
            comparison["vs_previous_raw_probe_macro_f1"] = physical_metrics.macro_f1 - float(
                previous_raw_probe_metrics.get("macro_f1", 0.0)
            )

    previous_joint = previous_baselines.get("previous_best_joint_board_reader")
    if isinstance(previous_joint, dict):
        previous_joint_metrics = previous_joint.get("eval_metrics")
        if isinstance(previous_joint_metrics, dict):
            comparison["vs_previous_joint_reader_non_empty_accuracy"] = (
                physical_metrics.non_empty_accuracy
                - float(previous_joint_metrics.get("non_empty_accuracy", 0.0))
            )
            comparison["vs_previous_joint_reader_macro_f1"] = physical_metrics.macro_f1 - float(
                previous_joint_metrics.get("macro_f1", 0.0)
            )
    return comparison


def render_summary_markdown(summary: dict[str, object]) -> str:
    physical_metrics = summary["physical_val_metrics"]
    tracker_report = summary["physical_val_tracker_report"]
    chessred_metrics = summary["chessred_val_metrics"]
    comparison = summary["comparison"]
    lines = [
        "# Direct full-image board reader",
        "",
        "## Setup",
        f"- encoder: `{summary['encoder_type']}`",
        f"- model: `{summary['model_name']}`",
        f"- input size: `{summary['input_size']}`",
        "- train counts by domain: "
        f"`{json.dumps(summary['train_counts_by_domain'], sort_keys=True)}`",
        "",
        "## Raw metrics",
        f"- physical val board exact: `{physical_metrics['board_exact_match']:.4f}`",
        f"- physical val non-empty accuracy: `{physical_metrics['non_empty_accuracy']:.4f}`",
        f"- physical val macro-F1: `{physical_metrics['macro_f1']:.4f}`",
        f"- ChessReD val board exact: `{chessred_metrics['board_exact_match']:.4f}`",
        f"- ChessReD val non-empty accuracy: `{chessred_metrics['non_empty_accuracy']:.4f}`",
        f"- ChessReD val macro-F1: `{chessred_metrics['macro_f1']:.4f}`",
        "",
        "## Tracker metrics on physical val",
        f"- board exact: `{tracker_report['metrics']['board_exact_match']:.4f}`",
        f"- non-empty accuracy: `{tracker_report['metrics']['non_empty_accuracy']:.4f}`",
        f"- macro-F1: `{tracker_report['metrics']['macro_f1']:.4f}`",
        f"- move detection recall: `{tracker_report['move_detection_recall']:.4f}`",
        f"- static false-change rate: `{tracker_report['static_frame_false_change_rate']:.4f}`",
        "",
        "## Comparison to previous",
    ]
    for key, value in comparison.items():
        lines.append(f"- {key}: `{value:.4f}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
