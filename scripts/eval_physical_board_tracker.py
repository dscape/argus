#!/usr/bin/env python3
"""Evaluate a chess-aware state tracker on held-out physical-board sequences."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.replay import build_replay_board
from pipeline.physical.board_data import PhysicalEvalBoardDataset
from pipeline.physical.oblique_square_context import (
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.square_classifier import (
    PhysicalBoardLogitsSequenceReader,
    read_board_logits_from_frame,
)
from pipeline.physical.square_probe import evaluate_probe
from pipeline.shared import (
    SQUARE_CLASS_NAMES,
    LegalMoveStateTracker,
    LookaheadLegalMoveStateTracker,
    board_to_class_ids,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "physical_board_tracker_eval.json"


class FramePrediction(NamedTuple):
    annotation_id: str
    frame_index: int
    target_labels: tuple[int, ...]
    predicted_labels: tuple[int, ...]
    move_uci: str | None


class IdentityProbe(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate physical board state tracker")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument(
        "--observation-input",
        type=str,
        choices=["rectified_board", "original_oblique"],
        default="rectified_board",
    )
    parser.add_argument(
        "--temporal-mode",
        type=str,
        choices=["off", "fixed", "metadata"],
        default="metadata",
    )
    parser.add_argument("--temporal-ema-alpha", type=float, default=0.0)
    parser.add_argument(
        "--tracker-mode",
        type=str,
        choices=["greedy", "lookahead"],
        default="greedy",
    )
    parser.add_argument("--move-accept-threshold", type=float, default=2.5)
    parser.add_argument("--move-accept-margin", type=float, default=0.75)
    parser.add_argument("--lookahead-window", type=int, default=3)
    parser.add_argument("--lookahead-margin", type=float, default=8.0)
    parser.add_argument("--move-match-tolerance", type=int, default=1)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    if args.observation_input == "rectified_board":
        dataset = PhysicalEvalBoardDataset()
        rows = sorted(
            dataset.rows,
            key=lambda row: (
                row.clip_path or row.annotation_id,
                -1 if row.frame_index is None else row.frame_index,
                row.annotation_id,
            ),
        )
    else:
        rows = load_annotated_oblique_rows(_PROJECT_ROOT / "data" / "physical" / "val")

    if args.temporal_mode == "fixed":
        if args.temporal_ema_alpha <= 0.0:
            raise ValueError("--temporal-ema-alpha must be > 0 when --temporal-mode=fixed")
        logits_reader = PhysicalBoardLogitsSequenceReader(
            device=args.device,
            ema_alpha=args.temporal_ema_alpha,
            weights_path=args.weights_path,
        )
    elif args.temporal_mode == "metadata":
        logits_reader = PhysicalBoardLogitsSequenceReader(
            device=args.device,
            weights_path=args.weights_path,
        )
    else:
        logits_reader = None

    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    square_predictions: list[int] = []
    square_targets: list[int] = []
    board_annotation_ids: list[str] = []
    clip_cache: dict[Path, dict[str, object]] = {}
    missing_predictions = 0

    rows_by_clip: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        rows_by_clip[row.clip_path or row.annotation_id].append(row)

    for clip_key, clip_rows in rows_by_clip.items():
        clip_rows.sort(
            key=lambda row: (
                -1 if row.frame_index is None else row.frame_index,
                row.annotation_id,
            )
        )
        if logits_reader is not None:
            logits_reader.reset()

        clip_logits: list[torch.Tensor] = []
        usable_rows: list[object] = []
        for row in clip_rows:
            image = _load_row_image(
                row,
                observation_input=args.observation_input,
                clip_cache=clip_cache,
            )
            corners = getattr(row, "corners", None)
            if logits_reader is None:
                logits = read_board_logits_from_frame(
                    image,
                    corners=corners,
                    device=args.device,
                    weights_path=args.weights_path,
                )
            else:
                logits = logits_reader.read_board_logits_from_frame(image, corners=corners)
            if logits is None:
                missing_predictions += 1
                continue
            usable_rows.append(row)
            clip_logits.append(logits)

        if not usable_rows:
            continue

        if args.tracker_mode == "lookahead":
            sequence_results = LookaheadLegalMoveStateTracker(
                _initial_board_fen_for_row(usable_rows[0]),
                lookahead_window=args.lookahead_window,
                move_score_margin=args.lookahead_margin,
            ).decode(clip_logits)
        else:
            tracker = LegalMoveStateTracker(
                _initial_board_fen_for_row(usable_rows[0]),
                move_accept_threshold=args.move_accept_threshold,
                move_accept_margin=args.move_accept_margin,
            )
            sequence_results = [tracker.update(logits) for logits in clip_logits]

        for row, result in zip(usable_rows, sequence_results):
            predicted_labels = tuple(board_to_class_ids(build_replay_board(result.fen)))
            target_labels = tuple(int(value) for value in row.labels)
            predictions_by_clip[clip_key].append(
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

    logits = torch.zeros((len(square_predictions), len(SQUARE_CLASS_NAMES)), dtype=torch.float32)
    for index, class_id in enumerate(square_predictions):
        logits[index, class_id] = 1.0
    metrics = evaluate_probe(
        IdentityProbe(),
        logits,
        torch.tensor(square_targets, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )

    move_recall, static_false_change_rate, diagnostics = compute_tracker_sequence_metrics(
        predictions_by_clip,
        tolerance=args.move_match_tolerance,
    )
    report = {
        "missing_predictions": missing_predictions,
        "evaluated_boards": len(square_predictions) // 64,
        "observation_input": args.observation_input,
        "weights_path": None if args.weights_path is None else str(args.weights_path),
        "temporal_mode": args.temporal_mode,
        "temporal_ema_alpha": args.temporal_ema_alpha,
        "tracker_mode": args.tracker_mode,
        "move_accept_threshold": args.move_accept_threshold,
        "move_accept_margin": args.move_accept_margin,
        "lookahead_window": args.lookahead_window,
        "lookahead_margin": args.lookahead_margin,
        "move_match_tolerance": args.move_match_tolerance,
        "metrics": metrics.to_dict(),
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": static_false_change_rate,
        "sequence_diagnostics": diagnostics,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))


def _load_row_image(
    row: object,
    *,
    observation_input: str,
    clip_cache: dict[Path, dict[str, object]],
) -> cv2.typing.MatLike:
    if observation_input == "rectified_board":
        image = cv2.imread(str(_PROJECT_ROOT / row.board_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load board image: {row.board_path}")
        return image
    return _load_clip_frame_bgr(row, clip_cache=clip_cache)


def _initial_board_fen_for_row(row: object) -> str:
    clip_path = getattr(row, "clip_path", None)
    if not isinstance(clip_path, str):
        raise ValueError("Held-out eval row is missing clip_path")
    clip = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    initial_board_fen = clip.get("initial_board_fen") if isinstance(clip, dict) else None
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    return initial_board_fen


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


if __name__ == "__main__":
    main()
