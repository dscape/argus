"""Failure-study tooling for physical board-tracker runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chess
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pipeline.paths import PROJECT_ROOT
from pipeline.physical.board_data import PhysicalEvalBoardDataset, PhysicalEvalBoardRow
from pipeline.physical.oblique_square_context import (
    PhysicalObliqueBoardRow,
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.square_classifier import (
    PhysicalBoardLogitsSequenceReader,
    board_observation_from_logits,
    read_board_logits_batch_from_frames,
)
from pipeline.shared import (
    SQUARE_CLASS_NAMES,
    BoardTrackerResult,
    LegalMoveStateTracker,
    LookaheadLegalMoveStateTracker,
    SequenceTrackerFrameResult,
    board_to_class_ids,
    score_board_state,
)

ObservationInput = Literal["rectified_board", "original_oblique"]
TemporalMode = Literal["off", "fixed", "metadata"]
TrackerMode = Literal["greedy", "lookahead"]
ObservationRow = PhysicalEvalBoardRow | PhysicalObliqueBoardRow
DecodedFrameResult = BoardTrackerResult | SequenceTrackerFrameResult

_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "physical_board_failure_study"
_LIGHT_SQUARE = (240, 217, 181)
_DARK_SQUARE = (181, 136, 99)
_CORRECT_COLOR = (51, 171, 88)
_ERROR_COLOR = (204, 65, 65)
_GT_CHANGED_BORDER = (244, 208, 63)
_PRED_CHANGED_BORDER = (73, 160, 255)
_TEXT_COLOR = (24, 24, 24)
_GRID_COLOR = (40, 40, 40)
_HEADER_BG = (248, 248, 248)
_PANEL_BG = (255, 255, 255)
_CONTACT_SHEET_BG = (245, 245, 245)
_EMPTY_CLASS_ID = 0
_INFERENCE_BATCH_SIZE = 16
_BUCKET_GUIDE = """# Manual failure buckets

Use `final_bucket` in `manual_buckets.csv`.

- `rectification`: board geometry is visibly wrong in the crop
  (warped grid, shifted squares, clipped files/ranks)
- `classifier`: geometry looks usable, but the board evidence itself is wrong
  (piece identity, occupancy, blur/confusable classes)
- `tracker_boundary_jitter`: decoded board matches the previous or next GT board;
  move timing is off by about a frame
- `tracker_desync`: decoded sequence inserted/missed a move and diverged from GT
- `eval_artifact`: annotation/eval issue, not a model failure

`Suggested_root_cause` is only a heuristic. Override it manually.
"""
_CLASS_ID_TO_SYMBOL = {
    class_id: ("" if name == "empty" else name)
    for class_id, name in enumerate(SQUARE_CLASS_NAMES)
}


@dataclass(frozen=True)
class TrackerFailureStudyConfig:
    observation_input: ObservationInput = "rectified_board"
    temporal_mode: TemporalMode = "off"
    temporal_ema_alpha: float = 0.0
    tracker_mode: TrackerMode = "lookahead"
    move_accept_threshold: float = 2.5
    move_accept_margin: float = 0.75
    lookahead_window: int = 3
    lookahead_margin: float = 8.0
    weights_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_input": self.observation_input,
            "temporal_mode": self.temporal_mode,
            "temporal_ema_alpha": self.temporal_ema_alpha,
            "tracker_mode": self.tracker_mode,
            "move_accept_threshold": self.move_accept_threshold,
            "move_accept_margin": self.move_accept_margin,
            "lookahead_window": self.lookahead_window,
            "lookahead_margin": self.lookahead_margin,
            "weights_path": self.weights_path,
        }


@dataclass(frozen=True)
class _FailureRecord:
    row: ObservationRow
    payload: dict[str, Any]


def load_config_from_eval_report(report_path: str | Path) -> TrackerFailureStudyConfig:
    payload = json.loads(Path(report_path).read_text())
    tracker_mode = str(payload.get("tracker_mode", ""))
    if tracker_mode not in {"greedy", "lookahead"}:
        raise ValueError(
            f"Tracker eval report is missing tracker_mode=greedy/lookahead: {report_path}"
        )

    observation_input = str(payload.get("observation_input", "rectified_board"))
    if observation_input not in {"rectified_board", "original_oblique"}:
        raise ValueError(f"Unsupported observation_input in {report_path}: {observation_input}")

    temporal_mode = str(payload.get("temporal_mode", "off"))
    if temporal_mode not in {"off", "fixed", "metadata"}:
        raise ValueError(f"Unsupported temporal_mode in {report_path}: {temporal_mode}")

    return TrackerFailureStudyConfig(
        observation_input=observation_input,
        temporal_mode=temporal_mode,
        temporal_ema_alpha=float(payload.get("temporal_ema_alpha", 0.0)),
        tracker_mode=tracker_mode,
        move_accept_threshold=float(payload.get("move_accept_threshold", 2.5)),
        move_accept_margin=float(payload.get("move_accept_margin", 0.75)),
        lookahead_window=int(payload.get("lookahead_window", 3)),
        lookahead_margin=float(payload.get("lookahead_margin", 8.0)),
        weights_path=(
            str(payload["weights_path"])
            if isinstance(payload.get("weights_path"), str) and payload.get("weights_path")
            else None
        ),
    )


def create_tracker_failure_study(
    *,
    config: TrackerFailureStudyConfig,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    limit: int = 100,
    device: str = "cpu",
    panel_size: int = 240,
    top_legal_candidates: int = 5,
    sample_mode: Literal["first", "round_robin"] = "round_robin",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be > 0, got {panel_size}")
    if top_legal_candidates <= 0:
        raise ValueError(f"top_legal_candidates must be > 0, got {top_legal_candidates}")

    rows_by_clip = _load_rows_by_clip(config.observation_input)
    failures = _collect_failures(
        rows_by_clip,
        config=config,
        device=device,
        top_legal_candidates=top_legal_candidates,
    )
    selected_failures = _select_failures(failures, limit=limit, sample_mode=sample_mode)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    rendered_images: list[Image.Image] = []
    manifest: list[dict[str, Any]] = []
    clip_cache: dict[Path, dict[str, Any]] = {}
    for index, failure in enumerate(selected_failures, start=1):
        image_bgr = _load_row_image(
            failure.row,
            observation_input=config.observation_input,
            clip_cache=clip_cache,
        )
        rendered = _render_failure_frame(
            failure.payload,
            crop_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            panel_size=panel_size,
        )
        filename = (
            f"{index:03d}_{_safe_filename_stem(Path(str(failure.payload['clip_filename'])).stem)}"
            f"_f{int(failure.payload['frame_index']):04d}.png"
        )
        rendered.save(frame_dir / filename)
        rendered_images.append(rendered)

        payload = dict(failure.payload)
        payload["image_path"] = _relative_to_project(frame_dir / filename)
        payload["selected_index"] = index
        manifest.append(payload)

    contact_sheet_path = output_path / "contact_sheet.png"
    if rendered_images:
        contact_sheet = _render_contact_sheet(
            rendered_images,
            title=(
                f"physical board failure study · {config.tracker_mode} · "
                f"{config.observation_input}"
            ),
            subtitle=(
                f"selected {len(rendered_images)} of {len(failures)} board_exact failures"
            ),
        )
        contact_sheet.save(contact_sheet_path)

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    buckets_path = output_path / "manual_buckets.csv"
    _write_manual_buckets_csv(buckets_path, manifest)

    guide_path = output_path / "BUCKETS.md"
    guide_path.write_text(_BUCKET_GUIDE)

    per_clip_failure_counts = Counter(str(record.payload["clip_path"]) for record in failures)
    suggested_root_cause_counts = Counter(
        str(record.payload["suggested_root_cause"]) for record in selected_failures
    )
    summary = {
        "report_path": None if report_path is None else _relative_to_project(Path(report_path)),
        "config": config.to_dict(),
        "total_clips": len(rows_by_clip),
        "total_failures": len(failures),
        "selected_failures": len(manifest),
        "sample_mode": sample_mode,
        "panel_size": panel_size,
        "top_legal_candidates": top_legal_candidates,
        "manifest": _relative_to_project(manifest_path),
        "manual_buckets_csv": _relative_to_project(buckets_path),
        "bucket_guide": _relative_to_project(guide_path),
        "contact_sheet": (
            _relative_to_project(contact_sheet_path) if rendered_images else None
        ),
        "per_clip_failure_counts": dict(sorted(per_clip_failure_counts.items())),
        "suggested_root_cause_counts": dict(sorted(suggested_root_cause_counts.items())),
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary["summary_path"] = _relative_to_project(summary_path)
    return summary


def _load_rows_by_clip(
    observation_input: ObservationInput,
) -> dict[str, list[ObservationRow]]:
    if observation_input == "rectified_board":
        rows: list[ObservationRow] = list(PhysicalEvalBoardDataset().rows)
    else:
        rows = list(load_annotated_oblique_rows(PROJECT_ROOT / "data" / "physical" / "val"))

    rows_by_clip: dict[str, list[ObservationRow]] = defaultdict(list)
    for row in rows:
        clip_key = row.clip_path or row.annotation_id
        rows_by_clip[str(clip_key)].append(row)

    return {
        clip_key: sorted(
            clip_rows,
            key=lambda row: (_frame_index(row), row.annotation_id),
        )
        for clip_key, clip_rows in rows_by_clip.items()
    }


def _collect_failures(
    rows_by_clip: dict[str, list[ObservationRow]],
    *,
    config: TrackerFailureStudyConfig,
    device: str,
    top_legal_candidates: int,
) -> list[_FailureRecord]:
    clip_state_cache: dict[str, tuple[str, str | None]] = {}
    clip_cache: dict[Path, dict[str, Any]] = {}
    failures: list[_FailureRecord] = []

    for clip_key, clip_rows in sorted(rows_by_clip.items()):
        initial_board_fen, initial_side_to_move = _initial_board_state(
            clip_key,
            clip_state_cache=clip_state_cache,
        )
        raw_logits = _read_clip_logits(
            clip_rows,
            observation_input=config.observation_input,
            device=device,
            weights_path=config.weights_path,
            clip_cache=clip_cache,
        )
        tracker_input_logits = _apply_temporal_mode(
            raw_logits,
            config=config,
            device=device,
        )
        tracker_log_probs = [torch.log_softmax(logits, dim=1) for logits in tracker_input_logits]
        decoded_results = _decode_clip(
            tracker_input_logits,
            initial_board_fen=initial_board_fen,
            initial_side_to_move=initial_side_to_move,
            config=config,
        )

        gt_fens = [
            _class_ids_to_fen(tuple(int(value) for value in row.labels))
            for row in clip_rows
        ]
        previous_gt_class_ids: tuple[int, ...] | None = None
        previous_stateless_class_ids: tuple[int, ...] | None = None
        previous_tracker_input_class_ids: tuple[int, ...] | None = None
        previous_decoded_class_ids: tuple[int, ...] | None = None
        previous_decoded_board: chess.Board | None = None

        for frame_offset, (row, raw_logit, tracker_logit, decoded_result) in enumerate(
            zip(clip_rows, raw_logits, tracker_input_logits, decoded_results)
        ):
            probabilities = torch.softmax(raw_logit, dim=1)
            gt_class_ids = tuple(int(value) for value in row.labels)
            stateless_observation = board_observation_from_logits(raw_logit)
            tracker_input_observation = board_observation_from_logits(tracker_logit)
            stateless_class_ids = tuple(_fen_to_class_ids(stateless_observation.fen))
            tracker_input_class_ids = tuple(_fen_to_class_ids(tracker_input_observation.fen))
            decoded_board = chess.Board(decoded_result.full_fen)
            decoded_class_ids = tuple(board_to_class_ids(decoded_board))

            stateless_error_count = _count_differences(stateless_class_ids, gt_class_ids)
            tracker_input_error_count = _count_differences(
                tracker_input_class_ids,
                gt_class_ids,
            )
            decoded_error_count = _count_differences(decoded_class_ids, gt_class_ids)
            if decoded_error_count == 0:
                previous_gt_class_ids = gt_class_ids
                previous_stateless_class_ids = stateless_class_ids
                previous_tracker_input_class_ids = tracker_input_class_ids
                previous_decoded_class_ids = decoded_class_ids
                previous_decoded_board = decoded_board.copy(stack=False)
                continue

            gt_fen = gt_fens[frame_offset]
            previous_gt_fen = None if previous_gt_class_ids is None else gt_fens[frame_offset - 1]
            next_gt_fen = None if frame_offset + 1 >= len(gt_fens) else gt_fens[frame_offset + 1]

            legal_diagnostics = None
            if previous_decoded_board is not None:
                legal_diagnostics = _legal_transition_diagnostics(
                    previous_board=previous_decoded_board,
                    gt_fen=gt_fen,
                    log_probs=tracker_log_probs,
                    frame_index=frame_offset,
                    lookahead_window=(
                        config.lookahead_window if config.tracker_mode == "lookahead" else 1
                    ),
                    top_legal_candidates=top_legal_candidates,
                )

            square_diagnostics = _square_diagnostics(
                probabilities=probabilities,
                gt_class_ids=gt_class_ids,
                stateless_class_ids=stateless_class_ids,
                decoded_class_ids=decoded_class_ids,
                tracker_input_class_ids=tracker_input_class_ids,
                previous_gt_class_ids=previous_gt_class_ids,
                previous_decoded_class_ids=previous_decoded_class_ids,
            )

            payload = {
                "annotation_id": row.annotation_id,
                "clip_path": row.clip_path,
                "clip_filename": Path(str(row.clip_path or row.annotation_id)).name,
                "frame_index": _frame_index(row),
                "board_path": getattr(row, "board_path", None),
                "source_video_id": row.source_video_id,
                "gt_fen": gt_fen,
                "stateless_fen": stateless_observation.fen,
                "tracker_input_fen": tracker_input_observation.fen,
                "decoded_fen": decoded_board.board_fen(),
                "decoded_full_fen": decoded_result.full_fen,
                "decoded_move_uci": decoded_result.move_uci,
                "decoded_move_score": round(float(decoded_result.move_score), 4),
                "decoded_stay_score": round(float(decoded_result.stay_score), 4),
                "gt_changed_squares": _mask_to_square_names(
                    _changed_mask(gt_class_ids, previous_gt_class_ids)
                ),
                "stateless_changed_squares": _mask_to_square_names(
                    _changed_mask(stateless_class_ids, previous_stateless_class_ids)
                ),
                "tracker_input_changed_squares": _mask_to_square_names(
                    _changed_mask(
                        tracker_input_class_ids,
                        previous_tracker_input_class_ids,
                    )
                ),
                "decoded_changed_squares": _mask_to_square_names(
                    _changed_mask(decoded_class_ids, previous_decoded_class_ids)
                ),
                "stateless_error_squares": _difference_square_names(
                    stateless_class_ids,
                    gt_class_ids,
                ),
                "tracker_input_error_squares": _difference_square_names(
                    tracker_input_class_ids,
                    gt_class_ids,
                ),
                "decoded_error_squares": _difference_square_names(
                    decoded_class_ids,
                    gt_class_ids,
                ),
                "stateless_error_count": stateless_error_count,
                "tracker_input_error_count": tracker_input_error_count,
                "decoded_error_count": decoded_error_count,
                "stateless_exact_match": stateless_error_count == 0,
                "tracker_input_exact_match": tracker_input_error_count == 0,
                "decoded_exact_match": False,
                "decoded_matches_previous_gt": decoded_board.board_fen() == previous_gt_fen,
                "decoded_matches_next_gt": decoded_board.board_fen() == next_gt_fen,
                "stateless_matches_previous_gt": stateless_observation.fen == previous_gt_fen,
                "stateless_matches_next_gt": stateless_observation.fen == next_gt_fen,
                "legal_from_previous_decoded": legal_diagnostics,
                "square_diagnostics": square_diagnostics,
                "stateless_square_confidences": _class_confidences(
                    probabilities,
                    stateless_class_ids,
                ),
                "decoded_square_confidences": _class_confidences(
                    probabilities,
                    decoded_class_ids,
                ),
            }
            payload["suggested_root_cause"] = _suggested_root_cause(payload)
            failures.append(_FailureRecord(row=row, payload=payload))

            previous_gt_class_ids = gt_class_ids
            previous_stateless_class_ids = stateless_class_ids
            previous_tracker_input_class_ids = tracker_input_class_ids
            previous_decoded_class_ids = decoded_class_ids
            previous_decoded_board = decoded_board.copy(stack=False)

    failures.sort(
        key=lambda record: (
            str(record.payload["clip_path"]),
            int(record.payload["frame_index"]),
            record.payload["annotation_id"],
        )
    )
    return failures


def _initial_board_state(
    clip_path: str,
    *,
    clip_state_cache: dict[str, tuple[str, str | None]],
) -> tuple[str, str | None]:
    cached = clip_state_cache.get(clip_path)
    if cached is not None:
        return cached

    clip = torch.load(PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    if not isinstance(clip, dict):
        raise ValueError(f"Invalid clip payload: {clip_path}")
    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    raw_side_to_move = clip.get("initial_side_to_move")
    side_to_move = raw_side_to_move if isinstance(raw_side_to_move, str) else None
    state = (initial_board_fen, side_to_move)
    clip_state_cache[clip_path] = state
    return state


def _read_clip_logits(
    clip_rows: list[ObservationRow],
    *,
    observation_input: ObservationInput,
    device: str,
    weights_path: str | None,
    clip_cache: dict[Path, dict[str, Any]],
) -> list[torch.Tensor]:
    logits: list[torch.Tensor] = []
    for start in range(0, len(clip_rows), _INFERENCE_BATCH_SIZE):
        chunk_rows = clip_rows[start : start + _INFERENCE_BATCH_SIZE]
        chunk_images = [
            _load_row_image(
                row,
                observation_input=observation_input,
                clip_cache=clip_cache,
            )
            for row in chunk_rows
        ]
        corners_list = None
        if observation_input == "original_oblique":
            corners_list = [list(row.corners) for row in chunk_rows]
        chunk_logits = read_board_logits_batch_from_frames(
            chunk_images,
            corners_list=corners_list,
            device=device,
            weights_path=weights_path,
            batch_size=_INFERENCE_BATCH_SIZE,
        )
        if chunk_logits is None:
            raise FileNotFoundError("Failed to load physical board logits for failure study")
        logits.extend(chunk_logits)
    return logits


def _apply_temporal_mode(
    raw_logits: list[torch.Tensor],
    *,
    config: TrackerFailureStudyConfig,
    device: str,
) -> list[torch.Tensor]:
    if config.temporal_mode == "off":
        return raw_logits
    if config.temporal_mode == "fixed" and config.temporal_ema_alpha <= 0.0:
        raise ValueError("temporal_ema_alpha must be > 0 when temporal_mode=fixed")

    smoother = PhysicalBoardLogitsSequenceReader(
        device=device,
        ema_alpha=(config.temporal_ema_alpha if config.temporal_mode == "fixed" else None),
        weights_path=config.weights_path,
    )
    smoother.reset()
    return [smoother.smooth_logits(logit) for logit in raw_logits]


def _decode_clip(
    tracker_input_logits: list[torch.Tensor],
    *,
    initial_board_fen: str,
    initial_side_to_move: str | None,
    config: TrackerFailureStudyConfig,
) -> list[DecodedFrameResult]:
    if config.tracker_mode == "lookahead":
        return list(
            LookaheadLegalMoveStateTracker(
                initial_board_fen,
                initial_side_to_move=initial_side_to_move,
                lookahead_window=config.lookahead_window,
                move_score_margin=config.lookahead_margin,
            ).decode(tracker_input_logits)
        )

    tracker = LegalMoveStateTracker(
        initial_board_fen,
        initial_side_to_move=initial_side_to_move,
        move_accept_threshold=config.move_accept_threshold,
        move_accept_margin=config.move_accept_margin,
    )
    return [tracker.update(logit) for logit in tracker_input_logits]


def _legal_transition_diagnostics(
    *,
    previous_board: chess.Board,
    gt_fen: str,
    log_probs: list[torch.Tensor],
    frame_index: int,
    lookahead_window: int,
    top_legal_candidates: int,
) -> dict[str, Any]:
    candidates: list[tuple[str | None, chess.Board, float]] = []
    stay_board = previous_board.copy(stack=False)
    candidates.append(
        (
            None,
            stay_board,
            _windowed_board_score(
                log_probs,
                board=stay_board,
                frame_index=frame_index,
                lookahead_window=lookahead_window,
            ),
        )
    )
    for move in previous_board.legal_moves:
        next_board = previous_board.copy(stack=False)
        next_board.push(move)
        candidates.append(
            (
                move.uci(),
                next_board,
                _windowed_board_score(
                    log_probs,
                    board=next_board,
                    frame_index=frame_index,
                    lookahead_window=lookahead_window,
                ),
            )
        )

    candidates.sort(key=lambda item: item[2], reverse=True)
    top_candidates = candidates[:top_legal_candidates]
    gt_candidate = next(
        (
            (move_uci, board, score)
            for move_uci, board, score in candidates
            if board.board_fen() == gt_fen
        ),
        None,
    )
    best_move_uci, best_board, best_score = candidates[0]
    gt_rank = None
    gt_score = None
    gt_move_uci = None
    if gt_candidate is not None:
        gt_move_uci, _gt_board, gt_score = gt_candidate
        for rank, (_move_uci, board, _score) in enumerate(candidates, start=1):
            if board.board_fen() == gt_fen:
                gt_rank = rank
                break

    top_payload = [
        {
            "move_uci": move_uci,
            "fen": board.board_fen(),
            "score": round(float(score), 4),
            "matches_gt": board.board_fen() == gt_fen,
        }
        for move_uci, board, score in top_candidates
    ]
    return {
        "best_legal_move_uci": best_move_uci,
        "best_legal_fen": best_board.board_fen(),
        "best_legal_score": round(float(best_score), 4),
        "best_legal_matches_gt": best_board.board_fen() == gt_fen,
        "gt_is_legal_successor": gt_candidate is not None,
        "gt_legal_rank": gt_rank,
        "gt_legal_move_uci": gt_move_uci,
        "gt_legal_score": None if gt_score is None else round(float(gt_score), 4),
        "gt_score_gap_to_best": (
            None if gt_score is None else round(float(best_score - gt_score), 4)
        ),
        "top_legal_candidates": top_payload,
    }


def _square_diagnostics(
    *,
    probabilities: torch.Tensor,
    gt_class_ids: tuple[int, ...],
    stateless_class_ids: tuple[int, ...],
    decoded_class_ids: tuple[int, ...],
    tracker_input_class_ids: tuple[int, ...],
    previous_gt_class_ids: tuple[int, ...] | None,
    previous_decoded_class_ids: tuple[int, ...] | None,
) -> list[dict[str, Any]]:
    changed_gt = _changed_mask(gt_class_ids, previous_gt_class_ids)
    changed_decoded = _changed_mask(decoded_class_ids, previous_decoded_class_ids)
    highlight_indices = {
        index
        for index, (gt_value, stateless_value, decoded_value, tracker_value) in enumerate(
            zip(
                gt_class_ids,
                stateless_class_ids,
                decoded_class_ids,
                tracker_input_class_ids,
            )
        )
        if (
            gt_value != stateless_value
            or gt_value != decoded_value
            or gt_value != tracker_value
            or changed_gt[index]
            or changed_decoded[index]
        )
    }

    if not highlight_indices:
        return []

    top_probs, top_indices = probabilities.topk(k=2, dim=1)
    payload: list[dict[str, Any]] = []
    for square_index in sorted(highlight_indices):
        payload.append(
            {
                "square": _square_name(square_index),
                "gt_class": SQUARE_CLASS_NAMES[gt_class_ids[square_index]],
                "stateless_class": SQUARE_CLASS_NAMES[stateless_class_ids[square_index]],
                "tracker_input_class": SQUARE_CLASS_NAMES[
                    tracker_input_class_ids[square_index]
                ],
                "decoded_class": SQUARE_CLASS_NAMES[decoded_class_ids[square_index]],
                "top1_class": SQUARE_CLASS_NAMES[int(top_indices[square_index, 0].item())],
                "top1_prob": round(float(top_probs[square_index, 0].item()), 4),
                "top2_class": SQUARE_CLASS_NAMES[int(top_indices[square_index, 1].item())],
                "top2_prob": round(float(top_probs[square_index, 1].item()), 4),
                "gt_changed": bool(changed_gt[square_index]),
                "decoded_changed": bool(changed_decoded[square_index]),
            }
        )
    return payload


def _suggested_root_cause(payload: dict[str, Any]) -> str:
    if bool(payload["decoded_matches_previous_gt"]) or bool(payload["decoded_matches_next_gt"]):
        return "tracker_boundary_jitter"

    legal = payload.get("legal_from_previous_decoded")
    if isinstance(legal, dict) and bool(legal.get("best_legal_matches_gt")):
        return "tracker_desync"

    if bool(payload.get("stateless_exact_match")) or bool(payload.get("tracker_input_exact_match")):
        return "tracker_desync"

    return "rectification_or_classifier"


def _select_failures(
    failures: list[_FailureRecord],
    *,
    limit: int,
    sample_mode: Literal["first", "round_robin"],
) -> list[_FailureRecord]:
    if len(failures) <= limit or sample_mode == "first":
        return failures[:limit]

    by_clip: dict[str, list[_FailureRecord]] = defaultdict(list)
    for failure in failures:
        by_clip[str(failure.payload["clip_path"])].append(failure)

    selected: list[_FailureRecord] = []
    clip_keys = sorted(by_clip)
    while len(selected) < limit and clip_keys:
        next_clip_keys: list[str] = []
        for clip_key in clip_keys:
            clip_failures = by_clip[clip_key]
            if not clip_failures:
                continue
            selected.append(clip_failures.pop(0))
            if len(selected) >= limit:
                break
            if clip_failures:
                next_clip_keys.append(clip_key)
        clip_keys = next_clip_keys
    return selected


def _load_row_image(
    row: ObservationRow,
    *,
    observation_input: ObservationInput,
    clip_cache: dict[Path, dict[str, Any]],
) -> np.ndarray:
    if observation_input == "rectified_board":
        board_path = getattr(row, "board_path", None)
        if not isinstance(board_path, str):
            raise ValueError(f"Rectified row is missing board_path: {row.annotation_id}")
        image = cv2.imread(str(PROJECT_ROOT / board_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load board image: {board_path}")
        return image
    return _load_clip_frame_bgr(row, clip_cache=clip_cache)


def _render_failure_frame(
    payload: dict[str, Any],
    *,
    crop_rgb: np.ndarray,
    panel_size: int,
) -> Image.Image:
    margin = 12
    panel_gap = 12
    header_height = 68
    panel_labels_height = 22
    width = margin * 2 + panel_size * 4 + panel_gap * 3
    height = margin * 2 + header_height + panel_labels_height + panel_size

    image = Image.new("RGB", (width, height), _PANEL_BG)
    draw = ImageDraw.Draw(image)
    header_font = _load_font(20)
    body_font = _load_font(15)

    move_label = payload["decoded_move_uci"] or "stay"
    legal = payload.get("legal_from_previous_decoded") or {}
    rank_label = legal.get("gt_legal_rank")
    rank_text = "-" if rank_label is None else str(rank_label)
    header_text = (
        f"{payload['clip_filename']} · f{int(payload['frame_index']):04d} · "
        f"move={move_label} · suggested={payload['suggested_root_cause']}"
    )
    metrics_text = (
        f"single err={payload['stateless_error_count']:02d} "
        f"tracker err={payload['decoded_error_count']:02d} "
        f"gt legal rank={rank_text}"
    )
    flags_text = (
        f"prevGT={_flag(payload['decoded_matches_previous_gt'])} "
        f"nextGT={_flag(payload['decoded_matches_next_gt'])} "
        f"bestLegalGT={_flag(bool(legal.get('best_legal_matches_gt')))}"
    )

    draw.rectangle((0, 0, width, margin + header_height), fill=_HEADER_BG)
    draw.text((margin, margin), header_text, fill=_TEXT_COLOR, font=header_font)
    draw.text((margin, margin + 24), metrics_text, fill=_TEXT_COLOR, font=body_font)
    draw.text((margin, margin + 44), flags_text, fill=_TEXT_COLOR, font=body_font)

    top = margin + header_height + panel_labels_height
    lefts = [margin + (panel_size + panel_gap) * index for index in range(4)]
    labels = ["crop", "ground truth", "stateless", "decoded"]
    gt_class_ids = tuple(_fen_to_class_ids(str(payload["gt_fen"])))
    stateless_class_ids = tuple(_fen_to_class_ids(str(payload["stateless_fen"])))
    decoded_class_ids = tuple(_fen_to_class_ids(str(payload["decoded_fen"])))
    gt_changed_mask = _square_mask(payload.get("gt_changed_squares"))
    stateless_changed_mask = _square_mask(payload.get("stateless_changed_squares"))
    decoded_changed_mask = _square_mask(payload.get("decoded_changed_squares"))

    panels = [
        _render_crop_panel(
            crop_rgb,
            panel_size=panel_size,
            gt_class_ids=gt_class_ids,
            decoded_class_ids=decoded_class_ids,
            decoded_confidences=tuple(payload["decoded_square_confidences"]),
            gt_changed_mask=gt_changed_mask,
            decoded_changed_mask=decoded_changed_mask,
        ),
        _render_board_panel(
            gt_class_ids,
            panel_size=panel_size,
            confidences=None,
            target_class_ids=gt_class_ids,
            explicit_changed_mask=gt_changed_mask,
            border_color=_GT_CHANGED_BORDER,
        ),
        _render_board_panel(
            stateless_class_ids,
            panel_size=panel_size,
            confidences=tuple(payload["stateless_square_confidences"]),
            target_class_ids=gt_class_ids,
            explicit_changed_mask=stateless_changed_mask,
            border_color=_PRED_CHANGED_BORDER,
        ),
        _render_board_panel(
            decoded_class_ids,
            panel_size=panel_size,
            confidences=tuple(payload["decoded_square_confidences"]),
            target_class_ids=gt_class_ids,
            explicit_changed_mask=decoded_changed_mask,
            border_color=_PRED_CHANGED_BORDER,
        ),
    ]

    for left, label, panel in zip(lefts, labels, panels):
        draw.text((left, margin + header_height), label, fill=_TEXT_COLOR, font=body_font)
        image.paste(panel, (left, top))

    return image


def _render_crop_panel(
    crop_rgb: np.ndarray,
    *,
    panel_size: int,
    gt_class_ids: tuple[int, ...],
    decoded_class_ids: tuple[int, ...],
    decoded_confidences: tuple[float, ...],
    gt_changed_mask: tuple[bool, ...],
    decoded_changed_mask: tuple[bool, ...],
) -> Image.Image:
    image = Image.fromarray(crop_rgb).resize(
        (panel_size, panel_size),
        Image.Resampling.BILINEAR,
    )
    overlay = Image.new("RGBA", (panel_size, panel_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell = panel_size / 8.0

    for square_index, (target, predicted, confidence) in enumerate(
        zip(gt_class_ids, decoded_class_ids, decoded_confidences)
    ):
        row = square_index // 8
        col = square_index % 8
        x0 = int(round(col * cell))
        y0 = int(round(row * cell))
        x1 = int(round((col + 1) * cell))
        y1 = int(round((row + 1) * cell))

        if target != predicted:
            alpha = 40 + int(110 * float(confidence))
            draw.rectangle((x0, y0, x1, y1), fill=(*_ERROR_COLOR, alpha))

        if gt_changed_mask[square_index]:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=_GT_CHANGED_BORDER, width=3)
        elif decoded_changed_mask[square_index]:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=_PRED_CHANGED_BORDER, width=2)

    grid_draw = ImageDraw.Draw(overlay)
    for line_index in range(9):
        x = int(round(line_index * cell))
        y = int(round(line_index * cell))
        grid_draw.line((x, 0, x, panel_size), fill=(*_GRID_COLOR, 180), width=1)
        grid_draw.line((0, y, panel_size, y), fill=(*_GRID_COLOR, 180), width=1)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _render_board_panel(
    class_ids: tuple[int, ...],
    *,
    panel_size: int,
    confidences: tuple[float, ...] | None,
    target_class_ids: tuple[int, ...],
    explicit_changed_mask: tuple[bool, ...],
    border_color: tuple[int, int, int],
) -> Image.Image:
    image = Image.new("RGB", (panel_size, panel_size), _PANEL_BG)
    draw = ImageDraw.Draw(image)
    cell = panel_size / 8.0
    symbol_font = _load_font(max(14, int(cell * 0.55)))

    confidence_values = confidences or (1.0,) * 64
    for square_index, (class_id, target_class_id, confidence) in enumerate(
        zip(class_ids, target_class_ids, confidence_values)
    ):
        row = square_index // 8
        col = square_index % 8
        x0 = int(round(col * cell))
        y0 = int(round(row * cell))
        x1 = int(round((col + 1) * cell))
        y1 = int(round((row + 1) * cell))
        base_color = _LIGHT_SQUARE if (row + col) % 2 == 0 else _DARK_SQUARE
        if confidences is None:
            fill = base_color
        else:
            tint = _CORRECT_COLOR if class_id == target_class_id else _ERROR_COLOR
            fill = _blend(base_color, tint, 0.20 + 0.55 * float(confidence))
        draw.rectangle((x0, y0, x1, y1), fill=fill)

        if explicit_changed_mask[square_index]:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=border_color, width=3)

        symbol = _CLASS_ID_TO_SYMBOL[class_id]
        if symbol:
            text_bbox = draw.textbbox((0, 0), symbol, font=symbol_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                (
                    x0 + (x1 - x0 - text_width) / 2,
                    y0 + (y1 - y0 - text_height) / 2 - 1,
                ),
                symbol,
                fill=_TEXT_COLOR,
                font=symbol_font,
            )

    for line_index in range(9):
        x = int(round(line_index * cell))
        y = int(round(line_index * cell))
        draw.line((x, 0, x, panel_size), fill=_GRID_COLOR, width=1)
        draw.line((0, y, panel_size, y), fill=_GRID_COLOR, width=1)
    return image


def _render_contact_sheet(
    frame_images: list[Image.Image],
    *,
    title: str,
    subtitle: str,
) -> Image.Image:
    if not frame_images:
        raise ValueError("frame_images must be non-empty")

    width = max(image.width for image in frame_images)
    title_height = 76
    total_height = title_height + sum(image.height for image in frame_images)
    contact_sheet = Image.new("RGB", (width, total_height), _CONTACT_SHEET_BG)
    draw = ImageDraw.Draw(contact_sheet)
    title_font = _load_font(24)
    body_font = _load_font(16)

    draw.rectangle((0, 0, width, title_height), fill=_HEADER_BG)
    draw.text((12, 10), title, fill=_TEXT_COLOR, font=title_font)
    draw.text((12, 42), subtitle, fill=_TEXT_COLOR, font=body_font)

    top = title_height
    for frame_image in frame_images:
        contact_sheet.paste(frame_image, (0, top))
        top += frame_image.height
    return contact_sheet


def _write_manual_buckets_csv(path: Path, manifest: list[dict[str, Any]]) -> None:
    fieldnames = [
        "selected_index",
        "annotation_id",
        "clip_path",
        "frame_index",
        "suggested_root_cause",
        "decoded_error_count",
        "stateless_error_count",
        "decoded_matches_previous_gt",
        "decoded_matches_next_gt",
        "best_legal_matches_gt",
        "best_legal_move_uci",
        "gt_legal_rank",
        "image_path",
        "final_bucket",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            legal = row.get("legal_from_previous_decoded") or {}
            writer.writerow(
                {
                    "selected_index": row.get("selected_index"),
                    "annotation_id": row.get("annotation_id"),
                    "clip_path": row.get("clip_path"),
                    "frame_index": row.get("frame_index"),
                    "suggested_root_cause": row.get("suggested_root_cause"),
                    "decoded_error_count": row.get("decoded_error_count"),
                    "stateless_error_count": row.get("stateless_error_count"),
                    "decoded_matches_previous_gt": row.get("decoded_matches_previous_gt"),
                    "decoded_matches_next_gt": row.get("decoded_matches_next_gt"),
                    "best_legal_matches_gt": legal.get("best_legal_matches_gt"),
                    "best_legal_move_uci": legal.get("best_legal_move_uci"),
                    "gt_legal_rank": legal.get("gt_legal_rank"),
                    "image_path": row.get("image_path"),
                    "final_bucket": "",
                    "notes": "",
                }
            )


def _windowed_board_score(
    log_probs: list[torch.Tensor],
    *,
    board: chess.Board,
    frame_index: int,
    lookahead_window: int,
) -> float:
    end_index = min(len(log_probs), frame_index + lookahead_window)
    return float(
        sum(score_board_state(log_probs[index], board) for index in range(frame_index, end_index))
    )


def _class_ids_to_fen(class_ids: tuple[int, ...]) -> str:
    ranks: list[str] = []
    for row in range(8):
        empty_run = 0
        parts: list[str] = []
        for col in range(8):
            class_id = int(class_ids[row * 8 + col])
            class_name = SQUARE_CLASS_NAMES[class_id]
            if class_name == "empty":
                empty_run += 1
                continue
            if empty_run > 0:
                parts.append(str(empty_run))
                empty_run = 0
            parts.append(class_name)
        if empty_run > 0:
            parts.append(str(empty_run))
        ranks.append("".join(parts) or "8")
    return "/".join(ranks)


def _fen_to_class_ids(fen: str) -> list[int]:
    placement = fen.split(" ", 1)[0]
    class_name_to_index = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}
    class_ids: list[int] = []
    for rank in placement.split("/"):
        for char in rank:
            if char.isdigit():
                class_ids.extend([_EMPTY_CLASS_ID] * int(char))
            else:
                class_ids.append(class_name_to_index[char])
    if len(class_ids) != 64:
        raise ValueError(f"Expected 64 class ids from FEN, got {len(class_ids)}: {fen}")
    return class_ids


def _frame_index(row: ObservationRow) -> int:
    frame_index = getattr(row, "frame_index", None)
    if frame_index is None:
        raise ValueError(f"Row is missing frame_index: {row.annotation_id}")
    return int(frame_index)


def _class_confidences(
    probabilities: torch.Tensor,
    class_ids: tuple[int, ...],
) -> list[float]:
    indices = torch.tensor(class_ids, dtype=torch.long)
    values = probabilities.gather(1, indices.unsqueeze(1)).squeeze(1)
    return [round(float(value), 4) for value in values.tolist()]


def _count_differences(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(int(left_value != right_value) for left_value, right_value in zip(left, right))


def _difference_square_names(
    predicted_class_ids: tuple[int, ...],
    target_class_ids: tuple[int, ...],
) -> list[str]:
    return [
        _square_name(index)
        for index, (predicted, target) in enumerate(zip(predicted_class_ids, target_class_ids))
        if predicted != target
    ]


def _changed_mask(
    current: tuple[int, ...],
    previous: tuple[int, ...] | None,
) -> tuple[bool, ...]:
    if previous is None:
        return (False,) * len(current)
    return tuple(
        current_value != previous_value
        for current_value, previous_value in zip(current, previous)
    )


def _mask_to_square_names(mask: tuple[bool, ...]) -> list[str]:
    return [_square_name(index) for index, enabled in enumerate(mask) if enabled]


def _square_mask(square_names: Any) -> tuple[bool, ...]:
    names = set(square_names or [])
    return tuple(_square_name(index) in names for index in range(64))


def _square_name(square_index: int) -> str:
    file_name = chr(ord("a") + (square_index % 8))
    rank = 8 - square_index // 8
    return f"{file_name}{rank}"


def _flag(value: bool) -> str:
    return "y" if value else "n"


def _blend(
    base: tuple[int, int, int],
    tint: tuple[int, int, int],
    weight: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, weight))
    return tuple(
        int(round((1.0 - clamped) * base_value + clamped * tint_value))
        for base_value, tint_value in zip(base, tint)
    )


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _safe_filename_stem(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or "frame"


def _relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)
