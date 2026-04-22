"""Heuristic auto-labeler for temporal move-execution events.

Per-move labels emitted:
    t_lift            — source square starts vacating
    t_capture         — destination (or captured) square disturbed before placement
    t_occ_start/end   — hand-occlusion window around the move
    t_place           — new piece visible on destination
    t_settle          — board logits / appearance stable after placement

Signals used (all descriptor-based, no ML forward pass):
    S_pre[t, sq]  — descriptor delta vs. a pre-move baseline frame
    dS[t, sq]     — frame-to-frame descriptor delta
    M[t]          — median per-square dS (robust hand-over-board motion)
    skin[t]       — fraction of frame pixels that pass an HSV skin filter

Outputs match the schema consumed by
``api/services/annotate/physical_eval_service.save_transient_annotation`` —
``start_frame_index`` = t_lift, ``end_frame_index`` = t_settle, and
hand_occlusion_spans = merged (t_occ_start, t_occ_end) windows across moves.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import torch

from pipeline.overlay.grid_detector import GridResult
from pipeline.overlay.sequence_reader import (
    descriptor_delta_scores,
    extract_square_descriptors,
)
from pipeline.physical.shared.annotation_dataset import rectify_board_image
from pipeline.physical.shared.board_localizer import (
    BoardLocalization,
    localize_board,
    track_corners as track_board_corners,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RECTIFIED_SIZE = 224
_LIFT_SEARCH_BACKWARD = 10
_CAPTURE_SEARCH_FORWARD = 4
_OCCLUSION_SEARCH_RADIUS = 12
_SETTLE_HORIZON = 8
_SETTLE_STABILITY_FRAMES = 2
_LIFT_DESCRIPTOR_THRESHOLD = 0.22
_LIFT_FRAME_DELTA_THRESHOLD = 0.12
_CAPTURE_DESCRIPTOR_THRESHOLD = 0.30
_PLACE_DESCRIPTOR_THRESHOLD = 0.30
_SETTLE_SQUARE_STABILITY = 0.18
_STABILITY_FRAME_DELTA_THRESHOLD = 0.08
_OCCLUSION_SKIN_BUMP = 0.03


@dataclass(frozen=True)
class AutoMoveLabel:
    move_index: int
    uci: str
    san: str | None
    move_frame_index: int
    side_to_move: str | None
    fen_before: str | None
    fen_after: str | None
    is_capture: bool | None
    start_frame_index: int | None
    end_frame_index: int | None
    t_lift: int | None
    t_capture: int | None
    t_place: int | None
    t_settle: int | None
    t_occ_start: int | None
    t_occ_end: int | None
    settle_quality: str

    def as_move_annotation(self) -> dict[str, Any]:
        return {
            "move_index": self.move_index,
            "uci": self.uci,
            "san": self.san,
            "move_frame_index": self.move_frame_index,
            "side_to_move": self.side_to_move,
            "fen_before": self.fen_before,
            "fen_after": self.fen_after,
            "is_capture": self.is_capture,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
        }


@dataclass(frozen=True)
class AutoLabelResult:
    clip_path: str
    frame_count: int
    corners_method: str
    corners_confidence: float
    move_labels: tuple[AutoMoveLabel, ...]
    hand_occlusion_spans: tuple[dict[str, int], ...]

    def to_transient_payload(self) -> dict[str, Any]:
        return {
            "move_annotations": [label.as_move_annotation() for label in self.move_labels],
            "hand_occlusion_spans": list(self.hand_occlusion_spans),
        }

    def to_detail_payload(self) -> dict[str, Any]:
        return {
            "clip_path": self.clip_path,
            "frame_count": self.frame_count,
            "corners_method": self.corners_method,
            "corners_confidence": self.corners_confidence,
            "move_labels": [asdict(label) for label in self.move_labels],
            "hand_occlusion_spans": list(self.hand_occlusion_spans),
        }


@dataclass(frozen=True)
class _MoveTarget:
    move_index: int
    uci: str
    san: str | None
    move_frame_index: int
    sampled_move_frame: int
    side_to_move: str | None
    fen_before: str | None
    fen_after: str | None
    affected_squares: tuple[int, ...]
    from_sq: int
    to_sq: int
    captured_sq: int | None
    is_capture: bool


def auto_label_clip(clip_path: str | Path) -> AutoLabelResult:
    resolved = _resolve_clip_path(clip_path)
    relative_clip_path = str(resolved.relative_to(_PROJECT_ROOT))
    clip = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(clip, dict):
        raise ValueError(f"Unsupported clip payload for {relative_clip_path}")

    frames_rgb = _load_frames_rgb(clip)
    frame_count = frames_rgb.shape[0]
    if frame_count == 0:
        raise ValueError(f"Clip has no frames: {relative_clip_path}")

    corners, method, confidence = _resolve_corners(frames_rgb)
    rectified = _rectify_all(frames_rgb, corners)
    descriptors = _compute_descriptors(rectified)
    targets = _enumerate_move_targets(clip)

    baseline_descriptors = descriptors[0].copy()
    skin = _skin_fraction_series(frames_rgb)
    global_motion = _frame_to_frame_motion(descriptors)

    motion_baseline = _median_absolute_deviation_threshold(global_motion)
    skin_baseline = float(np.percentile(skin, 10))

    move_labels: list[AutoMoveLabel] = []
    occlusion_windows: list[tuple[int, int]] = []
    previous_settle = 0
    for target in targets:
        label = _label_move(
            target=target,
            descriptors=descriptors,
            baseline=baseline_descriptors,
            global_motion=global_motion,
            motion_baseline=motion_baseline,
            skin=skin,
            skin_baseline=skin_baseline,
            frame_count=frame_count,
            earliest_allowed=previous_settle,
        )
        move_labels.append(label)
        if label.t_occ_start is not None and label.t_occ_end is not None:
            occlusion_windows.append((label.t_occ_start, label.t_occ_end))
        if label.t_settle is not None:
            previous_settle = max(previous_settle, label.t_settle)
        elif label.t_place is not None:
            previous_settle = max(previous_settle, label.t_place)

    merged_spans = _merge_spans(occlusion_windows)
    return AutoLabelResult(
        clip_path=relative_clip_path,
        frame_count=frame_count,
        corners_method=method,
        corners_confidence=confidence,
        move_labels=tuple(move_labels),
        hand_occlusion_spans=tuple(
            {"start_frame_index": int(start), "end_frame_index": int(end)}
            for start, end in merged_spans
        ),
    )


def _resolve_clip_path(clip_path: str | Path) -> Path:
    candidate = Path(clip_path)
    if not candidate.is_absolute():
        candidate = (_PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"Clip path outside project root: {clip_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    return candidate


def _load_frames_rgb(clip: dict[str, Any]) -> np.ndarray:
    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor):
        raise ValueError("Clip missing tensor 'frames'")
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    if frames.ndim == 4 and frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1).contiguous()
    elif frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frames shape: {tuple(frames.shape)}")
    return frames.numpy()


def _resolve_corners(frames_rgb: np.ndarray) -> tuple[np.ndarray, str, float]:
    first_bgr = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR)
    localization: BoardLocalization | None = localize_board(first_bgr)
    if localization is None:
        raise ValueError("Could not locate board on frame 0; annotate corners manually first")
    per_frame = np.zeros((frames_rgb.shape[0], 4, 2), dtype=np.float32)
    per_frame[0] = np.asarray(localization.corners, dtype=np.float32)
    prev_bgr = first_bgr
    prev_corners = localization.corners
    confidence_sum = float(localization.confidence)
    confidence_count = 1
    for index in range(1, frames_rgb.shape[0]):
        curr_bgr = cv2.cvtColor(frames_rgb[index], cv2.COLOR_RGB2BGR)
        tracked = track_board_corners(prev_bgr, curr_bgr, prev_corners)
        if tracked is None:
            fallback = localize_board(curr_bgr)
            if fallback is None:
                per_frame[index] = per_frame[index - 1]
            else:
                per_frame[index] = np.asarray(fallback.corners, dtype=np.float32)
                prev_corners = fallback.corners
                confidence_sum += float(fallback.confidence)
                confidence_count += 1
        else:
            per_frame[index] = np.asarray(tracked.corners, dtype=np.float32)
            prev_corners = tracked.corners
            confidence_sum += float(tracked.confidence)
            confidence_count += 1
        prev_bgr = curr_bgr
    mean_confidence = confidence_sum / max(confidence_count, 1)
    return per_frame, localization.method, mean_confidence


def _rectify_all(frames_rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rectified = np.empty(
        (frames_rgb.shape[0], _RECTIFIED_SIZE, _RECTIFIED_SIZE, 3),
        dtype=np.uint8,
    )
    for index in range(frames_rgb.shape[0]):
        rectified[index] = rectify_board_image(
            frames_rgb[index],
            corners[index].tolist(),
            output_size=_RECTIFIED_SIZE,
        )
    return rectified


def _synthetic_grid() -> GridResult:
    step = _RECTIFIED_SIZE // 8
    lines = [step * i for i in range(9)]
    return GridResult(v_lines=lines, h_lines=lines, sq_size=step)


def _compute_descriptors(rectified_rgb: np.ndarray) -> np.ndarray:
    grid = _synthetic_grid()
    descriptors = np.empty(
        (rectified_rgb.shape[0], 64, 12 * 12),
        dtype=np.float32,
    )
    for index in range(rectified_rgb.shape[0]):
        bgr = cv2.cvtColor(rectified_rgb[index], cv2.COLOR_RGB2BGR)
        descriptors[index] = extract_square_descriptors(bgr, grid)
    return descriptors


def _frame_to_frame_motion(descriptors: np.ndarray) -> np.ndarray:
    if descriptors.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    diff = np.abs(np.diff(descriptors, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], diff]).astype(np.float32)


def _median_absolute_deviation_threshold(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    median = float(np.median(signal))
    deviation = float(np.median(np.abs(signal - median))) or 1e-6
    return median + 2.0 * deviation


def _skin_fraction_series(frames_rgb: np.ndarray) -> np.ndarray:
    series = np.zeros((frames_rgb.shape[0],), dtype=np.float32)
    for index in range(frames_rgb.shape[0]):
        hsv = cv2.cvtColor(frames_rgb[index], cv2.COLOR_RGB2HSV)
        lower_a = np.array([0, 40, 60], dtype=np.uint8)
        upper_a = np.array([25, 180, 255], dtype=np.uint8)
        lower_b = np.array([160, 40, 60], dtype=np.uint8)
        upper_b = np.array([180, 180, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_a, upper_a),
            cv2.inRange(hsv, lower_b, upper_b),
        )
        series[index] = float(mask.sum()) / float(mask.size * 255)
    return series


def _enumerate_move_targets(clip: dict[str, Any]) -> list[_MoveTarget]:
    move_ucis = _coerce_list(clip.get("move_ucis"))
    move_sans = _coerce_list(clip.get("move_sans"))
    move_frame_indices = _coerce_int_list(clip.get("move_frame_indices"))
    frame_indices = _coerce_int_list(clip.get("frame_indices"))
    initial_fen = clip.get("initial_board_fen")
    initial_side_to_move = clip.get("initial_side_to_move")

    if not isinstance(initial_fen, str) or not move_ucis:
        return []

    board = _build_starting_board(initial_fen, initial_side_to_move)
    frame_index_array = np.asarray(frame_indices, dtype=np.int64) if frame_indices else None
    targets: list[_MoveTarget] = []
    for move_index, uci in enumerate(move_ucis):
        if not isinstance(uci, str) or len(uci) < 4:
            continue
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move not in board.legal_moves:
            legal = next(
                (legal for legal in board.legal_moves if legal.uci() == uci),
                None,
            )
            if legal is None:
                break
            move = legal

        san = board.san(move)
        fen_before = board.fen()
        side_to_move = "white" if board.turn == chess.WHITE else "black"
        is_capture = board.is_capture(move)
        captured_square = _captured_square(board, move) if is_capture else None
        board.push(move)
        fen_after = board.fen()

        if move_index < len(move_frame_indices):
            move_frame_index = int(move_frame_indices[move_index])
        else:
            move_frame_index = -1

        sampled = _map_to_sampled(move_frame_index, frame_index_array)
        san_text = (
            move_sans[move_index]
            if move_index < len(move_sans) and isinstance(move_sans[move_index], str)
            else san
        )
        affected = [
            _board_square_to_index(move.from_square),
            _board_square_to_index(move.to_square),
        ]
        if captured_square is not None:
            affected.append(_board_square_to_index(captured_square))
        if move.uci() in ("e1g1", "e1c1", "e8g8", "e8c8"):
            rook_from, rook_to = _castling_rook_squares(move)
            affected.extend(
                [
                    _board_square_to_index(rook_from),
                    _board_square_to_index(rook_to),
                ]
            )

        targets.append(
            _MoveTarget(
                move_index=move_index,
                uci=move.uci(),
                san=san_text,
                move_frame_index=move_frame_index,
                sampled_move_frame=sampled,
                side_to_move=side_to_move,
                fen_before=fen_before,
                fen_after=fen_after,
                affected_squares=tuple(dict.fromkeys(affected)),
                from_sq=_board_square_to_index(move.from_square),
                to_sq=_board_square_to_index(move.to_square),
                captured_sq=(
                    _board_square_to_index(captured_square)
                    if captured_square is not None
                    else None
                ),
                is_capture=is_capture,
            )
        )

    return targets


def _build_starting_board(
    initial_board_fen: str,
    initial_side_to_move: Any,
) -> chess.Board:
    side = "w" if initial_side_to_move == "white" else "b"
    if isinstance(initial_side_to_move, str) and initial_side_to_move not in ("white", "black"):
        side = initial_side_to_move[0]
    full_fen = f"{initial_board_fen} {side} - - 0 1"
    try:
        return chess.Board(full_fen)
    except ValueError:
        board = chess.Board()
        board.set_board_fen(initial_board_fen)
        board.turn = chess.WHITE if side == "w" else chess.BLACK
        return board


def _captured_square(board: chess.Board, move: chess.Move) -> int | None:
    if board.is_en_passant(move):
        direction = -8 if board.turn == chess.WHITE else 8
        return move.to_square + direction
    if board.piece_at(move.to_square) is not None:
        return move.to_square
    return None


def _castling_rook_squares(move: chess.Move) -> tuple[int, int]:
    if move.uci() == "e1g1":
        return chess.H1, chess.F1
    if move.uci() == "e1c1":
        return chess.A1, chess.D1
    if move.uci() == "e8g8":
        return chess.H8, chess.F8
    return chess.A8, chess.D8


def _board_square_to_index(square: int) -> int:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    return (7 - rank_index) * 8 + file_index


def _map_to_sampled(move_frame_index: int, frame_indices: np.ndarray | None) -> int:
    if move_frame_index < 0:
        return 0
    if frame_indices is None or frame_indices.size == 0:
        return move_frame_index
    idx = int(np.searchsorted(frame_indices, move_frame_index, side="left"))
    return int(max(0, min(idx, frame_indices.size - 1)))


def _label_move(
    *,
    target: _MoveTarget,
    descriptors: np.ndarray,
    baseline: np.ndarray,
    global_motion: np.ndarray,
    motion_baseline: float,
    skin: np.ndarray,
    skin_baseline: float,
    frame_count: int,
    earliest_allowed: int,
) -> AutoMoveLabel:
    anchor = int(np.clip(target.sampled_move_frame, 0, frame_count - 1))
    per_square_delta = np.mean(np.abs(descriptors - baseline), axis=-1)
    per_square_frame_delta = np.zeros_like(per_square_delta)
    if per_square_delta.shape[0] > 1:
        per_square_frame_delta[1:] = np.abs(np.diff(per_square_delta, axis=0))

    affected = list(target.affected_squares)
    t_lift = _detect_lift_frame(
        per_square_delta=per_square_delta,
        per_square_frame_delta=per_square_frame_delta,
        square=target.from_sq,
        anchor=anchor,
        frame_count=frame_count,
        earliest_allowed=earliest_allowed,
    )
    t_capture = (
        _detect_capture_frame(
            per_square_delta=per_square_delta,
            square=target.captured_sq if target.captured_sq is not None else target.to_sq,
            start=t_lift if t_lift is not None else max(earliest_allowed, anchor - _LIFT_SEARCH_BACKWARD),
            anchor=anchor,
            frame_count=frame_count,
        )
        if target.is_capture
        else None
    )
    t_occ_start, t_occ_end = _detect_occlusion_span(
        global_motion=global_motion,
        motion_baseline=motion_baseline,
        skin=skin,
        skin_baseline=skin_baseline,
        anchor=anchor,
        frame_count=frame_count,
        earliest_allowed=earliest_allowed,
    )
    search_place_start = t_occ_end if t_occ_end is not None else anchor
    t_place = _detect_place_frame(
        per_square_delta=per_square_delta,
        square=target.to_sq,
        start=search_place_start,
        frame_count=frame_count,
    )
    t_settle, settle_quality = _detect_settle_frame(
        per_square_frame_delta=per_square_frame_delta,
        squares=affected,
        start=t_place if t_place is not None else search_place_start,
        frame_count=frame_count,
    )

    start_frame_index = t_lift if t_lift is not None else t_occ_start
    end_frame_index = t_settle if t_settle is not None else t_place
    if (
        start_frame_index is not None
        and end_frame_index is not None
        and end_frame_index < start_frame_index
    ):
        end_frame_index = start_frame_index

    return AutoMoveLabel(
        move_index=target.move_index,
        uci=target.uci,
        san=target.san,
        move_frame_index=target.move_frame_index,
        side_to_move=target.side_to_move,
        fen_before=target.fen_before,
        fen_after=target.fen_after,
        is_capture=target.is_capture,
        start_frame_index=_maybe_int(start_frame_index),
        end_frame_index=_maybe_int(end_frame_index),
        t_lift=_maybe_int(t_lift),
        t_capture=_maybe_int(t_capture),
        t_place=_maybe_int(t_place),
        t_settle=_maybe_int(t_settle),
        t_occ_start=_maybe_int(t_occ_start),
        t_occ_end=_maybe_int(t_occ_end),
        settle_quality=settle_quality,
    )


def _detect_lift_frame(
    *,
    per_square_delta: np.ndarray,
    per_square_frame_delta: np.ndarray,
    square: int,
    anchor: int,
    frame_count: int,
    earliest_allowed: int,
) -> int | None:
    lower = max(earliest_allowed, anchor - _LIFT_SEARCH_BACKWARD)
    if lower > anchor:
        return None
    last_stable = anchor
    for t in range(anchor, lower - 1, -1):
        if per_square_frame_delta[t, square] < _LIFT_FRAME_DELTA_THRESHOLD:
            last_stable = t
        else:
            if last_stable > t:
                candidate = t + 1
                if per_square_delta[candidate, square] >= _LIFT_DESCRIPTOR_THRESHOLD:
                    return candidate
                last_stable = t
    return None


def _detect_capture_frame(
    *,
    per_square_delta: np.ndarray,
    square: int,
    start: int,
    anchor: int,
    frame_count: int,
) -> int | None:
    upper = min(anchor + _CAPTURE_SEARCH_FORWARD, frame_count)
    for t in range(max(0, start), upper):
        if per_square_delta[t, square] >= _CAPTURE_DESCRIPTOR_THRESHOLD:
            return t
    return None


def _detect_occlusion_span(
    *,
    global_motion: np.ndarray,
    motion_baseline: float,
    skin: np.ndarray,
    skin_baseline: float,
    anchor: int,
    frame_count: int,
    earliest_allowed: int,
) -> tuple[int | None, int | None]:
    lower = max(earliest_allowed, anchor - _OCCLUSION_SEARCH_RADIUS)
    upper = min(frame_count, anchor + _OCCLUSION_SEARCH_RADIUS + 1)
    if upper <= lower:
        return None, None
    mask = np.zeros(frame_count, dtype=bool)
    mask[lower:upper] = (
        (global_motion[lower:upper] >= motion_baseline)
        | (skin[lower:upper] >= skin_baseline + _OCCLUSION_SKIN_BUMP)
    )
    if not mask.any():
        return None, None
    start = anchor
    while start > lower and mask[start - 1]:
        start -= 1
    end = anchor
    while end + 1 < upper and mask[end + 1]:
        end += 1
    if not mask[anchor] and start == anchor and end == anchor:
        candidates = np.flatnonzero(mask[lower:upper])
        if candidates.size == 0:
            return None, None
        nearest = lower + int(candidates[np.argmin(np.abs(candidates + lower - anchor))])
        start = end = nearest
        while start > lower and mask[start - 1]:
            start -= 1
        while end + 1 < upper and mask[end + 1]:
            end += 1
    return int(start), int(end)


def _detect_place_frame(
    *,
    per_square_delta: np.ndarray,
    square: int,
    start: int,
    frame_count: int,
) -> int | None:
    for t in range(max(0, start), frame_count):
        if per_square_delta[t, square] >= _PLACE_DESCRIPTOR_THRESHOLD:
            return t
    return None


def _detect_settle_frame(
    *,
    per_square_frame_delta: np.ndarray,
    squares: list[int],
    start: int,
    frame_count: int,
) -> tuple[int | None, str]:
    if not squares or frame_count == 0:
        return None, "none"
    upper = min(frame_count, start + _SETTLE_HORIZON)
    for t in range(max(0, start), upper):
        window_end = min(frame_count, t + _SETTLE_STABILITY_FRAMES)
        if window_end - t < _SETTLE_STABILITY_FRAMES:
            break
        window = per_square_frame_delta[t:window_end, squares]
        if float(window.max()) < _SETTLE_SQUARE_STABILITY:
            return t, "strong"
    relaxed = _SETTLE_SQUARE_STABILITY * 2.0
    for t in range(max(0, start), upper):
        window_end = min(frame_count, t + _SETTLE_STABILITY_FRAMES)
        if window_end - t < _SETTLE_STABILITY_FRAMES:
            break
        window = per_square_frame_delta[t:window_end, squares]
        if float(window.max()) < relaxed:
            return t, "weak"
    return (upper - 1 if upper > 0 else None), "weak"


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda pair: (pair[0], pair[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _coerce_int_list(value: Any) -> list[int]:
    return [int(item) for item in _coerce_list(value) if isinstance(item, (int, float))]


def _maybe_int(value: int | None) -> int | None:
    return None if value is None else int(value)
