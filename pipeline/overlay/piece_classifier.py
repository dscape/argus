"""Tiny ONNX CNN per-square piece classifier for 2D overlay boards.

Runtime architecture: fixed 8×8 square crops → tiny CNN → 13 classes.
The committed runtime artifact is an ONNX model under ``weights/overlay/``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chess
import cv2
import numpy as np

from pipeline.overlay.grid_detector import GridResult, find_board_in_frame
from pipeline.overlay.square_classifier_model import INPUT_SIZE, MODEL_CODE_VERSION, NUM_CLASSES
from pipeline.runtime_assets import OVERLAY_PIECE_CLASSIFIER_WEIGHTS, ensure_runtime_asset

logger = logging.getLogger(__name__)

CLASS_TO_PIECE: dict[int, chess.Piece | None] = {
    0: None,
    1: chess.Piece(chess.PAWN, chess.WHITE),
    2: chess.Piece(chess.KNIGHT, chess.WHITE),
    3: chess.Piece(chess.BISHOP, chess.WHITE),
    4: chess.Piece(chess.ROOK, chess.WHITE),
    5: chess.Piece(chess.QUEEN, chess.WHITE),
    6: chess.Piece(chess.KING, chess.WHITE),
    7: chess.Piece(chess.PAWN, chess.BLACK),
    8: chess.Piece(chess.KNIGHT, chess.BLACK),
    9: chess.Piece(chess.BISHOP, chess.BLACK),
    10: chess.Piece(chess.ROOK, chess.BLACK),
    11: chess.Piece(chess.QUEEN, chess.BLACK),
    12: chess.Piece(chess.KING, chess.BLACK),
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "overlay"

_cached_session: Any | None = None
_cached_input_name: str | None = None
_logged_non_cpu_devices: set[str] = set()
_NON_EMPTY_MARGIN_THRESHOLD = 0.31


@dataclass(frozen=True)
class BoardRead:
    """Classified board state for a known grid."""

    fen: str
    class_grid: list[list[int]]
    flipped: bool


def preprocess_square_crops(square_crops: Sequence[np.ndarray]) -> np.ndarray:
    """Convert BGR square crops to a normalized ``(N, 3, S, S)`` float32 array."""
    if not square_crops:
        return np.empty((0, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    resized: list[np.ndarray] = []
    for crop in square_crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= INPUT_SIZE else cv2.INTER_LINEAR
        resized_img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=interpolation)
        resized.append(resized_img)

    arr = np.stack(resized).astype(np.float32) / 255.0
    return np.transpose(arr, (0, 3, 1, 2))


def classify_square_crops(
    square_crops: Sequence[np.ndarray],
    device: str = "cpu",
) -> list[int]:
    """Classify a batch of square crops and return class ids."""
    del device
    logits = _predict_square_logits(square_crops)
    return np.argmax(logits, axis=1).astype(np.int64).tolist()


def classify_squares(
    squares: list[list[np.ndarray]],
    device: str = "cpu",
) -> list[list[int]]:
    """Classify all 64 squares. Returns 8×8 grid of class indices (0–12)."""
    del device
    logits = _predict_square_logits([squares[r][c] for r in range(8) for c in range(8)])
    class_ids = np.argmax(logits, axis=1).astype(np.int64)
    class_ids = _suppress_ambiguous_non_empty_predictions(class_ids, logits)
    class_ids = _repair_king_counts(class_ids, logits)
    return [class_ids[r * 8 : (r + 1) * 8].tolist() for r in range(8)]


def _detect_orientation(class_grid: list[list[int]]) -> bool:
    """Return True if board is flipped (black at bottom)."""

    def _score(flipped: bool) -> float:
        board = chess.Board(fen=None)
        for r in range(8):
            for c in range(8):
                piece = CLASS_TO_PIECE.get(class_grid[r][c])
                if piece is None:
                    continue
                if not flipped:
                    sq = chess.square(c, 7 - r)
                else:
                    sq = chess.square(7 - c, r)
                board.set_piece_at(sq, piece)

        score = 0.0
        wk = len(board.pieces(chess.KING, chess.WHITE))
        bk = len(board.pieces(chess.KING, chess.BLACK))
        if wk == 1:
            score += 10
        if bk == 1:
            score += 10
        for sq in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                score -= 5
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            rank = chess.square_rank(sq)
            if piece.color == chess.WHITE and piece.piece_type != chess.PAWN and rank <= 1:
                score += 1
            if piece.color == chess.BLACK and piece.piece_type != chess.PAWN and rank >= 6:
                score += 1
        return score

    return _score(True) > _score(False)


def class_grid_to_fen(
    class_grid: list[list[int]],
    *,
    flipped: bool | None = None,
    detect_orientation: bool = True,
) -> tuple[str, bool]:
    """Convert a classified 8×8 grid into piece-placement FEN."""
    if flipped is None:
        flipped = _detect_orientation(class_grid) if detect_orientation else False

    board = chess.Board(fen=None)
    for r in range(8):
        for c in range(8):
            piece = CLASS_TO_PIECE.get(class_grid[r][c])
            if piece is None:
                continue
            if not flipped:
                sq = chess.square(c, 7 - r)
            else:
                sq = chess.square(7 - c, r)
            board.set_piece_at(sq, piece)

    return board.board_fen(), flipped


def read_board_with_grid(
    frame: np.ndarray,
    grid: GridResult,
    device: str = "cpu",
    detect_orientation: bool = True,
) -> BoardRead:
    """Classify pieces given a known grid."""
    _warn_if_non_cpu(device)
    squares = grid.crop_squares(frame)
    flat_squares = [squares[r][c] for r in range(8) for c in range(8)]
    logits = _predict_square_logits(flat_squares)
    class_ids = np.argmax(logits, axis=1).astype(np.int64)
    class_ids = _suppress_ambiguous_non_empty_predictions(class_ids, logits)
    class_ids = _repair_king_counts(class_ids, logits)
    class_grid = [class_ids[r * 8 : (r + 1) * 8].tolist() for r in range(8)]
    fen, flipped = class_grid_to_fen(
        class_grid,
        detect_orientation=detect_orientation,
    )
    return BoardRead(fen=fen, class_grid=class_grid, flipped=flipped)


def read_fen_with_grid(
    frame: np.ndarray,
    grid: GridResult,
    device: str = "cpu",
    detect_orientation: bool = True,
) -> str:
    """Classify pieces given a known grid. Returns piece-placement FEN."""
    return read_board_with_grid(
        frame,
        grid,
        device=device,
        detect_orientation=detect_orientation,
    ).fen


def read_fen_from_frame(
    frame: np.ndarray,
    device: str = "cpu",
) -> str | None:
    """Read chess position from a video frame using the ONNX square classifier."""
    _warn_if_non_cpu(device)
    grid = find_board_in_frame(frame)
    if grid is None:
        logger.warning("No grid found in frame")
        return None

    return read_fen_with_grid(frame, grid, device=device)


def _predict_square_logits(square_crops: Sequence[np.ndarray]) -> np.ndarray:
    if not square_crops:
        return np.empty((0, NUM_CLASSES), dtype=np.float32)

    session, input_name = _get_session()
    inputs = preprocess_square_crops(square_crops)
    return session.run(None, {input_name: inputs})[0]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _suppress_ambiguous_non_empty_predictions(
    class_ids: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    probs = _softmax(logits)
    refined = class_ids.copy()
    for index, class_id in enumerate(refined.tolist()):
        if class_id == 0:
            continue
        non_empty_margin = float(probs[index, class_id] - probs[index, 0])
        if non_empty_margin < _NON_EMPTY_MARGIN_THRESHOLD:
            refined[index] = 0
    return refined


def _repair_king_counts(class_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
    probs = _softmax(logits)
    refined = class_ids.copy()
    for king_class in (6, 12):
        indices = np.flatnonzero(refined == king_class).tolist()
        if len(indices) <= 1:
            continue
        best_index = max(indices, key=lambda idx: float(probs[idx, king_class] - probs[idx, 0]))
        for idx in indices:
            if idx != best_index:
                refined[idx] = 0
    return refined


def _warn_if_non_cpu(device: str) -> None:
    if device == "cpu" or device in _logged_non_cpu_devices:
        return
    _logged_non_cpu_devices.add(device)
    logger.info("Overlay piece classifier ignores device=%s and runs on CPU ONNX runtime", device)


def _get_session() -> tuple[Any, str]:
    global _cached_session, _cached_input_name
    if _cached_session is not None and _cached_input_name is not None:
        return _cached_session, _cached_input_name

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency failure is surfaced clearly
        raise RuntimeError(
            "onnxruntime is required for the overlay piece classifier. "
            "Install project dependencies again to fetch the new runtime."
        ) from exc

    model_path = ensure_runtime_asset(OVERLAY_PIECE_CLASSIFIER_WEIGHTS)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = min(8, max(1, os.cpu_count() or 1))

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    meta_path = WEIGHTS_DIR / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as handle:
            meta = json.load(handle)
        logger.info(
            "Loaded overlay piece classifier %s (val_acc=%.4f) from %s",
            meta.get("version", MODEL_CODE_VERSION),
            meta.get("best_val_accuracy", 0.0),
            model_path,
        )
    else:
        logger.info("Loaded overlay piece classifier from %s", model_path)

    _cached_session = session
    _cached_input_name = input_name
    return session, input_name
