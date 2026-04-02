"""DINOv2-based per-square piece classifier for 2D overlay boards.

Architecture: Frozen DINOv2-base → 768D pooled → MLP head → 13 classes.
Follows the ScreeningClassifier pattern from pipeline/screen/ai_classifier.py.

Classes: empty, P, N, B, R, Q, K, p, n, b, r, q, k  (indices 0–12).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
import torch.nn as nn

from pipeline.overlay.grid_detector import find_board_in_frame

logger = logging.getLogger(__name__)

# Bump this when model architecture or feature extraction changes. Format: v{N}
# v1: DINOv2-base frozen → MLP(768→256→13), trained on synthetic SVG+3D pieces
MODEL_CODE_VERSION = "v1"

NUM_CLASSES = 13
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pre-computed normalisation tensors (avoid re-creating on every call)
_NORM_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PieceClassifier(nn.Module):
    """DINOv2 → MLP piece classifier."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        from argus.model.vision_encoder import VisionEncoder

        self.encoder = VisionEncoder(frozen=True, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → logits (B, 13)."""
        with torch.no_grad():
            features = self.encoder.forward_pooled(x)  # (B, 768)
        return self.head(features)


# ---------------------------------------------------------------------------
# Preprocessing — batched for performance
# ---------------------------------------------------------------------------


def preprocess_squares_batch(squares: list[list[np.ndarray]]) -> torch.Tensor:
    """Convert an 8×8 grid of BGR square crops to a (64, 3, 224, 224) tensor.

    Batches all conversions and applies normalisation in one vectorised pass.
    """
    resized: list[np.ndarray] = []
    for r in range(8):
        for c in range(8):
            rgb = cv2.cvtColor(squares[r][c], cv2.COLOR_BGR2RGB)
            resized.append(cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)))

    # (64, 224, 224, 3) uint8 → (64, 3, 224, 224) float32
    arr = np.stack(resized)  # (64, H, W, 3)
    tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    tensor = (tensor - _NORM_MEAN) / _NORM_STD
    return tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_cached_model: PieceClassifier | None = None
_cached_device: str | None = None


def _get_model(device: str = "cpu") -> PieceClassifier:
    global _cached_model, _cached_device
    if _cached_model is not None and _cached_device == device:
        return _cached_model

    model = PieceClassifier()
    weights_path = WEIGHTS_DIR / "best.pt"
    if weights_path.exists():
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.head.load_state_dict(state["head"])
        # Log version if metadata exists
        meta_path = WEIGHTS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            logger.info(
                "Loaded piece classifier %s (val_acc=%.4f) from %s",
                meta.get("version", "?"),
                meta.get("best_val_accuracy", 0),
                weights_path,
            )
        else:
            logger.info("Loaded piece classifier weights from %s", weights_path)
    else:
        logger.warning("No piece classifier weights at %s — using random head", weights_path)

    model = model.to(device)
    model.eval()
    _cached_model = model
    _cached_device = device
    return model


def classify_squares(
    squares: list[list[np.ndarray]],
    device: str = "cpu",
) -> list[list[int]]:
    """Classify all 64 squares. Returns 8×8 grid of class indices (0–12)."""
    model = _get_model(device)
    input_tensor = preprocess_squares_batch(squares).to(device)

    with torch.no_grad():
        logits = model(input_tensor)  # (64, 13)
    preds = logits.argmax(dim=1).cpu().tolist()

    return [preds[r * 8 : (r + 1) * 8] for r in range(8)]


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------


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
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN:
                score -= 5
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None:
                continue
            rank = chess.square_rank(sq)
            if p.color == chess.WHITE and p.piece_type != chess.PAWN and rank <= 1:
                score += 1
            if p.color == chess.BLACK and p.piece_type != chess.PAWN and rank >= 6:
                score += 1
        return score

    return _score(True) > _score(False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_fen_with_grid(
    frame: np.ndarray,
    grid: "GridResult",
    device: str = "cpu",
    detect_orientation: bool = True,
) -> str:
    """Classify pieces given a known grid.  Returns piece-placement FEN.

    Unlike :func:`read_fen_from_frame` this never returns ``None`` — the
    caller provides a pre-validated grid.

    Set *detect_orientation* to ``False`` when the board orientation is
    known (e.g. chess-positions boards are always white-at-bottom).
    """
    squares = grid.crop_squares(frame)
    class_grid = classify_squares(squares, device=device)

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

    return board.board_fen()


def read_fen_from_frame(
    frame: np.ndarray,
    device: str = "cpu",
) -> str | None:
    """Read chess position from a video frame using the CNN classifier.

    Returns piece-placement FEN or None.
    """
    grid = find_board_in_frame(frame)
    if grid is None:
        logger.warning("No grid found in frame")
        return None

    return read_fen_with_grid(frame, grid, device=device)
