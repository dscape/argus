"""DINOv2-based per-square piece classifier (Approach B).

Architecture: Frozen DINOv2-base → 768D pooled → MLP head → 13 classes.
Follows the ScreeningClassifier pattern from pipeline/screen/ai_classifier.py.

Classes: empty, P, N, B, R, Q, K, p, n, b, r, q, k  (indices 0–12).
"""

from __future__ import annotations

import logging
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
import torch.nn as nn

from pipeline.overlay.grid_detector import find_board_in_frame

logger = logging.getLogger(__name__)

NUM_CLASSES = 13
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

WEIGHTS_DIR = Path("weights/piece_classifier")


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
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_square(square_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR square crop to a normalised DINOv2 input tensor."""
    rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


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
    batch: list[torch.Tensor] = []
    for r in range(8):
        for c in range(8):
            batch.append(preprocess_square(squares[r][c]))

    input_tensor = torch.stack(batch).to(device)  # (64, 3, 224, 224)
    with torch.no_grad():
        logits = model(input_tensor)  # (64, 13)
    preds = logits.argmax(dim=1).cpu().tolist()

    result: list[list[int]] = []
    for r in range(8):
        result.append(preds[r * 8 : (r + 1) * 8])
    return result


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

    squares = grid.crop_squares(frame)
    class_grid = classify_squares(squares, device=device)

    flipped = _detect_orientation(class_grid)

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
