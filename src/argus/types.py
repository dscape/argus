"""Core data types for Argus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch


@dataclass
class BoardDetection:
    """A detected board in a single frame."""

    board_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0,1]
    corners: np.ndarray  # (4, 2) corner points for perspective transform
    confidence: float


@dataclass
class FrameObservation:
    """Everything the model sees at one timestep."""

    frame_idx: int
    timestamp_sec: float
    image: torch.Tensor  # Full frame (C, H, W)
    board_crops: list[torch.Tensor]  # Per-board crops (C, h, w)
    board_ids: list[int]


@dataclass
class MoveEvent:
    """A single predicted move for a single game."""

    board_id: int
    move_uci: str  # e.g. "e2e4"
    fen_before: str
    fen_after: str
    confidence: float
    frame_idx: int
    is_legal: bool = True


@dataclass
class GameTrack:
    """Full tracked game across the video."""

    board_id: int
    moves: list[MoveEvent] = field(default_factory=list)
    pgn: str = ""
    initial_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    final_fen: str = ""
    first_frame: int = 0
    last_frame: int = 0
    status: Literal["in_progress", "completed", "lost_track"] = "in_progress"


@dataclass
class TrainingClip:
    """A clip of N consecutive frames with ground truth."""

    frames: torch.Tensor  # (T, C, H, W)
    board_bboxes: list[list[BoardDetection]]  # [T][num_boards]
    board_fens: dict[int, list[str]]  # board_id -> FEN at each frame
    move_events: list[MoveEvent]
    num_boards: int


@dataclass
class ModelOutput:
    """Output from the full Argus model."""

    # Move prediction
    move_logits: torch.Tensor  # (B, T, num_boards, vocab_size)
    move_probs: torch.Tensor  # (B, T, num_boards, vocab_size) after masking + softmax

    # Move detection (did a move occur?)
    detect_logits: torch.Tensor  # (B, T, num_boards)

    # Board detection (Phase 2+)
    board_bboxes: torch.Tensor | None = None  # (B, T, num_queries, 4)
    board_confidence: torch.Tensor | None = None  # (B, T, num_queries)
    board_identity: torch.Tensor | None = None  # (B, T, num_queries, identity_dim)
