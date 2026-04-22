"""Temporal board tracking: re-score the previous state plus one-step legal continuations."""

from __future__ import annotations

import chess
import numpy as np
import torch

from pipeline.shared.board_constraints import constrained_board_class_ids

from .state import BoardState, board_from_class_ids, score_board
from .solver import infer_board

# Logit-domain bonus when the best hypothesis equals the last internal board.
NO_MOVE_BONUS: float = 2.0
# Soft bias: full re-inference from per-square logits (recovery) is disfavored so that
# legal continuations and temporal continuity win when scores are close; the belief is
# that the model is noisier than a single argmax+constraint snapshot.
RECOVERY_PENALTY: float = 1.0
# Per-square argmax+constraint can jump to an inconsistent position; we still score it
# (for escape hatches) but penalize so one-step legal moves and recovery are preferred
# when evidence is similar.
ILLEGAL_JUMP_PENALTY: float = 2.0
# If mean absolute per-element logit change from the previous frame is below this, we
# treat the frame as visually stable and skip move / recovery / jump expansion
# (empirically; tune to your camera and frame rate).
DELTA_THRESHOLD: float = 0.05


def compute_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two (64, 13) logit arrays (L1 on the flattened step)."""
    return float(np.mean(np.abs(a - b)))


def _board_identity(board: chess.Board) -> tuple[str, bool]:
    return board.board_fen(), bool(board.turn)


def track_boards(
    logits_seq: list[np.ndarray],
    *,
    no_move_bonus: float = NO_MOVE_BONUS,
    k: int = 3,
    num_samples: int = 200,
    seed: int = 0,
) -> list[BoardState]:
    """Run :class:`Tracker` over a sequence, returning a :class:`BoardState` per step."""
    if not logits_seq:
        return []
    tracker = Tracker(
        no_move_bonus=no_move_bonus,
        k=k,
        num_samples=num_samples,
        seed=seed,
    )
    return [tracker.update(frm) for frm in logits_seq]


class Tracker:
    def __init__(
        self,
        *,
        no_move_bonus: float = NO_MOVE_BONUS,
        k: int = 3,
        num_samples: int = 200,
        seed: int = 0,
    ) -> None:
        self.current_board: chess.Board | None = None
        self._last_identity: tuple[str, bool] | None = None
        self.no_move_bonus = float(no_move_bonus)
        self.k = k
        self.num_samples = num_samples
        self._seed = seed
        self._frame = 0
        self.prev_logits: np.ndarray | None = None

    def update(self, logits: np.ndarray) -> BoardState:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (64, 13):
            raise ValueError(f"Expected logits shaped (64, 13), got {logits.shape}")
        if self.current_board is None:
            b = infer_board(
                logits,
                k=self.k,
                num_samples=self.num_samples,
                seed=self._seed + self._frame,
            )
            self._frame += 1
            self.current_board = b.board
            self._last_identity = _board_identity(self.current_board)
            self.prev_logits = logits.copy()
            return b
        if self.prev_logits is not None:
            # delta: how much per-square logits changed vs. last frame (0 = identical).
            delta = compute_delta(self.prev_logits, logits)
        else:
            delta = 1.0
        fresh_board: chess.Board | None = None
        jump_board: chess.Board | None = None
        if delta < DELTA_THRESHOLD:
            cands = [self.current_board]
        else:
            cands = [self.current_board]
            for mv in self.current_board.legal_moves:
                b = self.current_board.copy(stack=False)
                b.push(mv)
                cands.append(b)
            # Recovery: a full global re-inference from this frame’s logits, for drift
            # correction when the track disagrees with strong new evidence.
            fresh = infer_board(
                logits,
                k=self.k,
                num_samples=self.num_samples,
                seed=self._seed + self._frame,
            )
            fresh_board = fresh.board
            cands.append(fresh_board)
            # Illegal jump: the fast per-square constrained argmax; may be impossible to
            # reach in one legal move, but is a strong single-frame prior.
            class_ids = constrained_board_class_ids(torch.tensor(logits, dtype=torch.float32))
            jump_board = board_from_class_ids(
                [int(x) for x in class_ids.tolist()],
                turn=self.current_board.turn,
            )
            cands.append(jump_board)
        best: chess.Board | None = None
        best_s = float("-inf")
        for b in cands:
            s = score_board(b, logits)
            if fresh_board is not None and _board_identity(b) == _board_identity(fresh_board):
                s -= RECOVERY_PENALTY
            if jump_board is not None and _board_identity(b) == _board_identity(jump_board):
                s -= ILLEGAL_JUMP_PENALTY
            if self._last_identity is not None and _board_identity(b) == self._last_identity:
                s += self.no_move_bonus
            if s > best_s or best is None:
                best_s = s
                best = b
        assert best is not None
        self._frame += 1
        self.current_board = best
        self._last_identity = _board_identity(self.current_board)
        self.prev_logits = logits.copy()
        return BoardState(self.current_board, best_s)

    def reset(self) -> None:
        self.current_board = None
        self._last_identity = None
        self._frame = 0
        self.prev_logits = None

    def from_seed_board(self, board: chess.Board) -> None:
        """Set internal state to a specific position (e.g. the standard start)."""
        self.current_board = board.copy(stack=False)
        self._last_identity = _board_identity(self.current_board)
        self._frame = 0
        self.prev_logits = None
