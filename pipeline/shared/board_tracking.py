"""Shared chess-aware board-state tracking over per-square logits."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import torch

from pipeline.shared.board_state import fen_to_square_labels

_MAX_MOVE_CHANGED_SQUARES = 4


@dataclass(frozen=True)
class BoardTrackerResult:
    """Result returned after one tracker update."""

    fen: str
    full_fen: str
    move_uci: str | None
    move_score: float
    stay_score: float
    turn_resolved: bool


@dataclass(frozen=True)
class SequenceTrackerFrameResult:
    """Decoded board state for one sequence frame."""

    fen: str
    full_fen: str
    move_uci: str | None
    move_score: float
    stay_score: float


class LegalMoveStateTracker:
    """Track a chess position by preferring legal moves over full re-reads.

    The tracker operates only on `(64, C)` per-square logits. It assumes the
    current board state is already known and only commits a new board state when
    a legal move improves the observation score by a configured margin.
    """

    def __init__(
        self,
        initial_board_fen: str,
        *,
        move_accept_threshold: float = 2.5,
        move_accept_margin: float = 0.75,
    ) -> None:
        self.initial_board_fen = initial_board_fen
        self.move_accept_threshold = float(move_accept_threshold)
        self.move_accept_margin = float(move_accept_margin)
        self._candidate_boards = build_board_hypotheses(initial_board_fen)

    @property
    def board(self) -> chess.Board:
        return self._candidate_boards[0]

    @property
    def turn_resolved(self) -> bool:
        return len(self._candidate_boards) == 1

    def reset(self) -> None:
        self._candidate_boards = build_board_hypotheses(self.initial_board_fen)

    def update(self, square_logits: torch.Tensor) -> BoardTrackerResult:
        log_probs = torch.log_softmax(_validated_board_logits(square_logits), dim=1)

        best_action: _ScoredAction | None = None
        best_hypothesis_index = 0
        second_best_score = float("-inf")
        best_stay_score = float("-inf")

        for hypothesis_index, board in enumerate(self._candidate_boards):
            stay_score = score_board_state(log_probs, board)
            best_stay_score = max(best_stay_score, stay_score)
            if best_action is None or stay_score > best_action.score:
                if best_action is not None:
                    second_best_score = max(second_best_score, best_action.score)
                best_action = _ScoredAction(board=board, score=stay_score, move=None, delta=0.0)
                best_hypothesis_index = hypothesis_index
            else:
                second_best_score = max(second_best_score, stay_score)

            for move in board.legal_moves:
                action = score_legal_move(log_probs, board, move, stay_score=stay_score)
                if best_action is None or action.score > best_action.score:
                    if best_action is not None:
                        second_best_score = max(second_best_score, best_action.score)
                    best_action = action
                    best_hypothesis_index = hypothesis_index
                else:
                    second_best_score = max(second_best_score, action.score)

        assert best_action is not None
        stay_score = score_board_state(log_probs, self._candidate_boards[best_hypothesis_index])
        move_score = best_action.score if best_action.move is not None else stay_score

        if (
            best_action.move is not None
            and best_action.delta >= self.move_accept_threshold
            and move_score - second_best_score >= self.move_accept_margin
        ):
            self._candidate_boards = [best_action.board]
            return BoardTrackerResult(
                fen=best_action.board.board_fen(),
                full_fen=best_action.board.fen(),
                move_uci=best_action.move.uci(),
                move_score=move_score,
                stay_score=stay_score,
                turn_resolved=True,
            )

        if self.turn_resolved:
            board = self._candidate_boards[0]
        else:
            board = self._candidate_boards[best_hypothesis_index]
        return BoardTrackerResult(
            fen=board.board_fen(),
            full_fen=board.fen(),
            move_uci=None,
            move_score=move_score,
            stay_score=stay_score,
            turn_resolved=self.turn_resolved,
        )


@dataclass(frozen=True)
class _ScoredAction:
    board: chess.Board
    score: float
    move: chess.Move | None
    delta: float = 0.0


class LookaheadLegalMoveStateTracker:
    """Decode a full board-logit sequence with a fixed lookahead window.

    Unlike `LegalMoveStateTracker`, this decoder is sequence-oriented rather
    than strictly frame-causal. At each frame it compares `stay` against all
    one-move legal successors using the current frame plus a small future
    window, which helps when the first post-move frame is noisy but the next
    frames clearly show the settled position.
    """

    def __init__(
        self,
        initial_board_fen: str,
        *,
        lookahead_window: int = 3,
        move_score_margin: float = 8.0,
    ) -> None:
        if lookahead_window <= 0:
            raise ValueError(f"lookahead_window must be > 0, got {lookahead_window}")
        self.initial_board_fen = initial_board_fen
        self.lookahead_window = int(lookahead_window)
        self.move_score_margin = float(move_score_margin)

    def decode(
        self,
        sequence_logits: list[torch.Tensor],
        *,
        change_mask: list[bool] | None = None,
    ) -> list[SequenceTrackerFrameResult]:
        if not sequence_logits:
            return []
        if change_mask is not None and len(change_mask) != len(sequence_logits):
            raise ValueError(
                "change_mask must match sequence length, got "
                f"{len(change_mask)} vs {len(sequence_logits)}"
            )

        log_probs = [
            torch.log_softmax(_validated_board_logits(logits), dim=1) for logits in sequence_logits
        ]
        candidate_boards = build_board_hypotheses(self.initial_board_fen)
        current_board = max(
            candidate_boards, key=lambda board: score_board_state(log_probs[0], board)
        )
        results: list[SequenceTrackerFrameResult] = []

        for frame_index, frame_log_probs in enumerate(log_probs):
            should_consider_moves = True if change_mask is None else bool(change_mask[frame_index])
            stay_score = _windowed_board_score(
                log_probs,
                board=current_board,
                frame_index=frame_index,
                lookahead_window=self.lookahead_window,
            )
            move_score = stay_score
            move_uci: str | None = None

            if should_consider_moves:
                best_move_score = stay_score
                best_move_board: chess.Board | None = None
                best_move_uci: str | None = None
                for move in current_board.legal_moves:
                    next_board = current_board.copy(stack=False)
                    next_board.push(move)
                    candidate_score = _windowed_board_score(
                        log_probs,
                        board=next_board,
                        frame_index=frame_index,
                        lookahead_window=self.lookahead_window,
                    )
                    if candidate_score > best_move_score:
                        best_move_score = candidate_score
                        best_move_board = next_board
                        best_move_uci = move.uci()
                if (
                    best_move_board is not None
                    and best_move_score - stay_score >= self.move_score_margin
                ):
                    current_board = best_move_board
                    move_score = best_move_score
                    move_uci = best_move_uci

            results.append(
                SequenceTrackerFrameResult(
                    fen=current_board.board_fen(),
                    full_fen=current_board.fen(),
                    move_uci=move_uci,
                    move_score=move_score,
                    stay_score=stay_score,
                )
            )
        return results


def build_board_hypotheses(initial_board_fen: str) -> list[chess.Board]:
    """Build candidate boards when only piece placement is known.

    Real clip metadata often stores only `board_fen()` piece placement. Before
    the first detected move we do not know whose turn it is, so the tracker keeps
    both White-to-move and Black-to-move hypotheses alive.
    """

    white_board = chess.Board()
    white_board.set_board_fen(initial_board_fen)
    white_board.turn = chess.WHITE

    black_board = white_board.copy(stack=False)
    black_board.turn = chess.BLACK
    return [white_board, black_board]


def board_to_class_ids(board: chess.Board) -> list[int]:
    return [value for row in fen_to_square_labels(board.board_fen()) for value in row]


def score_board_state(log_probs: torch.Tensor, board: chess.Board) -> float:
    class_ids = torch.tensor(board_to_class_ids(board), dtype=torch.long, device=log_probs.device)
    indices = torch.arange(64, device=log_probs.device)
    return float(log_probs[indices, class_ids].sum().item())


def score_legal_move(
    log_probs: torch.Tensor,
    board: chess.Board,
    move: chess.Move,
    *,
    stay_score: float,
) -> _ScoredAction:
    before_class_ids = board_to_class_ids(board)
    next_board = board.copy(stack=False)
    next_board.push(move)
    after_class_ids = board_to_class_ids(next_board)
    changed_indices = [
        index
        for index, (before_id, after_id) in enumerate(zip(before_class_ids, after_class_ids))
        if before_id != after_id
    ]
    if not 1 <= len(changed_indices) <= _MAX_MOVE_CHANGED_SQUARES:
        raise ValueError(
            f"Expected 1..{_MAX_MOVE_CHANGED_SQUARES} changed squares for {move.uci()}, "
            f"got {len(changed_indices)}"
        )

    changed_tensor = torch.tensor(changed_indices, dtype=torch.long, device=log_probs.device)
    before_tensor = torch.tensor(
        [before_class_ids[index] for index in changed_indices],
        dtype=torch.long,
        device=log_probs.device,
    )
    after_tensor = torch.tensor(
        [after_class_ids[index] for index in changed_indices],
        dtype=torch.long,
        device=log_probs.device,
    )
    delta = float(
        (log_probs[changed_tensor, after_tensor] - log_probs[changed_tensor, before_tensor])
        .sum()
        .item()
    )
    return _ScoredAction(board=next_board, score=stay_score + delta, move=move, delta=delta)


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


def _validated_board_logits(square_logits: torch.Tensor) -> torch.Tensor:
    if square_logits.ndim != 2 or square_logits.shape[0] != 64:
        raise ValueError(f"Expected board logits shaped (64, C), got {tuple(square_logits.shape)}")
    return square_logits.detach().clone()
