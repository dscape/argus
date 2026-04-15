"""Shared chess-aware board-state tracking over per-square logits."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
import torch.nn.functional as F

from argus.chess.constraint_mask import apply_constraint_mask, get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, UNKNOWN_IDX, get_vocabulary
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


@dataclass(frozen=True)
class SequenceBeamSearchResult:
    """Best board-state trajectory returned by the sequence beam decoder."""

    frames: tuple[SequenceTrackerFrameResult, ...]
    total_score: float


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
        initial_side_to_move: str | None = None,
        move_accept_threshold: float = 2.5,
        move_accept_margin: float = 0.75,
    ) -> None:
        self.initial_board_fen = initial_board_fen
        self.initial_side_to_move = initial_side_to_move
        self.move_accept_threshold = float(move_accept_threshold)
        self.move_accept_margin = float(move_accept_margin)
        self._candidate_boards = build_board_hypotheses(
            initial_board_fen,
            initial_side_to_move=initial_side_to_move,
        )

    @property
    def board(self) -> chess.Board:
        return self._candidate_boards[0]

    @property
    def turn_resolved(self) -> bool:
        return len(self._candidate_boards) == 1

    def reset(self) -> None:
        self._candidate_boards = build_board_hypotheses(
            self.initial_board_fen,
            initial_side_to_move=self.initial_side_to_move,
        )

    def update(self, square_logits: torch.Tensor) -> BoardTrackerResult:
        log_probs = torch.log_softmax(_validated_board_logits(square_logits), dim=1)

        best_action: _ScoredAction | None = None
        best_hypothesis_index = 0
        second_best_score = float("-inf")

        for hypothesis_index, board in enumerate(self._candidate_boards):
            stay_score = score_board_state(log_probs, board)
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


@dataclass(frozen=True)
class _BeamState:
    board: chess.Board
    score: float
    search_score: float
    frames: tuple[SequenceTrackerFrameResult, ...]


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
        initial_side_to_move: str | None = None,
        lookahead_window: int = 3,
        move_score_margin: float = 8.0,
    ) -> None:
        if lookahead_window <= 0:
            raise ValueError(f"lookahead_window must be > 0, got {lookahead_window}")
        self.initial_board_fen = initial_board_fen
        self.initial_side_to_move = initial_side_to_move
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
        candidate_boards = build_board_hypotheses(
            self.initial_board_fen,
            initial_side_to_move=self.initial_side_to_move,
        )
        current_board = max(
            candidate_boards, key=lambda board: score_board_state(log_probs[0], board)
        )
        results: list[SequenceTrackerFrameResult] = []

        for frame_index, _frame_log_probs in enumerate(log_probs):
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


class LegalSequenceBeamDecoder:
    """Beam-search decoder over legal board-state trajectories.

    This decoder scores each frame with both board-state evidence (`64 x C`
    square logits) and optional move evidence (`move_logits`, `detect_logits`).
    It is the shared contract for the next physical/video-native reader: the
    model proposes square state plus move evidence, and the decoder searches
    only over legal trajectories.
    """

    def __init__(
        self,
        initial_board_fen: str,
        *,
        initial_side_to_move: str | None = None,
        beam_size: int = 8,
        top_move_candidates: int = 16,
        top_board_candidates: int = 0,
        board_weight: float = 1.0,
        move_weight: float = 1.0,
        detect_weight: float = 1.0,
        move_score_margin: float = 0.0,
        lookahead_window: int = 1,
        lookahead_weight: float = 0.0,
        lookahead_decay: float = 1.0,
    ) -> None:
        if beam_size <= 0:
            raise ValueError(f"beam_size must be > 0, got {beam_size}")
        if top_move_candidates <= 0:
            raise ValueError(f"top_move_candidates must be > 0, got {top_move_candidates}")
        if top_board_candidates < 0:
            raise ValueError(f"top_board_candidates must be >= 0, got {top_board_candidates}")
        if lookahead_window <= 0:
            raise ValueError(f"lookahead_window must be > 0, got {lookahead_window}")
        if lookahead_decay <= 0.0 or lookahead_decay > 1.0:
            raise ValueError(f"lookahead_decay must be in (0, 1], got {lookahead_decay}")
        self.initial_board_fen = initial_board_fen
        self.initial_side_to_move = initial_side_to_move
        self.beam_size = int(beam_size)
        self.top_move_candidates = int(top_move_candidates)
        self.top_board_candidates = int(top_board_candidates)
        self.board_weight = float(board_weight)
        self.move_weight = float(move_weight)
        self.detect_weight = float(detect_weight)
        self.move_score_margin = float(move_score_margin)
        self.lookahead_window = int(lookahead_window)
        self.lookahead_weight = float(lookahead_weight)
        self.lookahead_decay = float(lookahead_decay)
        self.vocab = get_vocabulary()

    def decode(
        self,
        sequence_square_logits: list[torch.Tensor] | torch.Tensor,
        *,
        sequence_move_logits: list[torch.Tensor] | torch.Tensor | None = None,
        sequence_detect_logits: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> SequenceBeamSearchResult:
        board_log_probs = [
            torch.log_softmax(_validated_board_logits(logits), dim=1)
            for logits in _as_frame_list(sequence_square_logits)
        ]
        move_logits = _as_optional_frame_list(
            sequence_move_logits, expected_length=len(board_log_probs)
        )
        detect_logits = _as_optional_frame_list(
            sequence_detect_logits,
            expected_length=len(board_log_probs),
        )

        beam = [
            _BeamState(board=board, score=0.0, search_score=0.0, frames=tuple())
            for board in build_board_hypotheses(
                self.initial_board_fen,
                initial_side_to_move=self.initial_side_to_move,
            )
        ]
        for frame_index, frame_board_log_probs in enumerate(board_log_probs):
            next_candidates: list[_BeamState] = []
            frame_move_logits = None if move_logits is None else move_logits[frame_index]
            frame_detect_logits = None if detect_logits is None else detect_logits[frame_index]

            for state in beam:
                stay_step_score = self._step_score(
                    board_score=score_board_state(frame_board_log_probs, state.board),
                    move_score=self._stay_move_score(state.board, frame_move_logits),
                    detect_score=_stay_detect_score(frame_detect_logits),
                )
                stay_action_score = state.score + stay_step_score
                next_candidates.append(
                    _BeamState(
                        board=state.board.copy(stack=False),
                        score=stay_action_score,
                        search_score=stay_action_score,
                        frames=state.frames
                        + (
                            SequenceTrackerFrameResult(
                                fen=state.board.board_fen(),
                                full_fen=state.board.fen(),
                                move_uci=None,
                                move_score=stay_action_score,
                                stay_score=stay_action_score,
                            ),
                        ),
                    )
                )

                for move, move_log_score in self._candidate_moves(
                    state.board,
                    move_logits=frame_move_logits,
                    board_log_probs=frame_board_log_probs,
                    sequence_board_log_probs=board_log_probs,
                    frame_index=frame_index,
                ):
                    next_board = state.board.copy(stack=False)
                    next_board.push(move)
                    move_step_score = self._step_score(
                        board_score=score_board_state(frame_board_log_probs, next_board),
                        move_score=move_log_score,
                        detect_score=_move_detect_score(frame_detect_logits),
                    )
                    lookahead_bonus = self._move_lookahead_bonus(
                        board_log_probs,
                        frame_index=frame_index,
                        stay_board=state.board,
                        move_board=next_board,
                    )
                    move_action_score = state.score + move_step_score
                    move_search_score = move_action_score + lookahead_bonus
                    if move_search_score < stay_action_score + self.move_score_margin:
                        continue
                    next_candidates.append(
                        _BeamState(
                            board=next_board,
                            score=move_action_score,
                            search_score=move_search_score,
                            frames=state.frames
                            + (
                                SequenceTrackerFrameResult(
                                    fen=next_board.board_fen(),
                                    full_fen=next_board.fen(),
                                    move_uci=move.uci(),
                                    move_score=move_action_score,
                                    stay_score=stay_action_score,
                                ),
                            ),
                        )
                    )

            beam = self._prune(next_candidates)

        best_state = max(beam, key=lambda state: state.score)
        return SequenceBeamSearchResult(frames=best_state.frames, total_score=best_state.score)

    def _step_score(self, *, board_score: float, move_score: float, detect_score: float) -> float:
        return (
            self.board_weight * board_score
            + self.move_weight * move_score
            + self.detect_weight * detect_score
        )

    def _stay_move_score(self, board: chess.Board, move_logits: torch.Tensor | None) -> float:
        if move_logits is None:
            return 0.0
        log_probs = _legal_move_log_probs(board, move_logits)
        return float(log_probs[NO_MOVE_IDX].item())

    def _candidate_moves(
        self,
        board: chess.Board,
        *,
        move_logits: torch.Tensor | None,
        board_log_probs: torch.Tensor,
        sequence_board_log_probs: list[torch.Tensor],
        frame_index: int,
    ) -> list[tuple[chess.Move, float]]:
        scored_by_uci: dict[str, tuple[chess.Move, float]] = {}
        log_probs = None if move_logits is None else _legal_move_log_probs(board, move_logits)

        if log_probs is None:
            for move in list(board.legal_moves)[: self.top_move_candidates]:
                scored_by_uci[move.uci()] = (move, 0.0)
        else:
            scored_moves: list[tuple[chess.Move, float]] = []
            for move in board.legal_moves:
                uci = move.uci()
                if not self.vocab.contains(uci):
                    continue
                move_index = self.vocab.uci_to_index(uci)
                if move_index in {NO_MOVE_IDX, UNKNOWN_IDX}:
                    continue
                scored_moves.append((move, float(log_probs[move_index].item())))
            scored_moves.sort(key=lambda item: item[1], reverse=True)
            for move, score in scored_moves[: self.top_move_candidates]:
                scored_by_uci[move.uci()] = (move, score)

        if self.top_board_candidates > 0:
            stay_score = score_board_state(board_log_probs, board)
            board_scored_moves = []
            for move in board.legal_moves:
                next_board = board.copy(stack=False)
                next_board.push(move)
                current_delta = score_legal_move(
                    board_log_probs,
                    board,
                    move,
                    stay_score=stay_score,
                ).delta
                lookahead_bonus = self._move_lookahead_bonus(
                    sequence_board_log_probs,
                    frame_index=frame_index,
                    stay_board=board,
                    move_board=next_board,
                )
                board_scored_moves.append((move, current_delta + lookahead_bonus))
            board_scored_moves.sort(key=lambda item: item[1], reverse=True)
            for move, _delta in board_scored_moves[: self.top_board_candidates]:
                uci = move.uci()
                if uci in scored_by_uci:
                    continue
                move_score = 0.0
                if log_probs is not None and self.vocab.contains(uci):
                    move_index = self.vocab.uci_to_index(uci)
                    if move_index not in {NO_MOVE_IDX, UNKNOWN_IDX}:
                        move_score = float(log_probs[move_index].item())
                scored_by_uci[uci] = (move, move_score)

        return list(scored_by_uci.values())

    def _move_lookahead_bonus(
        self,
        board_log_probs: list[torch.Tensor],
        *,
        frame_index: int,
        stay_board: chess.Board,
        move_board: chess.Board,
    ) -> float:
        if self.lookahead_weight == 0.0 or self.lookahead_window <= 1:
            return 0.0
        bonus = 0.0
        end_index = min(len(board_log_probs), frame_index + self.lookahead_window)
        for future_index in range(frame_index + 1, end_index):
            offset = future_index - frame_index - 1
            decay = self.lookahead_decay**offset
            bonus += decay * (
                score_board_state(board_log_probs[future_index], move_board)
                - score_board_state(board_log_probs[future_index], stay_board)
            )
        return self.lookahead_weight * bonus

    def _prune(self, candidates: list[_BeamState]) -> list[_BeamState]:
        best_by_fen: dict[str, _BeamState] = {}
        for candidate in candidates:
            key = _state_dedupe_key(candidate.board)
            previous = best_by_fen.get(key)
            if previous is None or candidate.search_score > previous.search_score:
                best_by_fen[key] = candidate
        return sorted(best_by_fen.values(), key=lambda state: state.search_score, reverse=True)[
            : self.beam_size
        ]


def build_board_hypotheses(
    initial_board_fen: str,
    *,
    initial_side_to_move: str | None = None,
) -> list[chess.Board]:
    """Build candidate boards when only piece placement is known.

    Real clip metadata often stores only `board_fen()` piece placement. Before
    the first detected move we do not know whose turn it is, so the tracker keeps
    both White-to-move and Black-to-move hypotheses alive unless the clip metadata
    already specifies the side to move.
    """

    if initial_side_to_move in {"w", "b"}:
        board = chess.Board()
        board.set_board_fen(initial_board_fen)
        board.turn = chess.WHITE if initial_side_to_move == "w" else chess.BLACK
        return [board]

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


def _as_frame_list(sequence: list[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(sequence, list):
        return sequence
    if sequence.ndim == 0:
        raise ValueError("Expected at least one frame dimension")
    return [sequence[index] for index in range(sequence.shape[0])]


def _as_optional_frame_list(
    sequence: list[torch.Tensor] | torch.Tensor | None,
    *,
    expected_length: int,
) -> list[torch.Tensor] | None:
    if sequence is None:
        return None
    frames = _as_frame_list(sequence)
    if len(frames) != expected_length:
        raise ValueError(
            f"Expected {expected_length} frame(s) of sequence evidence, got {len(frames)}"
        )
    return frames


def _legal_move_log_probs(board: chess.Board, move_logits: torch.Tensor) -> torch.Tensor:
    if move_logits.ndim != 1:
        raise ValueError(f"Expected move logits shaped (V,), got {tuple(move_logits.shape)}")
    legal_mask = get_legal_mask(board).to(device=move_logits.device)
    masked_logits = apply_constraint_mask(move_logits, legal_mask)
    return F.log_softmax(masked_logits, dim=-1)


def _stay_detect_score(detect_logits: torch.Tensor | None) -> float:
    if detect_logits is None:
        return 0.0
    return float(F.logsigmoid(-detect_logits.reshape(()).to(torch.float32)).item())


def _move_detect_score(detect_logits: torch.Tensor | None) -> float:
    if detect_logits is None:
        return 0.0
    return float(F.logsigmoid(detect_logits.reshape(()).to(torch.float32)).item())


def _state_dedupe_key(board: chess.Board) -> str:
    fen_fields = board.fen().split(" ")
    return " ".join(fen_fields[:4])
