"""Shared chess-aware board-state tracking over per-square logits."""

from __future__ import annotations

import math
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
    diagnostics: dict[str, object] | None = None


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


class SegmentalLegalSequenceDecoder:
    """Decode legal board trajectories over sparse proposed move events.

    Unlike the framewise beam, this decoder assumes the board is piecewise
    constant and only allows state changes at a small set of proposed event
    frames. Each accepted move is then scored on the whole following segment
    rather than one frame at a time, which is meant to suppress chatter.
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
        detect_peak_threshold: float = 0.1,
        board_change_peak_threshold: float = 2.0 / 64.0,
        min_event_separation: int = 4,
        secondary_min_event_separation: int | None = None,
        secondary_peak_ratio: float = 0.8,
        state_aware_proposal_passes: int = 0,
        anomaly_change_evidence_threshold: float = 0.25,
        anomaly_settled_gain_threshold: float = 0.0,
        segment_board_drop_worst_frames: int = 0,
        event_window_radius: int = 1,
        max_event_proposals: int = 32,
        diagnostic_settled_horizon: int = 8,
        diagnostic_top_peaks_per_segment: int = 3,
    ) -> None:
        if beam_size <= 0:
            raise ValueError(f"beam_size must be > 0, got {beam_size}")
        if top_move_candidates <= 0:
            raise ValueError(f"top_move_candidates must be > 0, got {top_move_candidates}")
        if top_board_candidates < 0:
            raise ValueError(f"top_board_candidates must be >= 0, got {top_board_candidates}")
        if not 0.0 <= detect_peak_threshold <= 1.0:
            raise ValueError(
                f"detect_peak_threshold must be in [0, 1], got {detect_peak_threshold}"
            )
        if board_change_peak_threshold < 0.0:
            raise ValueError(
                "board_change_peak_threshold must be >= 0, got "
                f"{board_change_peak_threshold}"
            )
        if min_event_separation <= 0:
            raise ValueError(f"min_event_separation must be > 0, got {min_event_separation}")
        if secondary_min_event_separation is not None and secondary_min_event_separation <= 0:
            raise ValueError(
                "secondary_min_event_separation must be > 0, got "
                f"{secondary_min_event_separation}"
            )
        if not 0.0 <= secondary_peak_ratio <= 1.0:
            raise ValueError(
                f"secondary_peak_ratio must be in [0, 1], got {secondary_peak_ratio}"
            )
        if state_aware_proposal_passes < 0:
            raise ValueError(
                "state_aware_proposal_passes must be >= 0, got "
                f"{state_aware_proposal_passes}"
            )
        if anomaly_change_evidence_threshold < 0.0:
            raise ValueError(
                "anomaly_change_evidence_threshold must be >= 0, got "
                f"{anomaly_change_evidence_threshold}"
            )
        if segment_board_drop_worst_frames < 0:
            raise ValueError(
                "segment_board_drop_worst_frames must be >= 0, got "
                f"{segment_board_drop_worst_frames}"
            )
        if event_window_radius < 0:
            raise ValueError(f"event_window_radius must be >= 0, got {event_window_radius}")
        if max_event_proposals <= 0:
            raise ValueError(f"max_event_proposals must be > 0, got {max_event_proposals}")
        if diagnostic_settled_horizon <= 0:
            raise ValueError(
                "diagnostic_settled_horizon must be > 0, got "
                f"{diagnostic_settled_horizon}"
            )
        if diagnostic_top_peaks_per_segment <= 0:
            raise ValueError(
                "diagnostic_top_peaks_per_segment must be > 0, got "
                f"{diagnostic_top_peaks_per_segment}"
            )
        self.initial_board_fen = initial_board_fen
        self.initial_side_to_move = initial_side_to_move
        self.beam_size = int(beam_size)
        self.top_move_candidates = int(top_move_candidates)
        self.top_board_candidates = int(top_board_candidates)
        self.board_weight = float(board_weight)
        self.move_weight = float(move_weight)
        self.detect_weight = float(detect_weight)
        self.move_score_margin = float(move_score_margin)
        self.detect_peak_threshold = float(detect_peak_threshold)
        self.board_change_peak_threshold = float(board_change_peak_threshold)
        self.min_event_separation = int(min_event_separation)
        self.secondary_min_event_separation = (
            None
            if secondary_min_event_separation is None
            else int(secondary_min_event_separation)
        )
        self.secondary_peak_ratio = float(secondary_peak_ratio)
        self.state_aware_proposal_passes = int(state_aware_proposal_passes)
        self.anomaly_change_evidence_threshold = float(anomaly_change_evidence_threshold)
        self.anomaly_settled_gain_threshold = float(anomaly_settled_gain_threshold)
        self.segment_board_drop_worst_frames = int(segment_board_drop_worst_frames)
        self.event_window_radius = int(event_window_radius)
        self.max_event_proposals = int(max_event_proposals)
        self.diagnostic_settled_horizon = int(diagnostic_settled_horizon)
        self.diagnostic_top_peaks_per_segment = int(diagnostic_top_peaks_per_segment)
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
        if not board_log_probs:
            return SequenceBeamSearchResult(frames=tuple(), total_score=0.0)
        move_logits = _as_optional_frame_list(
            sequence_move_logits,
            expected_length=len(board_log_probs),
        )
        detect_logits = _as_optional_frame_list(
            sequence_detect_logits,
            expected_length=len(board_log_probs),
        )
        candidate_boards = build_board_hypotheses(
            self.initial_board_fen,
            initial_side_to_move=self.initial_side_to_move,
        )
        global_event_centers = self._proposal_centers(board_log_probs, detect_logits)
        final_event_centers = list(global_event_centers)
        best_state = self._decode_with_event_centers(
            board_log_probs,
            move_logits,
            detect_logits,
            candidate_boards,
            event_centers=final_event_centers,
        )

        refinement_passes: list[dict[str, object]] = []
        best_move_count = self._move_count(best_state.frames)
        raw_candidate_scores = self._proposal_candidate_scores(
            board_log_probs,
            detect_logits,
            include_fallback=False,
        )
        for pass_index in range(self.state_aware_proposal_passes):
            refinement_candidates = self._state_aware_refinement_candidates(
                best_state.frames,
                board_log_probs,
                move_logits,
                detect_logits,
                existing_event_centers=final_event_centers,
            )
            if not refinement_candidates:
                break

            current_move_frames = {
                frame_index
                for frame_index, frame in enumerate(best_state.frames)
                if frame.move_uci is not None
            }
            removable_event_centers = sorted(
                [center for center in final_event_centers if center not in current_move_frames],
                key=lambda center: raw_candidate_scores.get(center, 0.0),
            )

            accepted_proposal: dict[str, object] | None = None
            accepted_replacement: int | None = None
            for proposal in refinement_candidates:
                frame_index = int(proposal["frame_index"])
                candidate_event_sets: list[tuple[list[int], int | None]] = []
                if len(final_event_centers) < self.max_event_proposals:
                    next_event_centers = sorted(set(final_event_centers) | {frame_index})
                    if next_event_centers != final_event_centers:
                        candidate_event_sets.append((next_event_centers, None))
                for removable_center in removable_event_centers:
                    if removable_center == frame_index:
                        continue
                    next_event_centers = sorted(
                        (set(final_event_centers) - {removable_center}) | {frame_index}
                    )
                    if next_event_centers == final_event_centers:
                        continue
                    candidate_event_sets.append((next_event_centers, removable_center))

                for next_event_centers, replaced_center in candidate_event_sets:
                    refined_state = self._decode_with_event_centers(
                        board_log_probs,
                        move_logits,
                        detect_logits,
                        candidate_boards,
                        event_centers=next_event_centers,
                    )
                    refined_move_count = self._move_count(refined_state.frames)
                    if refined_move_count <= best_move_count:
                        continue
                    accepted_proposal = proposal
                    accepted_replacement = replaced_center
                    final_event_centers = next_event_centers
                    best_state = refined_state
                    best_move_count = refined_move_count
                    refinement_passes.append(
                        {
                            "pass_index": pass_index + 1,
                            "added_proposals": [proposal],
                            "replaced_proposal_frame": replaced_center,
                            "proposal_centers": next_event_centers,
                            "move_count": refined_move_count,
                        }
                    )
                    break
                if accepted_proposal is not None:
                    break
            if accepted_proposal is None:
                break

        return SequenceBeamSearchResult(
            frames=best_state.frames,
            total_score=best_state.score,
            diagnostics=self._build_diagnostics(
                best_state.frames,
                board_log_probs,
                move_logits,
                detect_logits,
                global_event_centers=global_event_centers,
                final_event_centers=final_event_centers,
                refinement_passes=refinement_passes,
            ),
        )

    def _decode_with_event_centers(
        self,
        board_log_probs: list[torch.Tensor],
        move_logits: list[torch.Tensor] | None,
        detect_logits: list[torch.Tensor] | None,
        candidate_boards: list[chess.Board],
        *,
        event_centers: list[int],
    ) -> _BeamState:
        if not event_centers:
            best_state: _BeamState | None = None
            for board in candidate_boards:
                total_score = self.board_weight * self._segment_board_score(
                    board_log_probs,
                    board,
                    start=0,
                    end=len(board_log_probs),
                )
                candidate = _BeamState(
                    board=board.copy(stack=False),
                    score=total_score,
                    search_score=total_score,
                    frames=self._segment_results(
                        board,
                        start=0,
                        end=len(board_log_probs),
                        score=total_score,
                        stay_score=total_score,
                        move_uci=None,
                    ),
                )
                if best_state is None or candidate.score > best_state.score:
                    best_state = candidate
            assert best_state is not None
            return best_state

        first_center = event_centers[0]
        beam = []
        for board in candidate_boards:
            prefix_score = self.board_weight * self._segment_board_score(
                board_log_probs,
                board,
                start=0,
                end=first_center,
            )
            beam.append(
                _BeamState(
                    board=board.copy(stack=False),
                    score=prefix_score,
                    search_score=prefix_score,
                    frames=self._segment_results(
                        board,
                        start=0,
                        end=first_center,
                        score=prefix_score,
                        stay_score=prefix_score,
                        move_uci=None,
                    ),
                )
            )

        for event_index, center in enumerate(event_centers):
            segment_end = (
                len(board_log_probs)
                if event_index + 1 >= len(event_centers)
                else event_centers[event_index + 1]
            )
            event_window = self._event_window(center, total_frames=len(board_log_probs))
            next_candidates: list[_BeamState] = []
            for state in beam:
                stay_move_score = self._event_move_score(
                    state.board,
                    move_logits,
                    event_window=event_window,
                    move_uci=None,
                )
                stay_detect_score = self._event_detect_score(
                    detect_logits,
                    event_window=event_window,
                    is_move=False,
                )
                stay_increment = self._segment_increment(
                    board_log_probs,
                    state.board,
                    start=center,
                    end=segment_end,
                    move_score=stay_move_score,
                    detect_score=stay_detect_score,
                )
                stay_total = state.score + stay_increment
                next_candidates.append(
                    _BeamState(
                        board=state.board.copy(stack=False),
                        score=stay_total,
                        search_score=stay_total,
                        frames=state.frames
                        + self._segment_results(
                            state.board,
                            start=center,
                            end=segment_end,
                            score=stay_total,
                            stay_score=stay_total,
                            move_uci=None,
                        ),
                    )
                )

                for move, move_event_score in self._candidate_moves(
                    state.board,
                    move_logits=move_logits,
                    board_log_probs=board_log_probs,
                    start=center,
                    end=segment_end,
                    event_window=event_window,
                ):
                    next_board = state.board.copy(stack=False)
                    next_board.push(move)
                    move_increment = self._segment_increment(
                        board_log_probs,
                        next_board,
                        start=center,
                        end=segment_end,
                        move_score=move_event_score,
                        detect_score=self._event_detect_score(
                            detect_logits,
                            event_window=event_window,
                            is_move=True,
                        ),
                    )
                    if move_increment < stay_increment + self.move_score_margin:
                        continue
                    move_total = state.score + move_increment
                    next_candidates.append(
                        _BeamState(
                            board=next_board,
                            score=move_total,
                            search_score=move_total,
                            frames=state.frames
                            + self._segment_results(
                                next_board,
                                start=center,
                                end=segment_end,
                                score=move_total,
                                stay_score=stay_total,
                                move_uci=move.uci(),
                            ),
                        )
                    )

            beam = self._prune(next_candidates)

        return max(beam, key=lambda state: state.score)

    def _state_aware_refinement_candidates(
        self,
        frames: tuple[SequenceTrackerFrameResult, ...],
        board_log_probs: list[torch.Tensor],
        move_logits: list[torch.Tensor] | None,
        detect_logits: list[torch.Tensor] | None,
        *,
        existing_event_centers: list[int],
    ) -> list[dict[str, object]]:
        if not frames:
            return []

        detect_probs = (
            [float(torch.sigmoid(logit.reshape(())).item()) for logit in detect_logits]
            if detect_logits is not None
            else [0.0 for _ in range(len(board_log_probs))]
        )
        change_scores = self._board_change_scores(board_log_probs)
        raw_candidate_scores = self._proposal_candidate_scores(
            board_log_probs,
            detect_logits,
            include_fallback=False,
        )
        existing_centers = set(existing_event_centers)
        candidates: list[dict[str, object]] = []

        for segment_index, segment_start, segment_end, segment_board, applied_move_uci in self._segments(
            frames
        ):
            state_records = self._segment_state_records(
                board_log_probs,
                move_logits,
                detect_logits,
                detect_probs=detect_probs,
                change_scores=change_scores,
                board=segment_board,
                start=segment_start,
                end=segment_end,
            )
            records_by_frame = {
                int(record["frame_index"]): record for record in state_records
            }
            segment_candidates = []
            for frame_index, raw_candidate_score in raw_candidate_scores.items():
                if frame_index in existing_centers or frame_index <= segment_start:
                    continue
                if frame_index < segment_start or frame_index >= segment_end:
                    continue
                record = records_by_frame.get(frame_index)
                if record is None:
                    continue
                if float(record["settled_gain"]) < self.move_score_margin:
                    continue
                segment_candidates.append(
                    {
                        **record,
                        "raw_candidate_score": raw_candidate_score,
                        "segment_index": segment_index,
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "segment_applied_move_uci": applied_move_uci,
                    }
                )
            if not segment_candidates:
                continue
            segment_candidates.sort(
                key=lambda record: (
                    record["settled_gain"],
                    record["raw_candidate_score"],
                    record["change_evidence"],
                ),
                reverse=True,
            )
            candidates.append(segment_candidates[0])

        candidates.sort(
            key=lambda record: (
                record["settled_gain"],
                record["raw_candidate_score"],
                record["change_evidence"],
            ),
            reverse=True,
        )
        return candidates

    def _build_diagnostics(
        self,
        frames: tuple[SequenceTrackerFrameResult, ...],
        board_log_probs: list[torch.Tensor],
        move_logits: list[torch.Tensor] | None,
        detect_logits: list[torch.Tensor] | None,
        *,
        global_event_centers: list[int],
        final_event_centers: list[int],
        refinement_passes: list[dict[str, object]],
    ) -> dict[str, object]:
        detect_probs = (
            [float(torch.sigmoid(logit.reshape(())).item()) for logit in detect_logits]
            if detect_logits is not None
            else [0.0 for _ in range(len(board_log_probs))]
        )
        change_scores = self._board_change_scores(board_log_probs)
        segments: list[dict[str, object]] = []
        unexplained_peaks: list[dict[str, object]] = []
        possible_illegal_segments: list[dict[str, object]] = []

        for segment_index, segment_start, segment_end, segment_board, applied_move_uci in self._segments(
            frames
        ):
            segment_records = self._segment_state_records(
                board_log_probs,
                move_logits,
                detect_logits,
                detect_probs=detect_probs,
                change_scores=change_scores,
                board=segment_board,
                start=segment_start,
                end=segment_end,
            )
            segment_peaks = self._segment_peaks_from_records(segment_records)
            unexplained_peaks.extend(
                peak for peak in segment_peaks if bool(peak.get("unexplained_change"))
            )
            anomaly_peak = self._segment_anomaly_peak(segment_records)
            segment_summary = {
                "segment_index": segment_index,
                "start_frame": segment_start,
                "end_frame": segment_end,
                "length": segment_end - segment_start,
                "board_fen": segment_board.board_fen(),
                "full_fen": segment_board.fen(),
                "applied_move_uci": applied_move_uci,
                "top_state_peaks": segment_peaks[: self.diagnostic_top_peaks_per_segment],
                "possible_illegal_move": anomaly_peak is not None,
                "anomaly_peak": anomaly_peak,
            }
            segments.append(segment_summary)
            if anomaly_peak is not None:
                possible_illegal_segments.append(
                    {
                        "segment_index": segment_index,
                        "start_frame": segment_start,
                        "end_frame": segment_end,
                        "applied_move_uci": applied_move_uci,
                        **anomaly_peak,
                    }
                )

        return {
            "global_proposals": self._proposal_records(
                global_event_centers,
                detect_probs=detect_probs,
                change_scores=change_scores,
            ),
            "final_proposals": self._proposal_records(
                final_event_centers,
                detect_probs=detect_probs,
                change_scores=change_scores,
            ),
            "state_aware_refinement_passes": refinement_passes,
            "segments": segments,
            "unexplained_peaks": unexplained_peaks,
            "possible_illegal_segments": possible_illegal_segments,
            "anomaly_settings": {
                "change_evidence_threshold": self.anomaly_change_evidence_threshold,
                "settled_gain_threshold": self.anomaly_settled_gain_threshold,
            },
            "segment_scoring_settings": {
                "board_drop_worst_frames": self.segment_board_drop_worst_frames,
            },
        }

    def _segments(
        self,
        frames: tuple[SequenceTrackerFrameResult, ...],
    ) -> list[tuple[int, int, int, chess.Board, str | None]]:
        if not frames:
            return []
        segment_starts = [0] + [
            index for index in range(1, len(frames)) if frames[index].move_uci is not None
        ]
        segments: list[tuple[int, int, int, chess.Board, str | None]] = []
        for segment_index, segment_start in enumerate(segment_starts):
            segment_end = (
                len(frames)
                if segment_index + 1 >= len(segment_starts)
                else segment_starts[segment_index + 1]
            )
            segments.append(
                (
                    segment_index,
                    segment_start,
                    segment_end,
                    chess.Board(frames[segment_start].full_fen),
                    frames[segment_start].move_uci,
                )
            )
        return segments

    @staticmethod
    def _proposal_records(
        proposal_centers: list[int],
        *,
        detect_probs: list[float],
        change_scores: list[float],
    ) -> list[dict[str, object]]:
        return [
            {
                "frame_index": frame_index,
                "detect_prob": detect_probs[frame_index],
                "board_change_score": change_scores[frame_index],
            }
            for frame_index in proposal_centers
        ]

    def _proposal_candidate_scores(
        self,
        board_log_probs: list[torch.Tensor],
        detect_logits: list[torch.Tensor] | None,
        *,
        include_fallback: bool = True,
    ) -> dict[int, float]:
        candidate_scores: dict[int, float] = {}
        detect_probs = None
        if detect_logits is not None:
            detect_probs = [float(torch.sigmoid(logit.reshape(())).item()) for logit in detect_logits]
            for index in self._peak_indices(detect_probs):
                score = detect_probs[index]
                if score >= self.detect_peak_threshold:
                    candidate_scores[index] = max(candidate_scores.get(index, float("-inf")), score)

        change_scores = self._board_change_scores(board_log_probs)
        for index in self._peak_indices(change_scores):
            score = change_scores[index]
            if score >= self.board_change_peak_threshold:
                candidate_scores[index] = max(candidate_scores.get(index, float("-inf")), score)

        if include_fallback and not candidate_scores:
            fallback_scores = detect_probs if detect_probs is not None else change_scores
            if fallback_scores:
                best_index = max(range(len(fallback_scores)), key=fallback_scores.__getitem__)
                if fallback_scores[best_index] > 0.0:
                    candidate_scores[best_index] = fallback_scores[best_index]
        return candidate_scores

    def _proposal_centers(
        self,
        board_log_probs: list[torch.Tensor],
        detect_logits: list[torch.Tensor] | None,
    ) -> list[int]:
        candidate_scores = self._proposal_candidate_scores(board_log_probs, detect_logits)
        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        selected: list[tuple[int, float]] = []
        for index, score in ranked:
            nearest_existing = min(
                selected,
                key=lambda item: abs(index - item[0]),
                default=None,
            )
            if nearest_existing is not None:
                distance = abs(index - nearest_existing[0])
                if distance < self.min_event_separation:
                    if (
                        self.secondary_min_event_separation is None
                        or distance < self.secondary_min_event_separation
                        or score < nearest_existing[1] * self.secondary_peak_ratio
                    ):
                        continue
            selected.append((index, score))
            if len(selected) >= self.max_event_proposals:
                break
        return sorted(index for index, _score in selected)

    def _candidate_moves(
        self,
        board: chess.Board,
        *,
        move_logits: list[torch.Tensor] | None,
        board_log_probs: list[torch.Tensor],
        start: int,
        end: int,
        event_window: tuple[int, int],
    ) -> list[tuple[chess.Move, float]]:
        scored_by_uci: dict[str, tuple[chess.Move, float]] = {}

        if move_logits is not None:
            scored_moves: list[tuple[chess.Move, float]] = []
            for move in board.legal_moves:
                uci = move.uci()
                if not self.vocab.contains(uci):
                    continue
                move_index = self.vocab.uci_to_index(uci)
                if move_index in {NO_MOVE_IDX, UNKNOWN_IDX}:
                    continue
                event_score = self._event_move_score(
                    board,
                    move_logits,
                    event_window=event_window,
                    move_uci=uci,
                )
                scored_moves.append((move, event_score))
            scored_moves.sort(key=lambda item: item[1], reverse=True)
            for move, score in scored_moves[: self.top_move_candidates]:
                scored_by_uci[move.uci()] = (move, score)
        else:
            for move in list(board.legal_moves)[: self.top_move_candidates]:
                scored_by_uci[move.uci()] = (move, 0.0)

        if self.top_board_candidates > 0:
            stay_segment_score = self._segment_board_score(
                board_log_probs,
                board,
                start=start,
                end=end,
            )
            board_scored_moves: list[tuple[chess.Move, float]] = []
            for move in board.legal_moves:
                next_board = board.copy(stack=False)
                next_board.push(move)
                move_segment_score = self._segment_board_score(
                    board_log_probs,
                    next_board,
                    start=start,
                    end=end,
                )
                board_scored_moves.append((move, move_segment_score - stay_segment_score))
            board_scored_moves.sort(key=lambda item: item[1], reverse=True)
            for move, _score in board_scored_moves[: self.top_board_candidates]:
                uci = move.uci()
                if uci in scored_by_uci:
                    continue
                scored_by_uci[uci] = (
                    move,
                    self._event_move_score(
                        board,
                        move_logits,
                        event_window=event_window,
                        move_uci=uci,
                    ),
                )

        return list(scored_by_uci.values())

    def _segment_increment(
        self,
        board_log_probs: list[torch.Tensor],
        board: chess.Board,
        *,
        start: int,
        end: int,
        move_score: float,
        detect_score: float,
    ) -> float:
        return (
            self.board_weight * self._segment_board_score(board_log_probs, board, start=start, end=end)
            + self.move_weight * move_score
            + self.detect_weight * detect_score
        )

    def _segment_board_score(
        self,
        board_log_probs: list[torch.Tensor],
        board: chess.Board,
        *,
        start: int,
        end: int,
    ) -> float:
        if end <= start:
            return 0.0
        frame_scores = [score_board_state(board_log_probs[index], board) for index in range(start, end)]
        if self.segment_board_drop_worst_frames > 0 and len(frame_scores) > 1:
            kept_count = max(1, len(frame_scores) - self.segment_board_drop_worst_frames)
            frame_scores = sorted(frame_scores, reverse=True)[:kept_count]
        return float(sum(frame_scores))

    def _event_move_score(
        self,
        board: chess.Board,
        move_logits: list[torch.Tensor] | None,
        *,
        event_window: tuple[int, int],
        move_uci: str | None,
    ) -> float:
        if move_logits is None:
            return 0.0
        start, end = event_window
        values: list[float] = []
        for frame_index in range(start, end):
            log_probs = _legal_move_log_probs(board, move_logits[frame_index])
            if move_uci is None:
                values.append(float(log_probs[NO_MOVE_IDX].item()))
                continue
            if not self.vocab.contains(move_uci):
                continue
            move_index = self.vocab.uci_to_index(move_uci)
            if move_index in {NO_MOVE_IDX, UNKNOWN_IDX}:
                continue
            values.append(float(log_probs[move_index].item()))
        return _aggregate_logmeanexp(values)

    def _event_detect_score(
        self,
        detect_logits: list[torch.Tensor] | None,
        *,
        event_window: tuple[int, int],
        is_move: bool,
    ) -> float:
        if detect_logits is None:
            return 0.0
        start, end = event_window
        values = [
            float(
                (
                    F.logsigmoid(detect_logits[frame_index].reshape(()).to(torch.float32))
                    if is_move
                    else F.logsigmoid(-detect_logits[frame_index].reshape(()).to(torch.float32))
                ).item()
            )
            for frame_index in range(start, end)
        ]
        return _aggregate_logmeanexp(values)

    def _segment_state_records(
        self,
        board_log_probs: list[torch.Tensor],
        move_logits: list[torch.Tensor] | None,
        detect_logits: list[torch.Tensor] | None,
        *,
        detect_probs: list[float],
        change_scores: list[float],
        board: chess.Board,
        start: int,
        end: int,
    ) -> list[dict[str, object]]:
        if end <= start:
            return []

        frame_records: list[dict[str, object]] = []
        for frame_index in range(start, end):
            horizon_end = min(len(board_log_probs), frame_index + self.diagnostic_settled_horizon)
            stay_board_score = self._segment_board_score(
                board_log_probs,
                board,
                start=frame_index,
                end=horizon_end,
            )
            event_window = self._event_window(frame_index, total_frames=len(board_log_probs))
            stay_move_score = self._event_move_score(
                board,
                move_logits,
                event_window=event_window,
                move_uci=None,
            )
            stay_detect_score = self._event_detect_score(
                detect_logits,
                event_window=event_window,
                is_move=False,
            )

            best_move_uci: str | None = None
            best_combined_gain = float("-inf")
            best_board_gain = float("-inf")
            for move in board.legal_moves:
                next_board = board.copy(stack=False)
                next_board.push(move)
                move_board_score = self._segment_board_score(
                    board_log_probs,
                    next_board,
                    start=frame_index,
                    end=horizon_end,
                )
                move_move_score = self._event_move_score(
                    board,
                    move_logits,
                    event_window=event_window,
                    move_uci=move.uci(),
                )
                move_detect_score = self._event_detect_score(
                    detect_logits,
                    event_window=event_window,
                    is_move=True,
                )
                combined_gain = (
                    self.board_weight * (move_board_score - stay_board_score)
                    + self.move_weight * (move_move_score - stay_move_score)
                    + self.detect_weight * (move_detect_score - stay_detect_score)
                )
                if combined_gain > best_combined_gain:
                    best_combined_gain = combined_gain
                    best_board_gain = move_board_score - stay_board_score
                    best_move_uci = move.uci()

            if not math.isfinite(best_combined_gain):
                best_combined_gain = 0.0
                best_board_gain = 0.0
            change_evidence = max(
                0.0,
                detect_probs[frame_index] - self.detect_peak_threshold,
            ) + max(
                0.0,
                change_scores[frame_index] - self.board_change_peak_threshold,
            )
            unexplained_change = bool(change_evidence > 0.0 and best_combined_gain <= 0.0)
            possible_illegal_move = bool(
                change_evidence >= self.anomaly_change_evidence_threshold
                and best_combined_gain <= self.anomaly_settled_gain_threshold
            )
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "best_move_uci": best_move_uci,
                    "settled_gain": best_combined_gain,
                    "settled_board_gain": best_board_gain,
                    "detect_prob": detect_probs[frame_index],
                    "board_change_score": change_scores[frame_index],
                    "change_evidence": change_evidence,
                    "unexplained_change": unexplained_change,
                    "possible_illegal_move": possible_illegal_move,
                    "anomaly_score": (
                        change_evidence
                        + max(0.0, self.anomaly_settled_gain_threshold - best_combined_gain)
                        if possible_illegal_move
                        else 0.0
                    ),
                }
            )
        return frame_records

    def _segment_state_peaks(
        self,
        board_log_probs: list[torch.Tensor],
        move_logits: list[torch.Tensor] | None,
        detect_logits: list[torch.Tensor] | None,
        *,
        detect_probs: list[float],
        change_scores: list[float],
        board: chess.Board,
        start: int,
        end: int,
    ) -> list[dict[str, object]]:
        frame_records = self._segment_state_records(
            board_log_probs,
            move_logits,
            detect_logits,
            detect_probs=detect_probs,
            change_scores=change_scores,
            board=board,
            start=start,
            end=end,
        )
        return self._segment_peaks_from_records(frame_records)

    def _segment_peaks_from_records(
        self,
        frame_records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not frame_records:
            return []

        peak_scores = [max(0.0, float(record["settled_gain"])) for record in frame_records]
        local_peaks = self._peak_indices(peak_scores)
        if not local_peaks:
            local_peaks = sorted(
                range(len(frame_records)),
                key=lambda index: (
                    frame_records[index]["change_evidence"],
                    frame_records[index]["settled_gain"],
                ),
                reverse=True,
            )[: self.diagnostic_top_peaks_per_segment]
        peaks = [frame_records[index] for index in local_peaks]
        peaks.sort(
            key=lambda record: (record["settled_gain"], record["change_evidence"]),
            reverse=True,
        )
        return peaks

    @staticmethod
    def _segment_anomaly_peak(
        frame_records: list[dict[str, object]],
    ) -> dict[str, object] | None:
        anomaly_records = [
            record for record in frame_records if bool(record.get("possible_illegal_move"))
        ]
        if not anomaly_records:
            return None
        best = max(
            anomaly_records,
            key=lambda record: (record["anomaly_score"], record["change_evidence"]),
        )
        return dict(best)

    def _event_window(self, center: int, *, total_frames: int) -> tuple[int, int]:
        start = max(0, center - self.event_window_radius)
        end = min(total_frames, center + self.event_window_radius + 1)
        return start, max(start + 1, end)

    def _segment_results(
        self,
        board: chess.Board,
        *,
        start: int,
        end: int,
        score: float,
        stay_score: float,
        move_uci: str | None,
    ) -> tuple[SequenceTrackerFrameResult, ...]:
        if end <= start:
            return tuple()
        frames = []
        for offset, _frame_index in enumerate(range(start, end)):
            frames.append(
                SequenceTrackerFrameResult(
                    fen=board.board_fen(),
                    full_fen=board.fen(),
                    move_uci=move_uci if offset == 0 else None,
                    move_score=score,
                    stay_score=stay_score,
                )
            )
        return tuple(frames)

    def _prune(self, candidates: list[_BeamState]) -> list[_BeamState]:
        best_by_fen: dict[str, _BeamState] = {}
        for candidate in candidates:
            key = _state_dedupe_key(candidate.board)
            previous = best_by_fen.get(key)
            if previous is None or candidate.score > previous.score:
                best_by_fen[key] = candidate
        return sorted(best_by_fen.values(), key=lambda state: state.score, reverse=True)[
            : self.beam_size
        ]

    @staticmethod
    def _move_count(frames: tuple[SequenceTrackerFrameResult, ...]) -> int:
        return sum(frame.move_uci is not None for frame in frames)

    @staticmethod
    def _peak_indices(scores: list[float]) -> list[int]:
        peaks: list[int] = []
        for index, score in enumerate(scores):
            prev_score = scores[index - 1] if index > 0 else float("-inf")
            next_score = scores[index + 1] if index + 1 < len(scores) else float("-inf")
            if score <= 0.0:
                continue
            if (score > prev_score and score >= next_score) or (
                score >= prev_score and score > next_score
            ):
                peaks.append(index)
        return peaks

    @staticmethod
    def _board_change_scores(board_log_probs: list[torch.Tensor]) -> list[float]:
        if not board_log_probs:
            return []
        scores = [0.0]
        previous_class_ids = board_log_probs[0].argmax(dim=1)
        for frame_index in range(1, len(board_log_probs)):
            current_class_ids = board_log_probs[frame_index].argmax(dim=1)
            scores.append(
                float((current_class_ids != previous_class_ids).sum().item()) / 64.0
            )
            previous_class_ids = current_class_ids
        return scores



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


def _aggregate_logmeanexp(values: list[float]) -> float:
    if not values:
        return 0.0
    max_value = max(values)
    mean_exp = sum(math.exp(value - max_value) for value in values) / float(len(values))
    return max_value + math.log(max(mean_exp, 1e-12))
