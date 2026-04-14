from __future__ import annotations

import chess
import torch
from pipeline.shared.board_tracking import (
    LegalMoveStateTracker,
    LookaheadLegalMoveStateTracker,
    board_to_class_ids,
    build_board_hypotheses,
    score_board_state,
    score_legal_move,
)


def _logits_for_board(board: chess.Board, *, preferred_logit: float = 5.0) -> torch.Tensor:
    logits = torch.zeros((64, 13), dtype=torch.float32)
    for square_index, class_id in enumerate(board_to_class_ids(board)):
        logits[square_index, class_id] = preferred_logit
    return logits


def test_build_board_hypotheses_returns_white_and_black_turns() -> None:
    boards = build_board_hypotheses(chess.STARTING_BOARD_FEN)

    assert len(boards) == 2
    assert boards[0].board_fen() == chess.STARTING_BOARD_FEN
    assert boards[1].board_fen() == chess.STARTING_BOARD_FEN
    assert boards[0].turn is chess.WHITE
    assert boards[1].turn is chess.BLACK


def test_score_board_state_prefers_matching_board() -> None:
    board = chess.Board()
    other = board.copy(stack=False)
    other.push_uci("e2e4")

    logits = _logits_for_board(board)

    assert score_board_state(torch.log_softmax(logits, dim=1), board) > score_board_state(
        torch.log_softmax(logits, dim=1),
        other,
    )


def test_score_legal_move_prefers_expected_changed_squares() -> None:
    board = chess.Board()
    moved = board.copy(stack=False)
    moved.push_uci("e2e4")
    logits = _logits_for_board(moved)

    log_probs = torch.log_softmax(logits, dim=1)
    stay_score = score_board_state(log_probs, board)
    action = score_legal_move(
        log_probs,
        board,
        chess.Move.from_uci("e2e4"),
        stay_score=stay_score,
    )

    assert action.move is not None
    assert action.move.uci() == "e2e4"
    assert action.delta > 0.0
    assert action.score > stay_score


def test_tracker_stays_put_when_no_move_is_supported() -> None:
    tracker = LegalMoveStateTracker(chess.STARTING_BOARD_FEN)

    result = tracker.update(_logits_for_board(chess.Board()))

    assert result.move_uci is None
    assert result.fen == chess.STARTING_BOARD_FEN


def test_tracker_applies_supported_legal_move() -> None:
    tracker = LegalMoveStateTracker(
        chess.STARTING_BOARD_FEN,
        move_accept_threshold=0.1,
        move_accept_margin=0.1,
    )
    before = chess.Board()
    after = before.copy(stack=False)
    after.push_uci("e2e4")
    logits = _logits_for_board(after)

    result = tracker.update(logits)

    assert result.move_uci == "e2e4"
    assert result.fen == after.board_fen()
    assert result.turn_resolved is True


def test_tracker_respects_move_accept_threshold_on_delta_scale() -> None:
    tracker = LegalMoveStateTracker(
        chess.STARTING_BOARD_FEN,
        move_accept_threshold=100.0,
        move_accept_margin=0.1,
    )
    after = chess.Board()
    after.push_uci("e2e4")

    result = tracker.update(_logits_for_board(after))

    assert result.move_uci is None
    assert result.fen == chess.STARTING_BOARD_FEN


def test_lookahead_tracker_accumulates_future_evidence_for_a_move() -> None:
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    first_frame = _logits_for_board(board)
    noisy_transition = _logits_for_board(board)
    noisy_transition[36, board_to_class_ids(moved_board)[36]] = 3.0
    noisy_transition[52, board_to_class_ids(moved_board)[52]] = 0.0
    settled_frame = _logits_for_board(moved_board)

    tracker = LookaheadLegalMoveStateTracker(
        chess.STARTING_BOARD_FEN,
        lookahead_window=2,
        move_score_margin=1.0,
    )
    results = tracker.decode([first_frame, noisy_transition, settled_frame])

    assert [result.move_uci for result in results] == [None, "e2e4", None]
    assert results[-1].fen == moved_board.board_fen()


def test_lookahead_tracker_respects_change_mask() -> None:
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    tracker = LookaheadLegalMoveStateTracker(
        chess.STARTING_BOARD_FEN,
        lookahead_window=2,
        move_score_margin=0.1,
    )
    results = tracker.decode(
        [_logits_for_board(board), _logits_for_board(moved_board)],
        change_mask=[False, False],
    )

    assert [result.move_uci for result in results] == [None, None]
    assert results[-1].fen == chess.STARTING_BOARD_FEN


def test_tracker_resolves_unknown_turn_from_first_move() -> None:
    initial_board = "k7/8/8/8/8/8/4p3/7K"
    tracker = LegalMoveStateTracker(
        initial_board,
        move_accept_threshold=0.1,
        move_accept_margin=0.1,
    )

    black_board = chess.Board()
    black_board.set_board_fen(initial_board)
    black_board.turn = chess.BLACK
    black_board.push_uci("e2e1q")
    logits = _logits_for_board(black_board)

    result = tracker.update(logits)

    assert result.move_uci == "e2e1q"
    assert result.turn_resolved is True
    assert tracker.board.turn is chess.WHITE
