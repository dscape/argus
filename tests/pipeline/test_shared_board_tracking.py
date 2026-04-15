from __future__ import annotations

import chess
import torch
from pipeline.shared.board_tracking import (
    LegalMoveStateTracker,
    LegalSequenceBeamDecoder,
    LookaheadLegalMoveStateTracker,
    SegmentalLegalSequenceDecoder,
    board_to_class_ids,
    build_board_hypotheses,
    score_board_state,
    score_legal_move,
)

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary


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


def test_build_board_hypotheses_uses_known_initial_side_to_move() -> None:
    boards = build_board_hypotheses(chess.STARTING_BOARD_FEN, initial_side_to_move="b")

    assert len(boards) == 1
    assert boards[0].turn is chess.BLACK


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


def test_legal_sequence_beam_decoder_combines_board_and_move_evidence() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    square_logits = [_logits_for_board(board), _logits_for_board(moved_board)]
    move_logits = torch.full((2, vocab.size), -8.0)
    move_logits[0, vocab.uci_to_index("e2e4")] = -1.0
    move_logits[0, NO_MOVE_IDX] = 0.0
    move_logits[1, vocab.uci_to_index("e2e4")] = 6.0
    move_logits[1, NO_MOVE_IDX] = -2.0
    detect_logits = torch.tensor([-4.0, 4.0], dtype=torch.float32)

    decoded = LegalSequenceBeamDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        board_weight=1.0,
        move_weight=1.0,
        detect_weight=1.0,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert [frame.move_uci for frame in decoded.frames] == [None, "e2e4"]
    assert decoded.frames[-1].fen == moved_board.board_fen()


def test_legal_sequence_beam_decoder_can_recover_move_from_board_candidates() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    square_logits = [_logits_for_board(board), _logits_for_board(moved_board)]
    move_logits = torch.full((2, vocab.size), -8.0)
    move_logits[0, NO_MOVE_IDX] = 6.0
    move_logits[1, vocab.uci_to_index("a2a3")] = 6.0
    move_logits[1, vocab.uci_to_index("e2e4")] = -6.0
    move_logits[1, NO_MOVE_IDX] = 0.0
    detect_logits = torch.tensor([-4.0, 4.0], dtype=torch.float32)

    decoded = LegalSequenceBeamDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=1,
        top_board_candidates=1,
        board_weight=1.0,
        move_weight=0.2,
        detect_weight=0.1,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert [frame.move_uci for frame in decoded.frames] == [None, "e2e4"]
    assert decoded.frames[-1].fen == moved_board.board_fen()


def test_legal_sequence_beam_decoder_move_score_margin_penalizes_spurious_moves() -> None:
    vocab = get_vocabulary()
    move_logits = torch.full((1, vocab.size), -8.0)
    move_logits[0, vocab.uci_to_index("e2e4")] = 4.0
    move_logits[0, NO_MOVE_IDX] = 0.0
    detect_logits = torch.tensor([4.0], dtype=torch.float32)

    eager = LegalSequenceBeamDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=2,
        top_move_candidates=1,
        board_weight=0.0,
        move_weight=1.0,
        detect_weight=1.0,
        move_score_margin=0.0,
    ).decode(
        [_logits_for_board(chess.Board())],
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )
    conservative = LegalSequenceBeamDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=2,
        top_move_candidates=1,
        board_weight=0.0,
        move_weight=1.0,
        detect_weight=1.0,
        move_score_margin=9.0,
    ).decode(
        [_logits_for_board(chess.Board())],
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert eager.frames[0].move_uci == "e2e4"
    assert conservative.frames[0].move_uci is None


def test_legal_sequence_beam_decoder_lookahead_bonus_advances_supported_move() -> None:
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    first_frame = _logits_for_board(board)
    noisy_transition = _logits_for_board(board)
    noisy_transition[36, board_to_class_ids(moved_board)[36]] = 3.0
    noisy_transition[52, board_to_class_ids(moved_board)[52]] = 0.0
    settled_frame = _logits_for_board(moved_board)

    plain = LegalSequenceBeamDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=1,
        top_move_candidates=64,
        board_weight=1.0,
        move_weight=0.0,
        detect_weight=0.0,
        move_score_margin=0.1,
    ).decode([first_frame, noisy_transition, settled_frame])
    lookahead = LegalSequenceBeamDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=1,
        top_move_candidates=64,
        board_weight=1.0,
        move_weight=0.0,
        detect_weight=0.0,
        move_score_margin=0.1,
        lookahead_window=2,
        lookahead_weight=1.0,
    ).decode([first_frame, noisy_transition, settled_frame])

    assert plain.frames[1].move_uci is None
    assert plain.frames[1].fen == chess.STARTING_BOARD_FEN
    assert plain.frames[2].move_uci == "e2e4"
    assert lookahead.frames[1].move_uci == "e2e4"
    assert lookahead.frames[1].fen == moved_board.board_fen()


def test_segmental_decoder_suppresses_multi_frame_move_chatter() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    square_logits = [
        _logits_for_board(board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
    ]
    move_logits = torch.full((len(square_logits), vocab.size), -8.0)
    for frame_index in (1, 2, 3):
        move_logits[frame_index, vocab.uci_to_index("e2e4")] = 6.0
        move_logits[frame_index, NO_MOVE_IDX] = -2.0
    detect_logits = torch.tensor([-6.0, 4.0, 5.0, 4.0, -6.0, -6.0], dtype=torch.float32)

    decoded = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        board_weight=1.0,
        move_weight=1.0,
        detect_weight=1.0,
        min_event_separation=4,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    accepted_moves = [frame.move_uci for frame in decoded.frames if frame.move_uci is not None]
    assert accepted_moves == ["e2e4"]
    assert decoded.frames[-1].fen == moved_board.board_fen()


def test_segmental_decoder_returns_constant_board_when_no_events_are_proposed() -> None:
    board = chess.Board()
    square_logits = [_logits_for_board(board) for _ in range(4)]
    detect_logits = torch.full((4,), -8.0, dtype=torch.float32)

    decoded = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=2,
        top_move_candidates=4,
        detect_peak_threshold=0.9,
        board_change_peak_threshold=1.0,
    ).decode(
        square_logits,
        sequence_detect_logits=detect_logits,
    )

    assert [frame.move_uci for frame in decoded.frames] == [None, None, None, None]
    assert all(frame.fen == chess.STARTING_BOARD_FEN for frame in decoded.frames)


def test_segmental_decoder_can_drop_worst_board_frame_from_segment_scoring() -> None:
    board = chess.Board()
    board_log_probs = [
        torch.log_softmax(_logits_for_board(board, preferred_logit=4.0), dim=1),
        torch.log_softmax(_logits_for_board(board, preferred_logit=4.0), dim=1),
        torch.log_softmax(_logits_for_board(board, preferred_logit=1.0), dim=1),
    ]

    plain_score = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        segment_board_drop_worst_frames=0,
    )._segment_board_score(board_log_probs, board, start=0, end=3)
    trimmed_score = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        segment_board_drop_worst_frames=1,
    )._segment_board_score(board_log_probs, board, start=0, end=3)
    kept_frame_scores = sorted(
        [score_board_state(frame_log_probs, board) for frame_log_probs in board_log_probs],
        reverse=True,
    )[:2]

    assert trimmed_score > plain_score
    assert trimmed_score == sum(kept_frame_scores)


def test_segmental_decoder_emits_state_aware_diagnostics() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    square_logits = [
        _logits_for_board(board),
        _logits_for_board(board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
    ]
    move_logits = torch.full((len(square_logits), vocab.size), -8.0)
    move_logits[2, vocab.uci_to_index("e2e4")] = 6.0
    move_logits[2, NO_MOVE_IDX] = -2.0
    detect_logits = torch.tensor([-8.0, -2.0, 4.0, -8.0], dtype=torch.float32)

    decoded = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        detect_peak_threshold=0.1,
        diagnostic_settled_horizon=2,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert decoded.diagnostics is not None
    assert decoded.diagnostics["global_proposals"]
    segments = decoded.diagnostics["segments"]
    assert segments
    top_peaks = segments[0]["top_state_peaks"]
    assert top_peaks
    assert top_peaks[0]["best_move_uci"] == "e2e4"


def test_segmental_decoder_keeps_secondary_strong_peaks_inside_a_cluster() -> None:
    board = chess.Board()
    board_log_probs = [torch.log_softmax(_logits_for_board(board), dim=1) for _ in range(12)]
    detect_logits = torch.full((12,), -8.0, dtype=torch.float32)
    for frame_index, logit in ((1, 2.0), (5, 3.0), (9, 2.5)):
        detect_logits[frame_index] = logit

    centers = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        min_event_separation=10,
        secondary_min_event_separation=4,
        secondary_peak_ratio=0.7,
    )._proposal_centers(
        board_log_probs,
        [value.reshape(()) for value in detect_logits],
    )

    assert centers == [1, 5, 9]


def test_segmental_decoder_can_refine_proposals_with_state_aware_peaks() -> None:
    vocab = get_vocabulary()
    first_board = chess.Board()
    second_board = first_board.copy(stack=False)
    second_board.push_uci("e2e4")
    third_board = second_board.copy(stack=False)
    third_board.push_uci("e7e5")

    square_logits = [
        _logits_for_board(first_board),
        _logits_for_board(second_board),
        _logits_for_board(second_board),
        _logits_for_board(second_board),
        _logits_for_board(third_board),
        _logits_for_board(third_board),
        _logits_for_board(third_board),
    ]
    move_logits = torch.full((len(square_logits), vocab.size), -8.0)
    move_logits[1, vocab.uci_to_index("e2e4")] = 6.0
    move_logits[1, NO_MOVE_IDX] = -2.0
    move_logits[4, vocab.uci_to_index("e7e5")] = 6.0
    move_logits[4, NO_MOVE_IDX] = -2.0
    detect_logits = torch.tensor([-8.0, 5.0, -8.0, -8.0, 4.0, -8.0, -8.0], dtype=torch.float32)

    without_refinement = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        min_event_separation=8,
        move_score_margin=1.0,
        state_aware_proposal_passes=0,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )
    with_refinement = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        min_event_separation=8,
        move_score_margin=1.0,
        state_aware_proposal_passes=1,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert [
        frame.move_uci for frame in without_refinement.frames if frame.move_uci is not None
    ] == ["e2e4"]
    assert [frame.move_uci for frame in with_refinement.frames if frame.move_uci is not None] == [
        "e2e4",
        "e7e5",
    ]
    assert with_refinement.frames[-1].fen == third_board.board_fen()
    assert with_refinement.diagnostics is not None
    assert len(with_refinement.diagnostics["final_proposals"]) == 2
    assert with_refinement.diagnostics["state_aware_refinement_passes"]


def test_segmental_decoder_state_aware_refinement_does_not_retime_existing_move() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    square_logits = [
        _logits_for_board(board),
        _logits_for_board(board),
        _logits_for_board(board),
        _logits_for_board(board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
        _logits_for_board(moved_board),
    ]
    move_logits = torch.full((len(square_logits), vocab.size), -8.0)
    move_logits[1, vocab.uci_to_index("e2e4")] = 5.0
    move_logits[1, NO_MOVE_IDX] = -2.0
    move_logits[4, vocab.uci_to_index("e2e4")] = 6.0
    move_logits[4, NO_MOVE_IDX] = -2.0
    detect_logits = torch.tensor([-8.0, 3.0, -8.0, -8.0, 4.0, -8.0, -8.0], dtype=torch.float32)

    decoded = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        min_event_separation=8,
        move_score_margin=1.0,
        state_aware_proposal_passes=1,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert [frame.move_uci for frame in decoded.frames if frame.move_uci is not None] == ["e2e4"]
    assert decoded.frames[4].move_uci == "e2e4"
    assert decoded.diagnostics is not None
    assert not decoded.diagnostics["state_aware_refinement_passes"]


def test_segmental_decoder_can_replace_unused_proposal_when_budget_is_full() -> None:
    vocab = get_vocabulary()
    first_board = chess.Board()
    second_board = first_board.copy(stack=False)
    second_board.push_uci("e2e4")
    third_board = second_board.copy(stack=False)
    third_board.push_uci("e7e5")

    square_logits = [
        _logits_for_board(first_board, preferred_logit=2.0),
        _logits_for_board(second_board, preferred_logit=2.0),
        _logits_for_board(second_board, preferred_logit=2.0),
        _logits_for_board(second_board, preferred_logit=2.0),
        _logits_for_board(third_board, preferred_logit=2.0),
        _logits_for_board(third_board, preferred_logit=2.0),
        _logits_for_board(third_board, preferred_logit=2.0),
        _logits_for_board(third_board, preferred_logit=2.0),
        _logits_for_board(third_board, preferred_logit=2.0),
    ]
    move_logits = torch.full((len(square_logits), vocab.size), -8.0)
    move_logits[1, vocab.uci_to_index("e2e4")] = 6.0
    move_logits[1, NO_MOVE_IDX] = -2.0
    move_logits[4, vocab.uci_to_index("e7e5")] = 6.0
    move_logits[4, NO_MOVE_IDX] = -2.0
    detect_logits = torch.tensor(
        [-8.0, 5.0, -8.0, -8.0, 3.5, -8.0, -8.0, -8.0, 4.0], dtype=torch.float32
    )

    decoded = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        min_event_separation=7,
        max_event_proposals=2,
        move_score_margin=10.0,
        state_aware_proposal_passes=1,
    ).decode(
        square_logits,
        sequence_move_logits=move_logits,
        sequence_detect_logits=detect_logits,
    )

    assert [frame.move_uci for frame in decoded.frames if frame.move_uci is not None] == [
        "e2e4",
        "e7e5",
    ]
    assert decoded.frames[-1].fen == third_board.board_fen()
    assert decoded.diagnostics is not None
    refinement = decoded.diagnostics["state_aware_refinement_passes"]
    assert refinement
    assert refinement[0]["replaced_proposal_frame"] == 8
    assert [proposal["frame_index"] for proposal in decoded.diagnostics["final_proposals"]] == [
        1,
        4,
    ]


def test_segmental_decoder_reports_possible_illegal_segment() -> None:
    board = chess.Board()
    illegal_board = board.copy(stack=False)
    illegal_board.remove_piece_at(chess.D1)
    illegal_board.remove_piece_at(chess.C1)
    illegal_board.remove_piece_at(chess.B1)

    decoded = SegmentalLegalSequenceDecoder(
        chess.STARTING_BOARD_FEN,
        beam_size=4,
        top_move_candidates=8,
        anomaly_change_evidence_threshold=0.1,
        diagnostic_settled_horizon=2,
    ).decode(
        [
            _logits_for_board(board),
            _logits_for_board(illegal_board),
            _logits_for_board(illegal_board),
        ],
        sequence_detect_logits=torch.tensor([-8.0, 4.0, -8.0], dtype=torch.float32),
    )

    assert decoded.diagnostics is not None
    illegal_segments = decoded.diagnostics["possible_illegal_segments"]
    assert illegal_segments
    assert illegal_segments[0]["frame_index"] == 1
    assert illegal_segments[0]["possible_illegal_move"] is True
    assert decoded.diagnostics["segments"][0]["possible_illegal_move"] is True
