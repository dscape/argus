from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from pipeline.physical.move_data import (
    _build_replay_supervision_targets,
    _infer_annotated_moves,
    _replay_targets_for_clip,
    build_board_hypotheses_from_piece_fen,
)
from pipeline.shared import board_to_class_ids

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary


@dataclass(frozen=True)
class _Row:
    labels: tuple[int, ...]


def test_replay_targets_for_clip_builds_move_and_detect_targets() -> None:
    vocab = get_vocabulary()
    clip = {
        "frames": torch.zeros((3, 3, 8, 8), dtype=torch.float32),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": ["e2e4"],
        "move_frame_indices": [1],
        "frame_indices": [0, 1, 2],
    }

    replay_targets = _replay_targets_for_clip(clip)

    assert replay_targets is not None
    move_targets, detect_targets, legal_masks, board_fens = replay_targets
    assert move_targets.tolist() == [NO_MOVE_IDX, vocab.uci_to_index("e2e4"), NO_MOVE_IDX]
    assert detect_targets.tolist() == [0.0, 1.0, 0.0]
    assert legal_masks.shape[0] == 3
    assert board_fens[0].startswith(chess.STARTING_BOARD_FEN)


def test_build_replay_supervision_targets_supports_causal_tolerance() -> None:
    vocab = get_vocabulary()
    clip = {
        "frames": torch.zeros((5, 3, 8, 8), dtype=torch.float32),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": ["e2e4"],
        "move_frame_indices": [3],
        "frame_indices": [0, 1, 2, 3, 4],
    }

    replay_targets = _build_replay_supervision_targets(
        clip,
        move_target_pre_frames=2,
        detect_target_radius=1,
        detect_target_decay=0.5,
    )

    assert replay_targets is not None
    move_targets, detect_targets, _legal_masks, _board_fens, move_loss_mask, move_loss_weights = (
        replay_targets
    )
    expected_move = vocab.uci_to_index("e2e4")
    assert move_targets.tolist() == [
        NO_MOVE_IDX,
        expected_move,
        expected_move,
        expected_move,
        NO_MOVE_IDX,
    ]
    assert detect_targets.tolist() == [0.0, 0.0, 0.5, 1.0, 0.5]
    assert move_loss_mask.tolist() == [False, True, True, True, False]
    assert move_loss_weights.tolist() == [0.0, 0.25, 0.5, 1.0, 0.0]


def test_build_board_hypotheses_from_piece_fen_uses_known_side_to_move() -> None:
    boards = build_board_hypotheses_from_piece_fen(
        chess.STARTING_BOARD_FEN, initial_side_to_move="b"
    )

    assert len(boards) == 1
    assert boards[0].turn is chess.BLACK


def test_infer_annotated_moves_recovers_single_legal_transition() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")
    rows = [
        _Row(labels=tuple(board_to_class_ids(board))),
        _Row(labels=tuple(board_to_class_ids(moved_board))),
    ]

    move_targets, detect_targets = _infer_annotated_moves(rows, chess.STARTING_BOARD_FEN)

    assert move_targets == [NO_MOVE_IDX, vocab.uci_to_index("e2e4")]
    assert detect_targets == [0.0, 1.0]
