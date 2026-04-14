from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from pipeline.physical.move_data import _infer_annotated_moves, _replay_targets_for_clip
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
