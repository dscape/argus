from __future__ import annotations

import chess
import numpy as np
import torch

from pipeline.board_inference import Tracker, infer_board, score_board, track_boards
from pipeline.shared.board_tracking import board_to_class_ids


def _strong_logits_for_board(board: chess.Board) -> np.ndarray:
    cids = board_to_class_ids(board)
    log = np.full((64, 13), -20.0, dtype=np.float32)
    for i, c in enumerate(cids):
        log[i, c] = 5.0
    return log


def test_infer_board_prefers_starting_position_when_logits_match() -> None:
    b0 = chess.Board()
    logits = _strong_logits_for_board(b0)
    out = infer_board(logits, num_samples=32, seed=0)
    assert out.board.board_fen() == b0.board_fen()
    assert out.score == score_board(b0, logits)


def test_track_boards_stays_on_still_frame() -> None:
    b0 = chess.Board()
    logits = _strong_logits_for_board(b0)
    states = track_boards([logits, logits, logits], num_samples=32, seed=1)
    assert len(states) == 3
    for s in states:
        assert s.board.board_fen() == b0.board_fen()


def test_tracker_prefers_legal_reachability_over_random_snapshot() -> None:
    b0 = chess.Board()
    b0.push_san("e4")
    logits = _strong_logits_for_board(b0)
    t = Tracker(num_samples=32, seed=2)
    t.from_seed_board(chess.Board())
    s = t.update(logits)
    assert s.board.board_fen() == b0.board_fen()


def test_constrained_path_matches_torch() -> None:
    from pipeline.board_inference.state import numpy_logits_to_constrained_class_ids
    from pipeline.shared.board_constraints import constrained_board_class_ids

    l = _strong_logits_for_board(chess.Board())
    t = torch.as_tensor(l, dtype=torch.float32)
    expect = [int(x) for x in constrained_board_class_ids(t).tolist()]
    assert numpy_logits_to_constrained_class_ids(l) == expect
