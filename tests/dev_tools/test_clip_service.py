"""Tests for clip inspection service replay validation."""

from __future__ import annotations

import io

import chess
import torch
from api.services.data import clip_service

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary

VOCAB = get_vocabulary()
EXPECTED_INITIAL_BOARD_FEN = "rnbqkbnr/pp1p1ppp/2p1p3/8/4P3/3P2P1/PPP2P1P/RNBQKBNR"


def _make_midgame_clip_bytes() -> bytes:
    board = chess.Board()
    for uci in ["e2e4", "c7c6", "d2d3", "e7e6", "g2g3"]:
        board.push(chess.Move.from_uci(uci))

    move_targets = torch.full((4,), NO_MOVE_IDX, dtype=torch.long)
    move_targets[2] = VOCAB.uci_to_index("f1g2")

    clip = {
        "frames": torch.zeros((4, 3, 224, 224), dtype=torch.uint8),
        "move_targets": move_targets,
        "detect_targets": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        "initial_board_fen": board.board_fen(),
        "pgn_moves": "Bg2",
    }

    buffer = io.BytesIO()
    torch.save(clip, buffer)
    return buffer.getvalue()


def test_inspect_replays_from_initial_board_fen_for_midgame_clip() -> None:
    session_id = clip_service.create_session(_make_midgame_clip_bytes(), "midgame.pt")
    try:
        result = clip_service.inspect(session_id)
    finally:
        clip_service.delete_session(session_id)

    assert result["replay_valid"] is True
    assert result["replay_error"] is None
    assert result["moves"][0]["uci"] == "f1g2"
    assert result["moves"][0]["san"] == "Bg2"
    assert result["metadata"]["initial_board_fen"] == EXPECTED_INITIAL_BOARD_FEN
