"""Drive PGN moves into 3D piece positions for Blender rendering."""

from __future__ import annotations

import chess
from dataclasses import dataclass


@dataclass
class PiecePosition:
    piece_type: str
    color: str
    file: int
    rank: int
    x: float
    y: float
    z: float


@dataclass
class BoardState3D:
    board_id: int
    pieces: list[PiecePosition]
    fen: str


def board_to_3d_positions(
    board: chess.Board, board_id: int = 0,
    board_center: tuple[float, float, float] = (0.0, 0.0, 0.75),
    board_size: float = 0.45,
) -> BoardState3D:
    sq_size = board_size / 8
    cx, cy, cz = board_center
    pieces: list[PiecePosition] = []
    piece_map = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        pieces.append(PiecePosition(
            piece_type=piece_map[piece.piece_type],
            color="white" if piece.color == chess.WHITE else "black",
            file=file_idx, rank=rank_idx,
            x=cx + (file_idx - 3.5) * sq_size,
            y=cy + (rank_idx - 3.5) * sq_size,
            z=cz,
        ))
    return BoardState3D(board_id=board_id, pieces=pieces, fen=board.fen())


def drive_game(
    moves: list[str], board_id: int = 0,
    board_center: tuple[float, float, float] = (0.0, 0.0, 0.75),
    board_size: float = 0.45,
) -> list[BoardState3D]:
    board = chess.Board()
    states = [board_to_3d_positions(board, board_id, board_center, board_size)]
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
                states.append(board_to_3d_positions(board, board_id, board_center, board_size))
            else:
                break
        except (ValueError, chess.InvalidMoveError):
            break
    return states
