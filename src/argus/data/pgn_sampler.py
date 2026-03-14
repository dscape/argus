"""Sample and filter PGN games for synthetic data generation."""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.pgn


def sample_random_game(min_moves: int = 10, max_moves: int = 80, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    board = chess.Board()
    moves: list[str] = []
    target_length = rng.randint(min_moves, max_moves)
    while not board.is_game_over() and len(moves) < target_length:
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(move.uci())
        board.push(move)
    return moves


def load_pgn_games(pgn_path: str | Path, max_games: int = 1000, min_elo: int = 0, min_moves: int = 10) -> list[list[str]]:
    games: list[list[str]] = []
    path = Path(pgn_path)
    if not path.exists():
        return games
    with open(path) as f:
        while len(games) < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if min_elo > 0:
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0"))
                    black_elo = int(game.headers.get("BlackElo", "0"))
                    if (white_elo + black_elo) / 2 < min_elo:
                        continue
                except ValueError:
                    continue
            moves: list[str] = []
            node = game
            while node.variations:
                next_node = node.variation(0)
                moves.append(next_node.move.uci())
                node = next_node
            if len(moves) >= min_moves:
                games.append(moves)
    return games


def generate_game_dataset(
    num_games: int = 1000, pgn_path: str | Path | None = None,
    min_moves: int = 10, max_moves: int = 80, min_elo: int = 1500, seed: int = 42,
) -> list[list[str]]:
    games: list[list[str]] = []
    if pgn_path is not None:
        games = load_pgn_games(pgn_path, max_games=num_games, min_elo=min_elo, min_moves=min_moves)
    rng = random.Random(seed)
    while len(games) < num_games:
        game_seed = rng.randint(0, 2**31)
        game = sample_random_game(min_moves=min_moves, max_moves=max_moves, seed=game_seed)
        if len(game) >= min_moves:
            games.append(game)
    return games[:num_games]
