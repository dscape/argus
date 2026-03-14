"""Per-game chess state tracker wrapping python-chess.

Each tracked game gets one GameStateMachine instance. It is the
authoritative source of legality and state — the model proposes
moves, the state machine validates and commits them.
"""

from __future__ import annotations

import chess
import chess.pgn
import io

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import VOCAB_SIZE, get_vocabulary

import torch


class GameStateMachine:
    """Tracks a single chess game's state and enforces legality."""

    def __init__(self, fen: str | None = None) -> None:
        """Initialize with optional starting FEN (defaults to standard start)."""
        if fen is not None:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self._move_history: list[chess.Move] = []
        self._initial_fen = self.board.fen()

    def get_legal_mask(self) -> torch.Tensor:
        """Get the legal move mask for the current position.

        Returns:
            Boolean tensor of shape (VOCAB_SIZE,).
        """
        return get_legal_mask(self.board)

    def is_legal(self, uci: str) -> bool:
        """Check if a UCI move is legal in the current position."""
        try:
            move = chess.Move.from_uci(uci)
            return move in self.board.legal_moves
        except (ValueError, chess.InvalidMoveError):
            return False

    def push_move(self, uci: str) -> bool:
        """Attempt to push a move. Returns True if legal and applied.

        Args:
            uci: UCI move string (e.g. "e2e4").

        Returns:
            True if move was legal and applied, False otherwise.
        """
        try:
            move = chess.Move.from_uci(uci)
            if move not in self.board.legal_moves:
                return False
            self.board.push(move)
            self._move_history.append(move)
            return True
        except (ValueError, chess.InvalidMoveError):
            return False

    def get_fen(self) -> str:
        """Get the current FEN string."""
        return self.board.fen()

    def get_legal_moves_uci(self) -> list[str]:
        """Get all legal moves as UCI strings."""
        return [m.uci() for m in self.board.legal_moves]

    def get_best_legal_alternative(self, logits: torch.Tensor) -> str | None:
        """Given model logits, return the highest-scoring legal move.

        Args:
            logits: Raw logits of shape (VOCAB_SIZE,).

        Returns:
            UCI string of the best legal move, or None if no legal moves.
        """
        vocab = get_vocabulary()
        mask = self.get_legal_mask()

        # Mask out illegal moves and special tokens for this query
        masked = logits.clone()
        masked[~mask] = float("-inf")

        # Find the best move (excluding NO_MOVE and UNKNOWN)
        best_idx = masked[:VOCAB_SIZE - 2].argmax().item()
        if masked[best_idx] == float("-inf"):
            return None

        return vocab.index_to_uci(best_idx)

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.board.is_game_over()

    def result(self) -> str:
        """Get the game result (e.g. '1-0', '0-1', '1/2-1/2', '*')."""
        if self.board.is_game_over():
            return self.board.result()
        return "*"

    def to_pgn(
        self,
        white: str = "?",
        black: str = "?",
        event: str = "Argus Reconstruction",
        round_num: str = "?",
    ) -> str:
        """Export the game as a PGN string.

        Args:
            white: White player name.
            black: Black player name.
            event: Event name.
            round_num: Round number.

        Returns:
            PGN-formatted string.
        """
        game = chess.pgn.Game()
        game.headers["Event"] = event
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Round"] = round_num
        game.headers["Result"] = self.result()

        if self._initial_fen != chess.STARTING_FEN:
            game.headers["FEN"] = self._initial_fen
            game.headers["SetUp"] = "1"

        node = game
        for move in self._move_history:
            node = node.add_variation(move)

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)

    @property
    def move_count(self) -> int:
        """Number of half-moves (plies) played."""
        return len(self._move_history)

    @property
    def fullmove_number(self) -> int:
        """Current full move number."""
        return self.board.fullmove_number

    @property
    def turn(self) -> chess.Color:
        """Whose turn it is (chess.WHITE or chess.BLACK)."""
        return self.board.turn

    def copy(self) -> GameStateMachine:
        """Create a deep copy of this state machine."""
        new = GameStateMachine.__new__(GameStateMachine)
        new.board = self.board.copy()
        new._move_history = list(self._move_history)
        new._initial_fen = self._initial_fen
        return new
