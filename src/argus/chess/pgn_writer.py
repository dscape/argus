"""PGN serialization from MoveEvent sequences."""

from __future__ import annotations

from argus.chess.state_machine import GameStateMachine
from argus.types import GameTrack, MoveEvent


class PGNWriter:
    """Converts tracked game data into PGN format."""

    @staticmethod
    def from_move_events(
        events: list[MoveEvent],
        initial_fen: str | None = None,
        white: str = "?",
        black: str = "?",
        event: str = "Argus Reconstruction",
    ) -> str:
        """Convert a sequence of MoveEvents to PGN.

        Args:
            events: Ordered list of move events for a single game.
            initial_fen: Starting FEN (None = standard start).
            white: White player name.
            black: Black player name.
            event: Event name.

        Returns:
            PGN string.
        """
        sm = GameStateMachine(fen=initial_fen)

        for ev in events:
            if not sm.push_move(ev.move_uci):
                # Skip illegal moves (shouldn't happen with constrained head)
                continue

        return sm.to_pgn(white=white, black=black, event=event)

    @staticmethod
    def from_game_track(track: GameTrack) -> str:
        """Convert a GameTrack to PGN."""
        return PGNWriter.from_move_events(
            events=track.moves,
            initial_fen=track.initial_fen if track.initial_fen else None,
        )

    @staticmethod
    def from_uci_list(
        moves: list[str],
        initial_fen: str | None = None,
    ) -> str:
        """Convert a list of UCI move strings to PGN.

        Args:
            moves: List of UCI move strings (e.g. ["e2e4", "e7e5", ...]).
            initial_fen: Starting FEN.

        Returns:
            PGN string.
        """
        sm = GameStateMachine(fen=initial_fen)
        for uci in moves:
            if not sm.push_move(uci):
                break
        return sm.to_pgn()
