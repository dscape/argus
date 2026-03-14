"""Runtime board tracker with per-game chess state machines."""

from __future__ import annotations

import torch

from argus.chess.move_vocabulary import VOCAB_SIZE
from argus.chess.state_machine import GameStateMachine
from argus.types import GameTrack, MoveEvent


class MultiGameTracker:
    """Manages multiple concurrent chess games during inference.

    Each detected board gets a GameStateMachine that validates moves
    and generates legal move masks. Supports beam search for error recovery.
    """

    def __init__(self, beam_width: int = 1) -> None:
        """
        Args:
            beam_width: Number of hypotheses to maintain per game.
                1 = greedy decoding, >1 = beam search.
        """
        self.beam_width = beam_width
        self._games: dict[int, list[GameStateMachine]] = {}
        self._move_events: dict[int, list[MoveEvent]] = {}
        self._first_frames: dict[int, int] = {}
        self._last_frames: dict[int, int] = {}

    def get_or_create_game(self, board_id: int) -> list[GameStateMachine]:
        """Get or create state machines for a board.

        Returns list of beam hypotheses (length = beam_width).
        """
        if board_id not in self._games:
            self._games[board_id] = [GameStateMachine() for _ in range(self.beam_width)]
            self._move_events[board_id] = []
        return self._games[board_id]

    def get_legal_masks(
        self,
        board_id: int,
        batch_size: int = 1,
        seq_len: int = 1,
    ) -> torch.Tensor:
        """Get legal move masks for a board.

        For beam_width=1 (greedy), returns the single mask.
        For beam search, returns the mask for the top hypothesis.

        Args:
            board_id: Board identifier.
            batch_size: Batch size to expand to.
            seq_len: Sequence length to expand to.

        Returns:
            (batch_size, seq_len, VOCAB_SIZE) boolean tensor.
        """
        games = self.get_or_create_game(board_id)
        mask = games[0].get_legal_mask()  # Use top hypothesis
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

    def push_move(
        self,
        board_id: int,
        uci: str,
        confidence: float,
        frame_idx: int,
    ) -> bool:
        """Push a predicted move to the tracker.

        For greedy (beam_width=1): validates and pushes if legal.
        For beam search: pushes to all hypotheses that find it legal,
        and maintains the top-K scoring hypotheses.

        Args:
            board_id: Board identifier.
            uci: UCI move string.
            confidence: Model confidence for this move.
            frame_idx: Video frame index.

        Returns:
            True if the move was accepted by at least one hypothesis.
        """
        games = self.get_or_create_game(board_id)

        if board_id not in self._first_frames:
            self._first_frames[board_id] = frame_idx

        accepted = False

        if self.beam_width == 1:
            # Greedy
            sm = games[0]
            fen_before = sm.get_fen()
            if sm.push_move(uci):
                fen_after = sm.get_fen()
                self._move_events[board_id].append(MoveEvent(
                    board_id=board_id,
                    move_uci=uci,
                    fen_before=fen_before,
                    fen_after=fen_after,
                    confidence=confidence,
                    frame_idx=frame_idx,
                    is_legal=True,
                ))
                accepted = True
        else:
            # Beam search: try the move on all hypotheses
            new_beams: list[tuple[float, GameStateMachine]] = []

            for sm in games:
                copy = sm.copy()
                if copy.push_move(uci):
                    new_beams.append((confidence, copy))
                # Also keep the original without this move (lower scored)
                new_beams.append((0.0, sm))

            # Keep top-K by cumulative confidence
            new_beams.sort(key=lambda x: x[0], reverse=True)
            self._games[board_id] = [sm for _, sm in new_beams[: self.beam_width]]

            if any(conf > 0 for conf, _ in new_beams):
                top_sm = self._games[board_id][0]
                self._move_events[board_id].append(MoveEvent(
                    board_id=board_id,
                    move_uci=uci,
                    fen_before="",  # Beam search - FEN tracking is approximate
                    fen_after=top_sm.get_fen(),
                    confidence=confidence,
                    frame_idx=frame_idx,
                    is_legal=True,
                ))
                accepted = True

        self._last_frames[board_id] = frame_idx
        return accepted

    def finalize_game(self, board_id: int) -> GameTrack:
        """Finalize a game and produce PGN."""
        games = self.get_or_create_game(board_id)
        sm = games[0]  # Top hypothesis

        return GameTrack(
            board_id=board_id,
            moves=list(self._move_events.get(board_id, [])),
            pgn=sm.to_pgn(),
            initial_fen=sm._initial_fen,
            final_fen=sm.get_fen(),
            first_frame=self._first_frames.get(board_id, 0),
            last_frame=self._last_frames.get(board_id, 0),
            status="completed" if sm.is_game_over() else "in_progress",
        )

    def finalize_all(self) -> list[GameTrack]:
        """Finalize all tracked games."""
        return [self.finalize_game(bid) for bid in sorted(self._games.keys())]

    def reset(self) -> None:
        """Clear all tracked games."""
        self._games.clear()
        self._move_events.clear()
        self._first_frames.clear()
        self._last_frames.clear()
