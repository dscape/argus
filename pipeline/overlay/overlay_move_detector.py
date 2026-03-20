"""Detect chess moves by diffing consecutive board states from a 2D overlay.

Unlike frame-differencing approaches that detect motion, this module compares
actual board positions (FENs) extracted from the overlay. When the position
changes, it finds the legal move that transforms the old position into the new
one using python-chess.
"""

import logging
from dataclasses import dataclass, field

import chess

logger = logging.getLogger(__name__)

# Minimum consecutive frames with the same FEN before we consider
# the position "stable" (handles move animation frames).
STABILITY_WINDOW = 2


@dataclass
class OverlayDetectedMove:
    """A move detected from overlay FEN changes."""

    move_index: int  # Sequential move number (0-based ply)
    move_uci: str
    move_san: str
    frame_idx: int  # Frame index where the move was detected
    timestamp_seconds: float
    fen_before: str  # Board FEN before the move
    fen_after: str  # Board FEN after the move


@dataclass
class GameSegment:
    """A single game extracted from a (possibly multi-game) stream."""

    moves: list[OverlayDetectedMove] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def pgn_moves(self) -> str:
        """Return space-separated SAN moves."""
        return " ".join(m.move_san for m in self.moves)

    @property
    def num_moves(self) -> int:
        return len(self.moves)


def find_move_between_positions(
    board_before: chess.Board,
    fen_after: str,
) -> chess.Move | None:
    """Find the legal move that transforms board_before into fen_after.

    Enumerates all legal moves from board_before, applies each, and checks
    if the resulting board FEN matches fen_after.

    Returns the matching move, or None if no legal move produces the target.
    """
    for move in board_before.legal_moves:
        test_board = board_before.copy()
        test_board.push(move)
        if test_board.board_fen() == fen_after:
            return move

    return None


def detect_moves(
    fens: list[str | None],
    frame_indices: list[int],
    fps: float,
    start_time: float = 0.0,
    stability_window: int = STABILITY_WINDOW,
) -> list[GameSegment]:
    """Detect moves from a sequence of board FENs.

    Args:
        fens: Board FEN strings per frame (None for unreadable frames).
        frame_indices: Frame index for each FEN.
        fps: Frames per second of the sampled sequence.
        start_time: Start time offset in seconds.
        stability_window: Frames of identical FEN before accepting a position.

    Returns:
        List of GameSegments (one per game in multi-game streams).
    """
    if len(fens) < 2:
        return []

    # Filter out None FENs
    valid = [(i, fen, idx) for i, (fen, idx) in enumerate(zip(fens, frame_indices)) if fen is not None]

    if len(valid) < 2:
        return []

    segments: list[GameSegment] = []
    current_segment = GameSegment(
        start_frame=valid[0][2],
        start_time=start_time + valid[0][2] / fps if fps > 0 else start_time,
    )

    # Track the current stable position
    stable_fen: str | None = None
    stable_count = 0
    board = chess.Board()  # Assume games start from standard position

    # Check if the first FEN matches the starting position
    starting_fen = chess.STARTING_BOARD_FEN
    first_fen = valid[0][1]

    if first_fen != starting_fen:
        # The video might start mid-game. Try to use the first FEN directly.
        # We can't determine whose move it is without more context,
        # so we'll set up the board but may not detect the first few moves.
        logger.info("Video doesn't start from standard position, attempting mid-game pickup")
        board = chess.Board()
        board.set_board_fen(first_fen)

    stable_fen = first_fen
    stable_count = 1
    ply = 0

    for i in range(1, len(valid)):
        seq_idx, current_fen, frame_idx = valid[i]
        timestamp = start_time + frame_idx / fps if fps > 0 else start_time

        if current_fen == stable_fen:
            stable_count += 1
            continue

        # FEN changed. Check if the previous position was stable.
        if stable_count < stability_window:
            # Previous position wasn't stable yet (animation frame).
            # Update to new FEN and restart stability counter.
            stable_fen = current_fen
            stable_count = 1
            continue

        # Previous position was stable and now changed to a new position.
        # Check for new game (reset to starting position).
        if current_fen == starting_fen and ply > 4:
            # New game detected
            current_segment.end_frame = frame_idx
            current_segment.end_time = timestamp
            if current_segment.moves:
                segments.append(current_segment)

            current_segment = GameSegment(
                start_frame=frame_idx,
                start_time=timestamp,
            )
            board = chess.Board()
            stable_fen = current_fen
            stable_count = 1
            ply = 0
            continue

        # Try to find the legal move
        move = find_move_between_positions(board, current_fen)

        if move is not None:
            san = board.san(move)
            detected_move = OverlayDetectedMove(
                move_index=ply,
                move_uci=move.uci(),
                move_san=san,
                frame_idx=frame_idx,
                timestamp_seconds=timestamp,
                fen_before=stable_fen,
                fen_after=current_fen,
            )
            current_segment.moves.append(detected_move)
            board.push(move)
            ply += 1
        else:
            # No legal move found. Could be a misread or a position we
            # can't track (e.g., the overlay jumped to a different position
            # during analysis/review). Log and try to resync.
            logger.warning(
                f"No legal move found at frame {frame_idx}: "
                f"{stable_fen} -> {current_fen}"
            )
            # Try to resync by setting the board to the new position
            try:
                board = chess.Board()
                board.set_board_fen(current_fen)
                # We lose track of whose turn it is, so this is best-effort
            except ValueError:
                pass

        stable_fen = current_fen
        stable_count = 1

    # Close the last segment
    if valid:
        last_frame_idx = valid[-1][2]
        last_time = start_time + last_frame_idx / fps if fps > 0 else start_time
        current_segment.end_frame = last_frame_idx
        current_segment.end_time = last_time
    if current_segment.moves:
        segments.append(current_segment)

    return segments
