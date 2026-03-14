"""Fixed enumeration of all possible UCI chess moves.

The vocabulary maps each possible move to a unique index (0..1969).
This mapping is deterministic and never changes — every component
(model output head, loss function, metrics) depends on consistency.

Vocabulary layout:
  - Indices 0..1967: all legal UCI moves (source_square + target_square + optional promotion)
  - Index 1968: NO_MOVE (most frames have no move)
  - Index 1969: UNKNOWN (occlusion-uncertain frames)
"""

from __future__ import annotations

import chess

NO_MOVE_IDX = 1968
UNKNOWN_IDX = 1969
VOCAB_SIZE = 1970


class MoveVocabulary:
    """Bijective mapping between UCI move strings and vocabulary indices."""

    def __init__(self) -> None:
        self._uci_to_idx: dict[str, int] = {}
        self._idx_to_uci: dict[int, str] = {}
        self._build()

    def _build(self) -> None:
        """Enumerate all possible UCI moves deterministically.

        Only moves reachable by some chess piece are included:
        - Same rank or file (rook/queen)
        - Same diagonal (bishop/queen)
        - Knight L-shape (rank_diff, file_diff) in {(1,2), (2,1)}
        - King moves are a subset of queen moves (distance 1)
        - Pawn moves are subsets of the above, plus promotion suffixes

        This produces exactly 1968 unique moves for standard chess.
        """
        moves: list[str] = []

        for src in chess.SQUARES:
            for dst in chess.SQUARES:
                if src == dst:
                    continue

                src_name = chess.square_name(src)
                dst_name = chess.square_name(dst)
                src_rank = chess.square_rank(src)
                dst_rank = chess.square_rank(dst)
                src_file = chess.square_file(src)
                dst_file = chess.square_file(dst)

                rank_diff = abs(src_rank - dst_rank)
                file_diff = abs(src_file - dst_file)

                # Only include moves reachable by some chess piece
                is_reachable = (
                    src_rank == dst_rank  # rook/queen horizontal
                    or src_file == dst_file  # rook/queen vertical
                    or rank_diff == file_diff  # bishop/queen diagonal
                    or (rank_diff, file_diff) in ((1, 2), (2, 1))  # knight
                )
                if not is_reachable:
                    continue

                # Check if this could be a pawn promotion move
                is_promotion = (
                    (dst_rank == 7 and src_rank == 6 and file_diff <= 1)
                    or (dst_rank == 0 and src_rank == 1 and file_diff <= 1)
                )

                # Base move is always included (queen/rook/bishop can land here)
                moves.append(f"{src_name}{dst_name}")

                # Promotion variants are additional entries
                if is_promotion:
                    for promo in ["q", "r", "b", "n"]:
                        moves.append(f"{src_name}{dst_name}{promo}")

        # Sort for determinism
        moves.sort()

        for idx, uci in enumerate(moves):
            self._uci_to_idx[uci] = idx
            self._idx_to_uci[idx] = uci

        # Special tokens
        self._uci_to_idx["<no_move>"] = NO_MOVE_IDX
        self._idx_to_uci[NO_MOVE_IDX] = "<no_move>"
        self._uci_to_idx["<unknown>"] = UNKNOWN_IDX
        self._idx_to_uci[UNKNOWN_IDX] = "<unknown>"

        assert len(self._idx_to_uci) == VOCAB_SIZE, (
            f"Expected {VOCAB_SIZE} entries, got {len(self._idx_to_uci)}"
        )

    def uci_to_index(self, uci: str) -> int:
        """Convert a UCI move string to its vocabulary index."""
        return self._uci_to_idx[uci]

    def index_to_uci(self, idx: int) -> str:
        """Convert a vocabulary index to its UCI move string."""
        return self._idx_to_uci[idx]

    def contains(self, uci: str) -> bool:
        """Check if a UCI string is in the vocabulary."""
        return uci in self._uci_to_idx

    @property
    def size(self) -> int:
        return VOCAB_SIZE

    @property
    def num_moves(self) -> int:
        """Number of actual chess moves (excluding special tokens)."""
        return NO_MOVE_IDX


# Module-level singleton for convenience
_VOCAB: MoveVocabulary | None = None


def get_vocabulary() -> MoveVocabulary:
    """Get the singleton move vocabulary instance."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = MoveVocabulary()
    return _VOCAB
