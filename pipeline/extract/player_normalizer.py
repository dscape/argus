"""Normalize player names to FIDE IDs using fuzzy matching."""

import logging
from functools import lru_cache

from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)


class PlayerNormalizer:
    """Resolve player name strings to FIDE IDs.

    Uses a two-stage approach:
    1. Exact match against player_aliases table
    2. Fuzzy match using pg_trgm with configurable similarity threshold
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.threshold = similarity_threshold
        self._cache: dict[str, tuple[int | None, float]] = {}

    def normalize(self, name: str) -> tuple[int | None, float]:
        """Resolve a player name to a FIDE ID.

        Returns:
            (fide_id, confidence) where fide_id is None if not found.
            Confidence is 1.0 for exact match, similarity score for fuzzy.
        """
        if not name or len(name) < 2:
            return None, 0.0

        name_lower = name.strip().lower()

        # Check cache
        if name_lower in self._cache:
            return self._cache[name_lower]

        # Stage 1: Exact match
        result = self._exact_match(name_lower)
        if result:
            self._cache[name_lower] = result
            return result

        # Stage 2: Fuzzy match
        result = self._fuzzy_match(name.strip())
        self._cache[name_lower] = result
        return result

    def _exact_match(self, name_lower: str) -> tuple[int | None, float] | None:
        """Try exact match against aliases table."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fide_id
                    FROM player_aliases
                    WHERE lower(alias) = %s
                    LIMIT 1
                    """,
                    (name_lower,),
                )
                row = cur.fetchone()
                if row:
                    return (row[0], 1.0)
        return None

    def _fuzzy_match(self, name: str) -> tuple[int | None, float]:
        """Fuzzy match using pg_trgm similarity."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Set similarity threshold for the % operator
                cur.execute(
                    "SELECT set_limit(%s)",
                    (self.threshold,),
                )
                cur.execute(
                    """
                    SELECT pa.fide_id, pa.alias,
                           similarity(pa.alias, %s) AS sim
                    FROM player_aliases pa
                    WHERE pa.alias %% %s
                    ORDER BY sim DESC
                    LIMIT 3
                    """,
                    (name, name),
                )
                rows = cur.fetchall()

                if not rows:
                    return (None, 0.0)

                # If top match is significantly better than second, use it
                top_fide_id, top_alias, top_sim = rows[0]
                if len(rows) == 1 or (rows[1][2] < top_sim - 0.1):
                    logger.debug(
                        f"Fuzzy match: '{name}' -> '{top_alias}' "
                        f"(fide_id={top_fide_id}, sim={top_sim:.3f})"
                    )
                    return (top_fide_id, top_sim)

                # Multiple close matches — ambiguous
                # Check if they all resolve to the same FIDE ID
                fide_ids = {r[0] for r in rows if r[2] > self.threshold}
                if len(fide_ids) == 1:
                    return (top_fide_id, top_sim)

                logger.debug(
                    f"Ambiguous fuzzy match for '{name}': "
                    f"{[(r[1], r[0], r[2]) for r in rows]}"
                )
                return (None, 0.0)

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
