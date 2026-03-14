"""YouTube API quota tracking and enforcement."""

import logging
from datetime import datetime, timezone

from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)


class QuotaExhaustedError(Exception):
    """Raised when daily API quota is nearly exhausted."""
    pass


class QuotaTracker:
    """Track API quota usage in the database and halt when near limit."""

    def __init__(self, daily_limit: int = 10000, safety_margin: int = 500):
        self.daily_limit = daily_limit
        self.safety_margin = safety_margin

    def log_call(
        self,
        api_name: str,
        endpoint: str,
        quota_cost: int,
        details: dict | None = None,
    ):
        """Log an API call with its quota cost."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO api_quota_log (api_name, endpoint, quota_cost, details)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (api_name, endpoint, quota_cost, details),
                )
                conn.commit()

    def get_daily_usage(self, api_name: str = "youtube") -> int:
        """Get total quota used today (UTC)."""
        today = datetime.now(timezone.utc).date()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(quota_cost), 0)
                    FROM api_quota_log
                    WHERE api_name = %s
                      AND logged_at >= %s
                    """,
                    (api_name, today),
                )
                return cur.fetchone()[0]

    def get_remaining(self, api_name: str = "youtube") -> int:
        """Get remaining quota for today."""
        return self.daily_limit - self.get_daily_usage(api_name)

    def check_or_halt(self, required_cost: int, api_name: str = "youtube"):
        """Check if there's enough quota; raise QuotaExhaustedError if not."""
        remaining = self.get_remaining(api_name)
        if remaining < self.safety_margin:
            raise QuotaExhaustedError(
                f"Daily {api_name} quota nearly exhausted: {remaining} units remaining "
                f"(safety margin: {self.safety_margin}). Halting."
            )
        if remaining < required_cost:
            raise QuotaExhaustedError(
                f"Not enough {api_name} quota: {remaining} remaining, "
                f"{required_cost} required. Halting."
            )
        logger.debug(f"Quota check passed: {remaining} remaining for {api_name}")
