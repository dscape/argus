"""Service layer for crawl management — channels, videos, title scoring, AI classification."""

import logging
import os
from datetime import datetime, timezone

from pipeline.db.connection import get_conn
from pipeline.screen.title_filter import score_title

logger = logging.getLogger(__name__)


# ── Channels ────────────────────────────────────────────────


def list_channels() -> list[dict]:
    """List all channels with video counts."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.channel_id, c.channel_handle, c.channel_name,
                       c.tier, c.enabled, c.last_crawled_at, c.notes,
                       c.uploads_playlist_id,
                       COUNT(v.video_id) AS video_count
                FROM crawl_channels c
                LEFT JOIN youtube_videos v ON v.channel_id = c.channel_id
                GROUP BY c.channel_id
                ORDER BY c.tier ASC, c.channel_name ASC
                """
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_channel_detail(channel_id: str) -> dict | None:
    """Get channel info with per-status video counts."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM crawl_channels WHERE channel_id = %s",
                (channel_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            channel = dict(zip(cols, row))

            cur.execute(
                """
                SELECT screening_status, COUNT(*) AS cnt
                FROM youtube_videos
                WHERE channel_id = %s
                GROUP BY screening_status
                """,
                (channel_id,),
            )
            status_counts = {}
            for status, cnt in cur.fetchall():
                key = status if status else "unscreened"
                status_counts[key] = cnt
            channel["status_counts"] = status_counts
            channel["video_count"] = sum(status_counts.values())
            return channel


def add_channel(handle: str) -> dict:
    """Resolve a YouTube handle and insert into crawl_channels."""
    from pipeline.crawl.youtube_client import YouTubeClient
    from pipeline.crawl.quota_tracker import QuotaTracker

    client = YouTubeClient(quota_tracker=QuotaTracker())
    info = client.get_channel_by_handle(handle)
    if not info:
        raise ValueError(f"Could not resolve channel handle: {handle}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO crawl_channels
                    (channel_id, channel_handle, channel_name,
                     uploads_playlist_id, tier, enabled)
                VALUES (%s, %s, %s, %s, 3, true)
                ON CONFLICT (channel_id) DO UPDATE
                    SET channel_handle = EXCLUDED.channel_handle,
                        channel_name = EXCLUDED.channel_name,
                        uploads_playlist_id = EXCLUDED.uploads_playlist_id
                RETURNING *
                """,
                (
                    info["channel_id"],
                    handle if handle.startswith("@") else f"@{handle}",
                    info["title"],
                    info["uploads_playlist_id"],
                ),
            )
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            conn.commit()
            result = dict(zip(cols, row))
            result["video_count"] = 0
            return result


def toggle_channel(channel_id: str, enabled: bool) -> dict | None:
    """Enable or disable a channel."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE crawl_channels SET enabled = %s
                WHERE channel_id = %s RETURNING *
                """,
                (enabled, channel_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            conn.commit()
            return dict(zip(cols, row))


# ── Crawling ────────────────────────────────────────────────


def crawl_single_channel(channel_id: str) -> dict:
    """Trigger a crawl for one channel. Returns new video count."""
    from pipeline.crawl.youtube_client import YouTubeClient
    from pipeline.crawl.quota_tracker import QuotaTracker
    from pipeline.crawl.crawl_videos import crawl_channel

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT channel_id, channel_handle, uploads_playlist_id, last_crawled_at
                FROM crawl_channels
                WHERE channel_id = %s AND uploads_playlist_id IS NOT NULL
                """,
                (channel_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Channel {channel_id} not found or missing playlist")

    ch_id, ch_handle, playlist_id, last_crawled = row
    quota = QuotaTracker()
    client = YouTubeClient(quota_tracker=quota)

    new_count = crawl_channel(
        client=client,
        channel_id=ch_id,
        playlist_id=playlist_id,
        channel_handle=ch_handle,
        refresh=True,
        last_crawled_at=last_crawled,
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE crawl_channels SET last_crawled_at = %s WHERE channel_id = %s",
                (datetime.now(timezone.utc), ch_id),
            )
            conn.commit()

    return {"channel_id": ch_id, "new_videos": new_count}


def crawl_all_channels() -> dict:
    """Crawl all enabled channels. Returns summary."""
    from pipeline.crawl.youtube_client import YouTubeClient
    from pipeline.crawl.quota_tracker import QuotaTracker
    from pipeline.crawl.crawl_videos import crawl_channel

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT channel_id, channel_handle, uploads_playlist_id, last_crawled_at
                FROM crawl_channels
                WHERE enabled = true
                  AND uploads_playlist_id IS NOT NULL
                  AND channel_id NOT LIKE 'UNRESOLVED:%%'
                ORDER BY tier ASC, last_crawled_at ASC NULLS FIRST
                """
            )
            channels = cur.fetchall()

    if not channels:
        return {"channels_crawled": 0, "total_new_videos": 0}

    quota = QuotaTracker()
    client = YouTubeClient(quota_tracker=quota)
    total_new = 0
    crawled = 0

    for ch_id, ch_handle, playlist_id, last_crawled in channels:
        try:
            new_count = crawl_channel(
                client=client,
                channel_id=ch_id,
                playlist_id=playlist_id,
                channel_handle=ch_handle,
                refresh=True,
                last_crawled_at=last_crawled,
            )
            total_new += new_count
            crawled += 1

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE crawl_channels SET last_crawled_at = %s WHERE channel_id = %s",
                        (datetime.now(timezone.utc), ch_id),
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Error crawling {ch_handle or ch_id}: {e}")
            if "quota" in str(e).lower():
                break

    return {"channels_crawled": crawled, "total_new_videos": total_new}


# ── Videos ──────────────────────────────────────────────────


def get_video_counts(channel_id: str | None = None) -> dict:
    """Get video counts grouped by screening status."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            where = "WHERE channel_id = %s" if channel_id else ""
            params = [channel_id] if channel_id else []

            cur.execute(
                f"""
                SELECT screening_status, COUNT(*) AS cnt
                FROM youtube_videos
                {where}
                GROUP BY screening_status
                """,
                params,
            )
            counts: dict[str, int] = {}
            total = 0
            for status, cnt in cur.fetchall():
                key = status if status else "unscreened"
                counts[key] = cnt
                total += cnt
            counts["all"] = total
            return counts


def list_videos(
    channel_id: str | None = None,
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List videos with live title scoring."""
    conditions = []
    params: list = []

    if channel_id:
        conditions.append("channel_id = %s")
        params.append(channel_id)

    if status_filter == "unscreened":
        conditions.append("screening_status IS NULL")
    elif status_filter:
        conditions.append("screening_status = %s")
        params.append(status_filter)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM youtube_videos {where}",
                params,
            )
            total = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT video_id, channel_id, channel_handle, title,
                       description, published_at, screening_status,
                       screening_confidence
                FROM youtube_videos {where}
                ORDER BY published_at DESC NULLS LAST
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    # Score titles and auto-reject low-scoring unscreened videos
    auto_reject_ids: list[str] = []
    kept_rows: list[dict] = []

    for row in rows:
        is_candidate, confidence = score_title(row["title"])
        row["title_score"] = round(confidence, 3)
        row["title_is_candidate"] = is_candidate

        if row["screening_status"] is None and not is_candidate:
            auto_reject_ids.append(row["video_id"])
        else:
            kept_rows.append(row)

    # Persist auto-rejections
    if auto_reject_ids:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = 'rejected', updated_at = now()
                    WHERE video_id = ANY(%s)
                    """,
                    (auto_reject_ids,),
                )
                conn.commit()

    # When viewing unscreened, return only kept rows and adjust total
    if status_filter == "unscreened" and auto_reject_ids:
        total -= len(auto_reject_ids)
        return {"videos": kept_rows, "total": total}

    return {"videos": rows, "total": total}


def update_video_status(
    video_id: str, status: str | None, layout_type: str | None = None
) -> dict | None:
    """Set screening_status and optionally layout_type for a video."""
    if status is not None and status not in ("approved", "rejected"):
        raise ValueError(f"Invalid status: {status}")
    if layout_type is not None and layout_type not in ("overlay", "otb_only"):
        raise ValueError(f"Invalid layout_type: {layout_type}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            if layout_type is not None:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = %s, layout_type = %s, updated_at = now()
                    WHERE video_id = %s
                    RETURNING video_id, title, screening_status
                    """,
                    (status, layout_type, video_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = %s, updated_at = now()
                    WHERE video_id = %s
                    RETURNING video_id, title, screening_status
                    """,
                    (status, video_id),
                )
            row = cur.fetchone()
            if not row:
                return None
            conn.commit()
            return {"video_id": row[0], "title": row[1], "screening_status": row[2]}


def batch_update_status(video_ids: list[str], status: str) -> int:
    """Batch update screening_status. Returns count of updated rows."""
    if status not in ("approved", "rejected"):
        raise ValueError(f"Invalid status: {status}")
    if not video_ids:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE youtube_videos
                SET screening_status = %s, updated_at = now()
                WHERE video_id = ANY(%s)
                """,
                (status, video_ids),
            )
            conn.commit()
            return cur.rowcount


# ── Screening ──────────────────────────────────────────────


def screen_videos(channel_id: str | None = None, limit: int = 500) -> dict:
    """Run title-based screening on unscreened videos.

    Applies score_title() to each unscreened video and marks it as
    'candidate' (score >= 0.3) or 'rejected' (below threshold).
    """
    conditions = ["screening_status IS NULL"]
    params: list = []

    if channel_id:
        conditions.append("channel_id = %s")
        params.append(channel_id)

    where = "WHERE " + " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT video_id, title
                FROM youtube_videos
                {where}
                ORDER BY published_at DESC
                LIMIT %s
                """,
                params + [limit],
            )
            videos = cur.fetchall()

    if not videos:
        return {"screened": 0, "candidates": 0, "rejected": 0}

    candidates_ids: list[tuple[str, float]] = []
    rejected_ids: list[str] = []

    for video_id, title in videos:
        is_candidate, confidence = score_title(title)
        if is_candidate:
            candidates_ids.append((video_id, confidence))
        else:
            rejected_ids.append(video_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            if candidates_ids:
                for vid, conf in candidates_ids:
                    cur.execute(
                        """
                        UPDATE youtube_videos
                        SET screening_status = 'candidate',
                            screening_confidence = %s,
                            updated_at = now()
                        WHERE video_id = %s
                        """,
                        (conf, vid),
                    )
            if rejected_ids:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = 'rejected',
                        screening_confidence = 0,
                        updated_at = now()
                    WHERE video_id = ANY(%s)
                    """,
                    (rejected_ids,),
                )
            conn.commit()

    return {
        "screened": len(videos),
        "candidates": len(candidates_ids),
        "rejected": len(rejected_ids),
    }


# ── Quota ───────────────────────────────────────────────────


def get_quota_status() -> dict:
    """Get current YouTube API quota info."""
    from pipeline.crawl.quota_tracker import QuotaTracker

    qt = QuotaTracker()
    usage = qt.get_daily_usage()
    remaining = qt.get_remaining()
    return {
        "daily_usage": usage,
        "remaining": remaining,
        "daily_limit": qt.daily_limit,
    }


# ── AI Classification ──────────────────────────────────────


def classify_titles(video_ids: list[str]) -> str:
    """Use Claude to analyze training-positive titles and generate classification rules."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    # Fetch positive titles
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, channel_handle
                FROM youtube_videos
                WHERE video_id = ANY(%s)
                """,
                (video_ids,),
            )
            positive_titles = [
                f"{row[0]}  (channel: {row[1] or 'unknown'})"
                for row in cur.fetchall()
            ]

            # Fetch negative examples
            cur.execute(
                """
                SELECT title FROM youtube_videos
                WHERE screening_status = 'rejected'
                ORDER BY RANDOM() LIMIT 50
                """
            )
            rejected = [row[0] for row in cur.fetchall()]

            cur.execute(
                """
                SELECT title FROM youtube_videos
                WHERE screening_status IS NULL
                ORDER BY RANDOM() LIMIT 50
                """
            )
            unscreened = [row[0] for row in cur.fetchall()]

    negative_titles = rejected + unscreened

    positive_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(positive_titles))
    negative_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(negative_titles))

    user_prompt = f"""## Positive examples (titles confirmed to have OTB chess footage with overlay)
{positive_list}

## Negative examples (titles that do NOT have usable footage)
{negative_list}

Based on these examples, write a concise set of classification rules (as a prompt) \
that could be given to an AI to classify new YouTube video titles as likely containing \
OTB chess footage with a 2D board overlay or not. The rules should capture:
- What patterns indicate a positive match
- What patterns indicate a negative match
- Edge cases and how to handle them

Output ONLY the classification prompt text, nothing else."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system="You are an expert at classifying YouTube video titles. "
               "You will analyze examples of titles that contain OTB (over-the-board) "
               "chess footage with a 2D board overlay vs titles that do not.",
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def auto_classify_titles(
    channel_id: str | None = None, limit: int = 200
) -> dict:
    """Use Claude to batch-classify unscreened video titles.

    Sends titles to Claude with approved examples as context and asks for
    a structured classification. Updates screening_status in the database.
    """
    import json as _json
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    # Fetch positive examples
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, channel_handle
                FROM youtube_videos
                WHERE screening_status = 'approved'
                ORDER BY published_at DESC
                LIMIT 100
                """
            )
            positives = [
                f"{row[0]}  (channel: {row[1] or 'unknown'})"
                for row in cur.fetchall()
            ]

            # Fetch unscreened titles to classify
            conditions = ["screening_status IS NULL"]
            params: list = []
            if channel_id:
                conditions.append("channel_id = %s")
                params.append(channel_id)
            where = "WHERE " + " AND ".join(conditions)

            cur.execute(
                f"""
                SELECT video_id, title, channel_handle
                FROM youtube_videos
                {where}
                ORDER BY published_at DESC
                LIMIT %s
                """,
                params + [limit],
            )
            to_classify = [
                {"video_id": row[0], "title": row[1], "channel": row[2] or "unknown"}
                for row in cur.fetchall()
            ]

    if not to_classify:
        return {"classified": 0, "candidates": 0, "rejected": 0}

    if not positives:
        raise ValueError(
            "No approved videos to use as examples. Approve some videos first."
        )

    positive_list = "\n".join(f"  - {t}" for t in positives[:50])
    titles_list = "\n".join(
        f"  {i+1}. [{v['video_id']}] {v['title']}  (channel: {v['channel']})"
        for i, v in enumerate(to_classify)
    )

    user_prompt = f"""## Positive examples (titles confirmed to have OTB chess footage with overlay)
{positive_list}

## Titles to classify
{titles_list}

Classify each title as "candidate" (likely OTB chess game footage with board overlay) \
or "rejected" (unlikely to have usable footage).

Respond with a JSON array only, no other text:
[{{"id": "VIDEO_ID", "status": "candidate|rejected"}}]"""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            "You are an expert at classifying YouTube video titles for OTB "
            "(over-the-board) chess footage with 2D board overlay. You return valid JSON only.\n\n"
            "POSITIVE indicators:\n"
            "- Tournament names with years in format: '[Player] vs [Player] || [Tournament] ([Year])'\n"
            "- Formal chess event references (Candidates, World Championship, named tournaments)\n"
            "- Classical game analysis format with opponent names and tournament context\n"
            "- Titles suggesting serious competitive chess coverage\n\n"
            "NEGATIVE indicators:\n"
            "- Sensationalized language (ALL CAPS, multiple exclamation marks, 'DESTROYS', 'INSANE')\n"
            "- Casual/meme language ('rizz', 'cursed', reaction content)\n"
            "- First-person references ('I got destroyed', 'MY MOVES', 'Join me')\n"
            "- Clickbait phrases ('You won't believe', 'Wait for it', 'Do you see it?')\n"
            "- Player reaction focus ('slams table', 'pushes camera')\n"
            "- Educational/tutorial framing ('How To Win', 'Secret Rule')\n"
            "- Streaming/live content references (lichess.org, 'blitz or bullet')\n"
            "- Vague dramatic titles without specific game context\n"
            "- Questions without tournament context\n"
            "- Personal commentary format\n\n"
            "If title contains formal tournament structure with specific players, venue, and year, "
            "classify as 'candidate'. Otherwise, classify as 'rejected'.\n"
            "Edge cases: Titles mentioning famous players but without tournament context should be "
            "classified as 'rejected' unless they follow the formal game analysis format."
        ),
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Parse response
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    classifications = _json.loads(raw)

    # Apply classifications
    candidate_ids = [c["id"] for c in classifications if c["status"] == "candidate"]
    rejected_ids = [c["id"] for c in classifications if c["status"] == "rejected"]

    with get_conn() as conn:
        with conn.cursor() as cur:
            if candidate_ids:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = 'candidate', updated_at = now()
                    WHERE video_id = ANY(%s) AND screening_status IS NULL
                    """,
                    (candidate_ids,),
                )
            if rejected_ids:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = 'rejected', updated_at = now()
                    WHERE video_id = ANY(%s) AND screening_status IS NULL
                    """,
                    (rejected_ids,),
                )
            conn.commit()

    return {
        "classified": len(classifications),
        "candidates": len(candidate_ids),
        "rejected": len(rejected_ids),
    }
