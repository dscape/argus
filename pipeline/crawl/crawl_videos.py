"""Crawl YouTube channels and store video metadata."""

import json
import logging
from datetime import datetime, timezone

from pipeline.crawl.youtube_client import YouTubeClient
from pipeline.crawl.quota_tracker import QuotaTracker
from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)


def crawl_channel(
    client: YouTubeClient,
    channel_id: str,
    playlist_id: str,
    channel_handle: str | None,
    refresh: bool = False,
    last_crawled_at: datetime | None = None,
) -> int:
    """Crawl all videos from a channel's uploads playlist.

    Returns the number of new videos inserted.
    """
    total_new = 0
    page_token = None

    while True:
        response = client.list_playlist_items(
            playlist_id=playlist_id,
            page_token=page_token,
        )

        # Store raw response
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO youtube_api_raw
                        (channel_id, playlist_id, page_token, response_json)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (channel_id, playlist_id, page_token, json.dumps(response)),
                )
                conn.commit()

        # Parse and store videos
        items = response.get("items", [])
        if not items:
            break

        new_in_page = _store_videos(items, channel_id, channel_handle)
        total_new += new_in_page

        # If refreshing, stop when we hit videos we already have
        if refresh and new_in_page == 0:
            logger.info(f"Refresh: no new videos on this page, stopping.")
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return total_new


def _store_videos(
    items: list[dict],
    channel_id: str,
    channel_handle: str | None,
) -> int:
    """Parse playlist items and upsert into youtube_videos. Returns count of new rows."""
    new_count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for item in items:
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})

                video_id = content_details.get("videoId")
                if not video_id:
                    # Some items may be deleted/private
                    continue

                title = snippet.get("title", "")
                description = snippet.get("description", "")
                published_at = snippet.get("publishedAt")
                tags = snippet.get("tags", [])

                cur.execute(
                    """
                    INSERT INTO youtube_videos
                        (video_id, channel_id, channel_handle, title,
                         description, published_at, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (video_id) DO NOTHING
                    RETURNING video_id
                    """,
                    (video_id, channel_id, channel_handle, title,
                     description, published_at, tags or None),
                )
                if cur.fetchone() is not None:
                    new_count += 1

            conn.commit()
    return new_count


def crawl_all(
    channel_handle: str | None = None,
    refresh: bool = False,
):
    """Crawl all enabled channels (or a specific one).

    Args:
        channel_handle: If provided, only crawl this channel.
        refresh: If True, only fetch new videos since last crawl.
    """
    quota = QuotaTracker()
    client = YouTubeClient(quota_tracker=quota)

    with get_conn() as conn:
        with conn.cursor() as cur:
            if channel_handle:
                cur.execute(
                    """
                    SELECT channel_id, channel_handle, channel_name,
                           uploads_playlist_id, last_crawled_at
                    FROM crawl_channels
                    WHERE (channel_handle = %s OR channel_handle = %s)
                      AND enabled = true
                      AND uploads_playlist_id IS NOT NULL
                    """,
                    (channel_handle, channel_handle.lstrip("@")),
                )
            else:
                cur.execute(
                    """
                    SELECT channel_id, channel_handle, channel_name,
                           uploads_playlist_id, last_crawled_at
                    FROM crawl_channels
                    WHERE enabled = true
                      AND uploads_playlist_id IS NOT NULL
                      AND channel_id NOT LIKE 'UNRESOLVED:%%'
                    ORDER BY tier ASC, last_crawled_at ASC NULLS FIRST
                    """
                )
            channels = cur.fetchall()

    if not channels:
        print("No channels to crawl. Run resolve-channels first.")
        return

    print(f"Crawling {len(channels)} channels (refresh={refresh})")
    total_videos = 0

    for ch_id, ch_handle, ch_name, playlist_id, last_crawled in channels:
        print(f"\n{'='*60}")
        print(f"Crawling: {ch_name} ({ch_handle or ch_id})")

        remaining = quota.get_remaining()
        print(f"  Quota remaining: {remaining}")

        try:
            new_count = crawl_channel(
                client=client,
                channel_id=ch_id,
                playlist_id=playlist_id,
                channel_handle=ch_handle,
                refresh=refresh,
                last_crawled_at=last_crawled,
            )
            total_videos += new_count
            print(f"  New videos: {new_count}")

            # Update last_crawled_at
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE crawl_channels
                        SET last_crawled_at = %s
                        WHERE channel_id = %s
                        """,
                        (datetime.now(timezone.utc), ch_id),
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"Error crawling {ch_name}: {e}")
            print(f"  ERROR: {e}")
            if "quota" in str(e).lower():
                print("Quota exhausted. Stopping all crawls.")
                break

    print(f"\n{'='*60}")
    print(f"Total new videos: {total_videos}")
    print(f"Quota remaining: {quota.get_remaining()}")
