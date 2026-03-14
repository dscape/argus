"""Resolve YouTube channel handles to channel IDs and uploads playlist IDs."""

import logging

from pipeline.crawl.youtube_client import YouTubeClient
from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)


def resolve_channels(client: YouTubeClient | None = None):
    """Resolve all unresolved channels in crawl_channels.

    For channels with a known channel_id (UC-prefixed), fetches the uploads
    playlist ID. For channels with only a handle, resolves both.
    """
    if client is None:
        client = YouTubeClient()

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Find channels needing resolution
            cur.execute(
                """
                SELECT channel_id, channel_handle, channel_name
                FROM crawl_channels
                WHERE enabled = true
                  AND (uploads_playlist_id IS NULL
                       OR channel_id LIKE 'UNRESOLVED:%%')
                ORDER BY tier ASC
                """
            )
            channels = cur.fetchall()

    if not channels:
        print("All channels already resolved.")
        return

    print(f"Resolving {len(channels)} channels...")
    resolved = 0
    failed = []

    for channel_id, handle, name in channels:
        info = None

        # If we have a real channel ID, use it directly
        if channel_id and channel_id.startswith("UC"):
            info = client.get_channel_by_id(channel_id)
            if info:
                logger.info(f"Resolved by ID: {name} -> {info['uploads_playlist_id']}")
        # If we have a handle, resolve it
        elif handle:
            info = client.get_channel_by_handle(handle)
            if info:
                logger.info(f"Resolved by handle: {handle} -> {info['channel_id']}")
        else:
            # No handle and no real channel_id — needs manual search
            logger.warning(
                f"Channel '{name}' has no handle or channel_id. "
                "Use 'search_channels' to find it manually."
            )
            failed.append(name)
            continue

        if info is None:
            logger.warning(f"Failed to resolve channel: {name} (handle: {handle})")
            failed.append(name)
            continue

        # Update the database
        with get_conn() as conn:
            with conn.cursor() as cur:
                if channel_id and channel_id.startswith("UNRESOLVED:"):
                    # Replace placeholder with real channel_id
                    cur.execute(
                        "DELETE FROM crawl_channels WHERE channel_id = %s",
                        (channel_id,),
                    )
                    cur.execute(
                        """
                        INSERT INTO crawl_channels
                            (channel_id, channel_handle, channel_name, tier,
                             uploads_playlist_id, enabled)
                        VALUES (%s, %s, %s,
                                (SELECT tier FROM crawl_channels WHERE channel_id = %s),
                                %s, true)
                        ON CONFLICT (channel_id) DO UPDATE SET
                            uploads_playlist_id = EXCLUDED.uploads_playlist_id,
                            channel_handle = EXCLUDED.channel_handle
                        """,
                        (
                            info["channel_id"],
                            handle,
                            name,
                            channel_id,
                            info["uploads_playlist_id"],
                        ),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE crawl_channels
                        SET uploads_playlist_id = %s,
                            channel_id = %s
                        WHERE channel_id = %s
                        """,
                        (info["uploads_playlist_id"], info["channel_id"], channel_id),
                    )
                conn.commit()
        resolved += 1

    print(f"Resolved: {resolved}, Failed: {len(failed)}")
    if failed:
        print("Failed channels (need manual resolution):")
        for name in failed:
            print(f"  - {name}")
