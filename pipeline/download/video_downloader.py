"""Download approved YouTube videos using yt-dlp."""

import logging
import os
import time

import yt_dlp

from pipeline.db.connection import get_conn
from pipeline.paths import find_video_file, video_file

logger = logging.getLogger(__name__)


def download_approved_videos(
    limit: int | None = None,
    delay: float = 2.0,
):
    """Download videos that passed screening (overlay + OTB approved).

    Args:
        limit: Maximum number of videos to download.
        delay: Seconds to wait between downloads.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT video_id, channel_handle, title
                FROM youtube_videos
                WHERE screening_status = 'approved'
                ORDER BY published_at DESC
            """
            params: list = []

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            videos = cur.fetchall()

    if not videos:
        print("No approved videos to download.")
        return

    # Filter out already downloaded
    to_download = []
    for video_id, channel_handle, title in videos:
        if find_video_file(video_id) is None:
            filepath = str(video_file(video_id))
            to_download.append((video_id, title, filepath))

    if not to_download:
        print(f"All {len(videos)} approved videos already downloaded.")
        return

    print(
        f"Downloading {len(to_download)} videos "
        f"(skipping {len(videos) - len(to_download)} already downloaded)..."
    )

    downloaded = 0
    failed = 0

    for video_id, title, filepath in to_download:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"\n  [{downloaded + failed + 1}/{len(to_download)}] {title[:60]}...")

        ydl_opts = {
            # Prefer single-stream mp4 to avoid needing ffmpeg for merging.
            "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "outtmpl": filepath.replace(".mp4", ".%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "ratelimit": 5 * 1024 * 1024,  # 5MB/s
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            downloaded += 1
            print(f"    Downloaded: {filepath}")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to download {video_id}: {e}")
            print(f"    FAILED: {e}")

        if delay > 0 and (downloaded + failed) < len(to_download):
            time.sleep(delay)

    print(f"\nDownload complete: {downloaded} succeeded, {failed} failed")


def download_diverse_overlay_videos(
    per_channel: int = 3,
    delay: float = 5.0,
):
    """Download overlay videos from diverse channels for validation testing.

    Picks up to *per_channel* random overlay videos from each channel that has
    layout_type='overlay' entries, skipping already-downloaded files.  Uses a
    longer inter-download delay and exponential backoff to avoid YouTube
    throttling.

    Args:
        per_channel: Max videos to download per channel.
        delay: Base seconds between downloads (increases on failure).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Use DISTINCT ON + random ordering within each channel
            cur.execute(
                """
                SELECT video_id, channel_handle, title
                FROM (
                    SELECT video_id, channel_handle, title,
                           ROW_NUMBER() OVER (
                               PARTITION BY channel_handle
                               ORDER BY random()
                           ) AS rn
                    FROM youtube_videos
                    WHERE layout_type = 'overlay'
                      AND channel_handle IS NOT NULL
                ) sub
                WHERE rn <= %s
                ORDER BY channel_handle, rn
                """,
                (per_channel,),
            )
            videos = cur.fetchall()

    if not videos:
        print("No overlay videos found in DB.")
        return

    # Filter out already downloaded
    to_download = []
    for video_id, channel_handle, title in videos:
        if find_video_file(video_id) is None:
            filepath = str(video_file(video_id))
            to_download.append((video_id, channel_handle, title, filepath))

    channels = {v[1] for v in to_download}
    already = len(videos) - len(to_download)
    if not to_download:
        print(f"All {len(videos)} overlay videos already downloaded.")
        return

    print(
        f"Downloading {len(to_download)} overlay videos from {len(channels)} "
        f"channels (skipping {already} already downloaded)..."
    )

    downloaded = 0
    failed = 0
    consecutive_fails = 0

    for video_id, channel_handle, title, filepath in to_download:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        url = f"https://www.youtube.com/watch?v={video_id}"
        ch_dir = (channel_handle or "unknown").lstrip("@")
        print(f"\n  [{downloaded + failed + 1}/{len(to_download)}] [{ch_dir}] {title[:50]}...")

        ydl_opts = {
            # Prefer single-stream mp4 to avoid needing ffmpeg for merging.
            # Falls back to best merged format if ffmpeg is available.
            "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "outtmpl": filepath.replace(".mp4", ".%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "ratelimit": 5 * 1024 * 1024,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            downloaded += 1
            consecutive_fails = 0
            print(f"    OK: {filepath}")
        except Exception as e:
            failed += 1
            consecutive_fails += 1
            logger.error(f"Failed to download {video_id}: {e}")
            print(f"    FAILED: {e}")

            # Exponential backoff on consecutive failures
            if consecutive_fails >= 3:
                backoff = min(60, delay * (2 ** (consecutive_fails - 2)))
                print(f"    Backing off {backoff:.0f}s "
                      f"after {consecutive_fails} fails...")
                time.sleep(backoff)
                continue

        if (downloaded + failed) < len(to_download):
            time.sleep(delay)

    print(
        f"\nDiverse download complete: {downloaded} succeeded, {failed} failed "
        f"across {len(channels)} channels"
    )


def get_video_path(video_id: str) -> str | None:
    """Get the local file path for a downloaded video, or None if not downloaded."""
    path = find_video_file(video_id)
    return str(path) if path is not None else None
