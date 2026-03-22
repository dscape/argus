"""Download approved YouTube videos using yt-dlp."""

import logging
import os
import time

import yt_dlp

from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join("data", "videos")


def download_approved_videos(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    delay: float = 2.0,
):
    """Download videos that passed screening (overlay + OTB approved).

    Args:
        output_dir: Base directory for downloaded videos.
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
        channel_dir = (channel_handle or "unknown").lstrip("@")
        filepath = os.path.join(output_dir, channel_dir, f"{video_id}.mp4")
        if not os.path.exists(filepath):
            to_download.append((video_id, channel_dir, title, filepath))

    if not to_download:
        print(f"All {len(videos)} approved videos already downloaded.")
        return

    print(
        f"Downloading {len(to_download)} videos "
        f"(skipping {len(videos) - len(to_download)} already downloaded)..."
    )

    downloaded = 0
    failed = 0

    for video_id, channel_dir, title, filepath in to_download:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"\n  [{downloaded + failed + 1}/{len(to_download)}] {title[:60]}...")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
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


def get_video_path(
    video_id: str,
    channel_handle: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str | None:
    """Get the local file path for a downloaded video, or None if not downloaded."""
    channel_dir = (channel_handle or "unknown").lstrip("@")
    filepath = os.path.join(output_dir, channel_dir, f"{video_id}.mp4")
    return filepath if os.path.exists(filepath) else None
