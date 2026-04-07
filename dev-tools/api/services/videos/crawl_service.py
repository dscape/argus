"""Service layer for crawl management — channels and videos."""

import logging
import os
from datetime import datetime, timezone

from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)


# ── Channels ────────────────────────────────────────────────


def list_channels(screened_only: bool = False) -> list[dict]:
    """List all channels with video counts.

    If screened_only is True, only count approved/rejected videos and
    exclude channels with zero screened videos.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if screened_only:
                cur.execute(
                    """
                    SELECT c.channel_id, c.channel_handle, c.channel_name,
                           c.tier, c.enabled, c.last_crawled_at, c.notes,
                           c.uploads_playlist_id,
                           COUNT(v.video_id) AS video_count
                    FROM crawl_channels c
                    INNER JOIN youtube_videos v
                        ON v.channel_id = c.channel_id
                        AND v.screening_status IN ('approved', 'rejected')
                    GROUP BY c.channel_id
                    ORDER BY c.tier ASC, c.channel_name ASC
                    """
                )
            else:
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
    from pipeline.crawl.quota_tracker import QuotaTracker
    from pipeline.crawl.youtube_client import YouTubeClient

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
    from pipeline.crawl.crawl_videos import crawl_channel
    from pipeline.crawl.quota_tracker import QuotaTracker
    from pipeline.crawl.youtube_client import YouTubeClient

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
    from pipeline.crawl.crawl_videos import crawl_channel
    from pipeline.crawl.quota_tracker import QuotaTracker
    from pipeline.crawl.youtube_client import YouTubeClient

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


def fetch_frames_for_channel(channel_id: str, hires: bool = True) -> dict:
    """Fetch frames for approved overlay videos in a channel.

    Approved videos with ``layout_type IS NULL`` are treated as overlay.
    """
    from pipeline.screen.frame_fetcher import fetch_overlay_frames

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT video_id FROM youtube_videos
                   WHERE channel_id = %s
                     AND screening_status = 'approved'
                     AND (layout_type = 'overlay' OR layout_type IS NULL)
                   ORDER BY video_id""",
                (channel_id,),
            )
            video_ids = [row[0] for row in cur.fetchall()]

    if not video_ids:
        return {"channel_id": channel_id, "videos_processed": 0, "frames_fetched": 0}

    total_frames = 0
    for vid in video_ids:
        results = fetch_overlay_frames(vid, hires=hires)
        total_frames += len(results)

    return {
        "channel_id": channel_id,
        "videos_processed": len(video_ids),
        "frames_fetched": total_frames,
    }


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
    order_by: str | None = None,
    layout_type: str | None = None,
    video_ids: list[str] | None = None,
) -> dict:
    """List videos with live title scoring."""
    conditions = []
    params: list = []

    if channel_id:
        conditions.append("channel_id = %s")
        params.append(channel_id)

    if status_filter == "unscreened":
        conditions.append("screening_status IS NULL")
    elif status_filter == "screened":
        conditions.append("screening_status IN ('approved', 'rejected')")
    elif status_filter:
        conditions.append("screening_status = %s")
        params.append(status_filter)

    if layout_type:
        if layout_type.startswith("!"):
            # Negation: e.g. "!otb_only" means layout_type != 'otb_only' OR IS NULL
            conditions.append("(layout_type IS NULL OR layout_type != %s)")
            params.append(layout_type[1:])
        else:
            conditions.append("layout_type = %s")
            params.append(layout_type)

    if video_ids:
        placeholders = ", ".join(["%s"] * len(video_ids))
        conditions.append(f"video_id IN ({placeholders})")
        params.extend(video_ids)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Determine ORDER BY clause
    if order_by == "channel_name":
        order_clause = "ORDER BY channel_handle ASC NULLS LAST, published_at DESC NULLS LAST"
    else:
        order_clause = "ORDER BY published_at DESC NULLS LAST"

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
                       screening_confidence, layout_type, screened_by
                FROM youtube_videos {where}
                {order_clause}
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]

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
            screened_by = "human" if status is not None else None
            if layout_type is not None:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = %s, layout_type = %s,
                        screened_by = %s, updated_at = now()
                    WHERE video_id = %s
                    RETURNING video_id, title, screening_status
                    """,
                    (status, layout_type, screened_by, video_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET screening_status = %s, screened_by = %s,
                        updated_at = now()
                    WHERE video_id = %s
                    RETURNING video_id, title, screening_status
                    """,
                    (status, screened_by, video_id),
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
                SET screening_status = %s, screened_by = 'human',
                    updated_at = now()
                WHERE video_id = ANY(%s)
                """,
                (status, video_ids),
            )
            conn.commit()
            return cur.rowcount


def undo_auto_rejections(video_ids: list[str]) -> int:
    """Reset auto-rejected videos back to unscreened (NULL status)."""
    if not video_ids:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE youtube_videos
                SET screening_status = NULL, screened_by = NULL,
                    updated_at = now()
                WHERE video_id = ANY(%s)
                  AND screening_status = 'rejected'
                """,
                (video_ids,),
            )
            conn.commit()
            return cur.rowcount


def get_correction_stats() -> dict:
    """Return counts of human vs AI screening decisions."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE screening_status IS NOT NULL) AS total_labeled,
                    COUNT(*) FILTER (WHERE screened_by = 'human') AS total_human,
                    COUNT(*) FILTER (WHERE screened_by = 'ai') AS total_ai,
                    COUNT(*) FILTER (
                        WHERE screened_by = 'human' AND ai_screening_auto_decided = true
                    ) AS corrections
                FROM youtube_videos
            """)
            row = cur.fetchone()
            return {
                "total_labeled": row[0],
                "total_human": row[1],
                "total_ai": row[2],
                "corrections": row[3],
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


# ── Annotations ───────────────────────────────────────────


def get_video(video_id: str) -> dict | None:
    """Get a single video by ID with title scoring."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, channel_id, channel_handle, title,
                       description, published_at, screening_status,
                       screening_confidence, layout_type, annotations,
                       screened_by
                FROM youtube_videos
                WHERE video_id = %s
                """,
                (video_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            video = dict(zip(cols, row))

    return video


def _project_root() -> str:
    """Resolve the project root directory.

    The pipeline uses relative paths like ``data/videos/...`` which assume
    CWD is the project root.  The FastAPI server, however, starts from
    ``<root>/dev-tools/``.  We detect the root by walking up from this
    file until we find the ``pipeline/`` package directory.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "pipeline")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def get_download_status(video_id: str) -> dict:
    """Check if a video file is downloaded and return its status."""
    from pipeline.paths import find_video_file

    path = find_video_file(video_id)

    if path is not None:
        import cv2

        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
        duration_seconds = None
        try:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and frame_count > 0:
                duration_seconds = round(frame_count / fps, 1)
            cap.release()
        except Exception:
            pass
        return {
            "downloaded": True, "path": str(path),
            "file_size_mb": size_mb,
            "duration_seconds": duration_seconds,
        }

    return {
        "downloaded": False, "path": None,
        "file_size_mb": None, "duration_seconds": None,
    }


def get_asset_status(video_id: str) -> dict:
    """Report which assets exist for a video."""
    from pipeline.paths import asset_status
    return asset_status(video_id)


def fetch_video_assets(video_id: str) -> dict:
    """Fetch lores + hires frames for a video. Returns counts."""
    from pipeline.screen.frame_fetcher import fetch_overlay_frames

    lores = fetch_overlay_frames(video_id, hires=False)
    hires = fetch_overlay_frames(video_id, hires=True)
    return {
        "video_id": video_id,
        "lores_fetched": len(lores),
        "hires_fetched": len(hires),
    }


def download_single_video(video_id: str) -> dict:
    """Download a single video. Returns download status."""
    import yt_dlp
    from pipeline.paths import video_file

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id, title FROM youtube_videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Video {video_id} not found")

    vid, title = row
    filepath = str(video_file(vid))
    output_dir = os.path.dirname(filepath)

    if os.path.exists(filepath):
        size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
        return {"status": "already_downloaded", "path": filepath, "file_size_mb": size_mb}

    os.makedirs(output_dir, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={vid}"
    ydl_opts = {
        # Prefer H.264 (avc1) so OpenCV can decode without extra codecs
        "format": (
            "bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best"
        ),
        "outtmpl": filepath.replace(".mp4", ".%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "ratelimit": 5 * 1024 * 1024,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise ValueError(f"Download failed: {e}")

    if os.path.exists(filepath):
        size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
        return {"status": "downloaded", "path": filepath, "file_size_mb": size_mb}

    raise ValueError("Download completed but file not found")


def get_video_annotations(video_id: str) -> dict | None:
    """Get annotations stored for a video."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id, annotations FROM youtube_videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"video_id": row[0], "annotations": row[1]}


def save_video_annotations(video_id: str, data: dict) -> dict:
    """Save annotations for a video as JSONB."""
    import json as _json

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE youtube_videos
                SET annotations = %s::jsonb, updated_at = now()
                WHERE video_id = %s
                RETURNING video_id
                """,
                (_json.dumps(data), video_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Video {video_id} not found")
            conn.commit()
            return {"video_id": video_id, "annotations": data}


# ── Video Clips ──────────────────────────────────────────


def list_video_clips(video_id: str) -> list[dict]:
    """List all clips for a video, ordered by clip_index."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, video_id, clip_index, label, start_time, end_time,
                       overlay_bbox, camera_bbox, ref_resolution,
                       board_flipped, board_theme, is_gap
                FROM video_clips
                WHERE video_id = %s
                ORDER BY clip_index
                """,
                (video_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def create_video_clip(video_id: str, data: dict) -> dict:
    """Create a new clip for a video. Assigns next clip_index, validates no overlap."""
    import json as _json

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Get next clip_index
            cur.execute(
                "SELECT COALESCE(MAX(clip_index), -1) + 1 FROM video_clips WHERE video_id = %s",
                (video_id,),
            )
            next_index = cur.fetchone()[0]

            # Validate no overlap with existing clips
            start = data.get("start_time", 0.0)
            end = data.get("end_time")
            _validate_no_overlap(cur, video_id, start, end, exclude_id=None)

            cur.execute(
                """
                INSERT INTO video_clips
                    (video_id, clip_index, label, start_time, end_time,
                     overlay_bbox, camera_bbox, ref_resolution,
                     board_flipped, board_theme, is_gap)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s)
                RETURNING id, video_id, clip_index, label, start_time, end_time,
                          overlay_bbox, camera_bbox, ref_resolution,
                          board_flipped, board_theme, is_gap
                """,
                (
                    video_id,
                    next_index,
                    data.get("label"),
                    start,
                    end,
                    _json.dumps(data["overlay_bbox"]),
                    _json.dumps(data["camera_bbox"]),
                    _json.dumps(data.get("ref_resolution", [1920, 1080])),
                    data.get("board_flipped", False),
                    data.get("board_theme", "lichess_default"),
                    data.get("is_gap", False),
                ),
            )
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            conn.commit()
            return dict(zip(cols, row))


def update_video_clip(clip_id: int, data: dict) -> dict:
    """Update an existing clip's fields."""
    import json as _json

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Fetch current clip to get video_id for overlap validation
            cur.execute(
                "SELECT video_id, start_time, end_time FROM video_clips WHERE id = %s",
                (clip_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise ValueError(f"Clip {clip_id} not found")

            video_id = existing[0]
            start = data.get("start_time", existing[1])
            end = data.get("end_time", existing[2])

            if "start_time" in data or "end_time" in data:
                _validate_no_overlap(cur, video_id, start, end, exclude_id=clip_id)

            sets = []
            params = []
            scalar_fields = [
                "label", "start_time", "end_time",
                "board_flipped", "board_theme", "is_gap",
            ]
            for field in scalar_fields:
                if field in data:
                    sets.append(f"{field} = %s")
                    params.append(data[field])
            for json_field in ["overlay_bbox", "camera_bbox", "ref_resolution"]:
                if json_field in data:
                    sets.append(f"{json_field} = %s::jsonb")
                    params.append(_json.dumps(data[json_field]))

            if not sets:
                raise ValueError("No fields to update")

            sets.append("updated_at = now()")
            params.append(clip_id)

            cur.execute(
                f"""
                UPDATE video_clips SET {', '.join(sets)}
                WHERE id = %s
                RETURNING id, video_id, clip_index, label, start_time, end_time,
                          overlay_bbox, camera_bbox, ref_resolution,
                          board_flipped, board_theme, is_gap
                """,
                params,
            )
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            conn.commit()
            return dict(zip(cols, row))


def delete_video_clip(clip_id: int) -> bool:
    """Delete a clip and reindex remaining clips for the same video."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id, clip_index FROM video_clips WHERE id = %s",
                (clip_id,),
            )
            row = cur.fetchone()
            if not row:
                return False

            video_id, deleted_index = row
            cur.execute("DELETE FROM video_clips WHERE id = %s", (clip_id,))

            # Reindex remaining clips — use negative offset first
            # to avoid unique constraint violations during update
            cur.execute(
                """
                UPDATE video_clips
                SET clip_index = -clip_index - 1000, updated_at = now()
                WHERE video_id = %s AND clip_index > %s
                """,
                (video_id, deleted_index),
            )
            cur.execute(
                """
                UPDATE video_clips
                SET clip_index = -clip_index - 1001
                WHERE video_id = %s AND clip_index < 0
                """,
                (video_id,),
            )
            conn.commit()
            return True


def get_video_clip(clip_id: int) -> dict | None:
    """Get a single clip by ID."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, video_id, clip_index, label, start_time, end_time,
                       overlay_bbox, camera_bbox, ref_resolution,
                       board_flipped, board_theme, is_gap
                FROM video_clips
                WHERE id = %s
                """,
                (clip_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def _validate_no_overlap(cur, video_id: str, start: float, end: float | None, exclude_id: int | None):
    """Raise ValueError if the time range overlaps with existing clips."""
    if exclude_id is not None:
        cur.execute(
            """
            SELECT clip_index, start_time, end_time FROM video_clips
            WHERE video_id = %s AND id != %s
            ORDER BY clip_index
            """,
            (video_id, exclude_id),
        )
    else:
        cur.execute(
            """
            SELECT clip_index, start_time, end_time FROM video_clips
            WHERE video_id = %s
            ORDER BY clip_index
            """,
            (video_id,),
        )
    for idx, s, e in cur.fetchall():
        # Two ranges overlap if start < other_end AND end > other_start
        other_end = e if e is not None else float("inf")
        new_end = end if end is not None else float("inf")
        if start < other_end and new_end > s:
            raise ValueError(f"Overlaps with clip {idx} ({s}s - {e or 'end'}s)")
