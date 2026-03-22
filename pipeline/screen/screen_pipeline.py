"""Orchestrate video screening: title filtering + frame-level detection.

Two-stage screening process:
1. Title filter (cheap): regex-based keyword matching to identify candidates
2. Frame sampling (expensive): extract stills and detect overlay + OTB regions
"""

import logging

from pipeline.db.connection import get_conn
from pipeline.screen.dual_region_detector import (
    overlay_bbox_to_json,
    screen_video,
)
from pipeline.screen.title_filter import score_title

logger = logging.getLogger(__name__)


def screen_all(
    channel_handle: str | None = None,
    limit: int | None = None,
):
    """Screen crawled videos for overlay + OTB training suitability.

    Stage 1: Score titles with keyword matching. Videos that pass become
    'candidate' status.

    Stage 2: For candidates, sample frames and detect overlay + OTB regions.
    Videos with both get 'approved'; others get 'rejected'.

    Args:
        channel_handle: If provided, only screen videos from this channel.
        limit: Maximum number of videos to process.
    """
    # Fetch unscreened videos
    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT video_id, channel_handle, title
                FROM youtube_videos
                WHERE screening_status IS NULL
            """
            params: list = []

            if channel_handle:
                query += " AND channel_handle = %s"
                params.append(channel_handle)

            query += " ORDER BY published_at DESC"

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            videos = cur.fetchall()

    if not videos:
        print("No unscreened videos found.")
        return

    print(f"Screening {len(videos)} videos...")

    # Stage 1: Title filtering
    candidates = []
    rejected_title = 0

    for video_id, handle, title in videos:
        is_candidate, confidence = score_title(title)

        if is_candidate:
            candidates.append((video_id, handle, title, confidence))
        else:
            rejected_title += 1
            _update_screening_status(video_id, "rejected", confidence=0.0)

    print(
        f"Title filter: {len(candidates)} candidates, "
        f"{rejected_title} rejected"
    )

    if not candidates:
        return

    # Stage 2: Frame-level screening
    approved_count = 0
    rejected_frame = 0
    failed = 0

    for video_id, handle, title, title_confidence in candidates:
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            result = screen_video(url)

            if result.approved:
                _update_screening_status(
                    video_id,
                    status="approved",
                    confidence=result.otb_confidence,
                    overlay_bbox=overlay_bbox_to_json(result.overlay_bbox),
                    has_otb_footage=True,
                    layout_type="overlay",
                )
                approved_count += 1
                logger.info(
                    f"APPROVED: {video_id} "
                    f"(overlay={result.overlay_score:.2f}, "
                    f"otb={result.otb_confidence:.2f})"
                )
            else:
                layout = "otb_only" if not result.has_overlay else None
                _update_screening_status(
                    video_id,
                    status="rejected",
                    confidence=result.otb_confidence,
                    overlay_bbox=overlay_bbox_to_json(result.overlay_bbox),
                    has_otb_footage=result.has_otb,
                    layout_type=layout,
                )
                rejected_frame += 1

        except Exception as e:
            failed += 1
            logger.error(f"Failed to screen {video_id}: {e}")

    print(
        f"\nScreening complete: {approved_count} approved, "
        f"{rejected_frame} rejected (frame), {failed} failed"
    )


def _update_screening_status(
    video_id: str,
    status: str,
    confidence: float = 0.0,
    overlay_bbox: str | None = None,
    has_otb_footage: bool | None = None,
    layout_type: str | None = None,
):
    """Update screening results for a video in the database."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE youtube_videos
                SET screening_status = %s,
                    screening_confidence = %s,
                    overlay_bbox = %s,
                    has_otb_footage = %s,
                    layout_type = COALESCE(%s, layout_type),
                    updated_at = now()
                WHERE video_id = %s
                """,
                (status, confidence, overlay_bbox, has_otb_footage,
                 layout_type, video_id),
            )
            conn.commit()
