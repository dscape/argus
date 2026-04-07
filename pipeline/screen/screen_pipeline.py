"""Orchestrate video screening: title filtering + frame-level detection.

Two-stage screening process:
1. Title filter (cheap): regex-based keyword matching to reject non-chess titles
2. Frame sampling (expensive): extract stills and detect overlay + OTB regions

Optional AI screening stage:
- Uses DINOv2 frozen features + overlay/OTB scanner scores
- High-confidence predictions are auto-decided
- Low-confidence predictions are left for manual review
"""

import logging

from pipeline.db.connection import get_conn
from pipeline.screen.dual_region_detector import (
    overlay_bbox_to_json,
    screen_video,
)

logger = logging.getLogger(__name__)


def screen_all(
    channel_handle: str | None = None,
    limit: int | None = None,
):
    """Screen crawled videos for overlay + OTB training suitability.

    Samples frames and detects overlay + OTB regions. Videos with both
    get 'approved'; others get 'rejected'.

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

    approved_count = 0
    rejected_frame = 0
    failed = 0

    for video_id, handle, title in videos:
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
        f"{rejected_frame} rejected, {failed} failed"
    )


def screen_with_ai(
    channel_handle: str | None = None,
    limit: int | None = None,
    checkpoint_path: str | None = None,
    threshold: float = 0.85,
    device: str = "cpu",
):
    """Screen unscreened videos using the AI classifier.

    Runs the trained DINOv2-based classifier on videos that have not yet
    been screened. High-confidence predictions are auto-decided; low-
    confidence ones are left for manual review.

    Args:
        channel_handle: If provided, only screen this channel's videos.
        limit: Maximum videos to process.
        checkpoint_path: Path to the trained classifier checkpoint.
        threshold: Confidence threshold for auto-deciding.
        device: Torch device for DINOv2 inference.
    """
    from pipeline.screen.ai_predict import apply_predictions, predict_batch

    # Fetch unscreened videos
    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT video_id
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
            rows = cur.fetchall()

    if not rows:
        print("No unscreened videos found.")
        return

    video_ids = [r[0] for r in rows]
    print(f"AI screening {len(video_ids)} videos (threshold={threshold})...")

    predictions = predict_batch(
        video_ids,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device=device,
    )

    summary = apply_predictions(predictions)
    print(
        f"\nAI screening complete: {summary['auto_decided']} auto-decided, "
        f"{summary['manual_review']} queued for manual review"
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
