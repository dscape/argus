"""CLI-friendly workflows for video clip segmentation and calibration."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import cv2
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.overlay.auto_calibration import _get_video_path, inspect_clip_calibration
from pipeline.overlay.segmenter import segment_video_layouts

logger = logging.getLogger(__name__)


def auto_segment_video(
    video_id: str,
    sample_interval_sec: float = 30.0,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Create ``video_clips`` rows by segmenting a downloaded video."""
    video_path = _get_video_path(video_id)
    if not video_path:
        raise ValueError(f"Video {video_id} is not downloaded")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    existing_clips = _list_video_clips(video_id)
    if existing_clips and not replace_existing:
        return {
            "error": f"Video already has {len(existing_clips)} clip(s). "
            "Set replace_existing=true to replace them.",
            "existing_clips": len(existing_clips),
        }

    if replace_existing:
        for clip in existing_clips:
            _delete_video_clip(int(clip["id"]))

    started_at = time.monotonic()
    segments, gaps = segment_video_layouts(video_path, sample_interval_sec)
    elapsed = time.monotonic() - started_at

    created_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        created = _create_video_clip(
            video_id,
            {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "label": f"Segment {index + 1}",
                "overlay_bbox": list(segment.overlay_bbox)
                if segment.overlay_bbox
                else [0, 0, 100, 100],
                "camera_bbox": [0, 0, 100, 100],
                "ref_resolution": [width, height],
                "board_flipped": False,
                "board_theme": "lichess_default",
            },
        )
        created_segments.append(
            {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "overlay_bbox": list(segment.overlay_bbox) if segment.overlay_bbox else None,
                "score": round(segment.score, 3),
                "sample_count": segment.sample_count,
                "clip_id": created["id"],
            }
        )

    return {
        "segments": created_segments,
        "gaps": [{"start_time": gap[0], "end_time": gap[1]} for gap in gaps],
        "video_resolution": [width, height],
        "total_frames_sampled": sum(segment.sample_count for segment in segments),
        "processing_time_sec": round(elapsed, 1),
    }


def auto_calibrate_clip(video_id: str, clip_id: int) -> dict[str, Any]:
    """Auto-calibrate one ``video_clips`` row from real frames and apply it."""
    clip = _get_video_clip(clip_id)
    if clip is None:
        raise ValueError(f"Clip {clip_id} not found")
    if clip["video_id"] != video_id:
        raise ValueError(f"Clip {clip_id} does not belong to video {video_id}")

    video_path = _get_video_path(video_id)
    if not video_path:
        raise ValueError(f"Video {video_id} is not downloaded")

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    inspection = inspect_clip_calibration(
        video_path,
        start_time=float(clip["start_time"]),
        end_time=float(clip["end_time"]),
        ref_resolution=(vid_w, vid_h),
    )
    proposal = inspection.proposal

    preview_b64 = None
    if inspection.preview_frame is not None:
        preview_b64 = _frame_to_base64(
            _annotate_calibration_preview(
                inspection.preview_frame,
                overlay_bbox=inspection.overlay_bbox,
                camera_bbox=proposal.camera if proposal is not None else None,
            )
        )

    if proposal is None:
        return {
            "clip_id": clip_id,
            "proposal": None,
            "applied": False,
            "preview_frame_b64": preview_b64,
            "failure_reason": inspection.failure_reason,
            "detected_overlay_bbox": list(inspection.overlay_bbox)
            if inspection.overlay_bbox is not None
            else None,
        }

    _update_video_clip(
        clip_id,
        {
            "overlay_bbox": list(proposal.overlay),
            "camera_bbox": list(proposal.camera),
            "ref_resolution": list(proposal.ref_resolution),
            "board_flipped": proposal.board_flipped,
            "board_theme": proposal.board_theme,
        },
    )

    return {
        "clip_id": clip_id,
        "proposal": {
            "overlay_bbox": list(proposal.overlay),
            "camera_bbox": list(proposal.camera),
            "board_theme": proposal.board_theme,
            "theme_confidence": round(proposal.theme_confidence, 3),
            "board_flipped": proposal.board_flipped,
            "orientation_confidence": round(proposal.orientation_confidence, 3),
            "ref_resolution": list(proposal.ref_resolution),
        },
        "applied": True,
        "preview_frame_b64": preview_b64,
        "failure_reason": None,
        "detected_overlay_bbox": list(inspection.overlay_bbox)
        if inspection.overlay_bbox is not None
        else None,
    }


def _frame_to_base64(frame: np.ndarray, max_width: int = 640) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _annotate_calibration_preview(
    frame: np.ndarray,
    overlay_bbox: tuple[int, int, int, int] | None = None,
    camera_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    if overlay_bbox is not None:
        ox, oy, ow, oh = overlay_bbox
        cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            "Overlay",
            (ox + 6, oy + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    if camera_bbox is not None:
        cx, cy, cw, ch = camera_bbox
        cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 3)
        cv2.putText(
            annotated,
            "OTB board",
            (cx + 6, cy + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
    return annotated


def _list_video_clips(video_id: str) -> list[dict[str, Any]]:
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
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _create_video_clip(video_id: str, data: dict[str, Any]) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(clip_index), -1) + 1 FROM video_clips WHERE video_id = %s",
                (video_id,),
            )
            next_index = int(cur.fetchone()[0])

            start = float(data.get("start_time", 0.0))
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
                    json.dumps(data["overlay_bbox"]),
                    json.dumps(data["camera_bbox"]),
                    json.dumps(data.get("ref_resolution", [1920, 1080])),
                    data.get("board_flipped", False),
                    data.get("board_theme", "lichess_default"),
                    data.get("is_gap", False),
                ),
            )
            cols = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            conn.commit()
            return dict(zip(cols, row))


def _update_video_clip(clip_id: int, data: dict[str, Any]) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id, start_time, end_time FROM video_clips WHERE id = %s",
                (clip_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise ValueError(f"Clip {clip_id} not found")

            video_id = str(existing[0])
            start = data.get("start_time", existing[1])
            end = data.get("end_time", existing[2])
            if "start_time" in data or "end_time" in data:
                _validate_no_overlap(cur, video_id, start, end, exclude_id=clip_id)

            sets: list[str] = []
            params: list[Any] = []
            scalar_fields = [
                "label",
                "start_time",
                "end_time",
                "board_flipped",
                "board_theme",
                "is_gap",
            ]
            for field in scalar_fields:
                if field in data:
                    sets.append(f"{field} = %s")
                    params.append(data[field])
            for json_field in ["overlay_bbox", "camera_bbox", "ref_resolution"]:
                if json_field in data:
                    sets.append(f"{json_field} = %s::jsonb")
                    params.append(json.dumps(data[json_field]))

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
            cols = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            conn.commit()
            return dict(zip(cols, row))


def _delete_video_clip(clip_id: int) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT video_id, clip_index FROM video_clips WHERE id = %s", (clip_id,))
            row = cur.fetchone()
            if not row:
                return False

            video_id, deleted_index = row
            cur.execute("DELETE FROM video_clips WHERE id = %s", (clip_id,))
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


def _get_video_clip(clip_id: int) -> dict[str, Any] | None:
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
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))


def _validate_no_overlap(
    cur: Any,
    video_id: str,
    start: float,
    end: float | None,
    exclude_id: int | None,
) -> None:
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

    for clip_index, existing_start, existing_end in cur.fetchall():
        other_end = existing_end if existing_end is not None else float("inf")
        candidate_end = end if end is not None else float("inf")
        if start < other_end and candidate_end > existing_start:
            raise ValueError(
                f"Clip time range overlaps with existing clip {clip_index}: "
                f"[{existing_start}, {existing_end}]"
            )
