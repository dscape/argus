"""Service layer for segmenter accuracy evaluation.

Uses downstream validators (overlay detection, grid detection, piece
classification) to verify that predicted segments and gaps are correct.
Follows the same session CRUD pattern as overlay_test_service.
"""

import base64
import glob
import json
import logging
import os
import random
import time
import uuid

import chess
import cv2
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.overlay.grid_detector import detect_grid
from pipeline.overlay.piece_classifier import CLASS_TO_PIECE, classify_squares
from pipeline.overlay.scanner import detect_overlay_in_frame
from pipeline.overlay.segmenter import segment_video_layouts

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────


def _get_video_path(video_id: str) -> str | None:
    base_dirs = [
        "data/videos",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "videos"),
    ]
    for base in base_dirs:
        base = os.path.normpath(base)
        for ext in ("mp4", "mkv", "webm"):
            pattern = os.path.join(base, "*", f"{video_id}.{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
    return None


def _frame_to_base64(frame: np.ndarray, max_width: int = 300) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _is_valid_fen(class_grid: list[list[int]]) -> bool:
    """Check that the class grid contains exactly 1 white king and 1 black king."""
    white_kings = 0
    black_kings = 0
    for row in class_grid:
        for cls_idx in row:
            piece = CLASS_TO_PIECE.get(cls_idx)
            if piece is not None and piece.piece_type == chess.KING:
                if piece.color == chess.WHITE:
                    white_kings += 1
                else:
                    black_kings += 1
    return white_kings == 1 and black_kings == 1


# ── Sampling ──────────────────────────────────────────────


def sample_downloaded_videos(
    limit: int = 10,
    exclude: list[str] | None = None,
) -> list[str]:
    """Return random sample of approved video IDs that have local files."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id FROM youtube_videos WHERE screening_status = 'approved'"
            )
            all_ids = [row[0] for row in cur.fetchall()]

    exclude_set = set(exclude) if exclude else set()
    available = [
        vid for vid in all_ids
        if vid not in exclude_set and _get_video_path(vid) is not None
    ]

    if limit < len(available):
        available = random.sample(available, limit)

    return available


# ── Inspection ─────────────────────────────────────────────


def _load_stored_clips(video_id: str):
    """Load existing video_clips from DB if available.

    Returns (segments, gaps) in the same shape as segment_video_layouts,
    or (None, None) if no clips are stored.
    """
    from pipeline.overlay.segmenter import LayoutSegment

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT clip_index, start_time, end_time, overlay_bbox
                   FROM video_clips
                   WHERE video_id = %s
                   ORDER BY clip_index""",
                (video_id,),
            )
            rows = cur.fetchall()

    if not rows:
        return None, None

    segments = []
    for _, start_time, end_time, overlay_bbox_raw in rows:
        if overlay_bbox_raw is None:
            continue
        bbox = overlay_bbox_raw if isinstance(overlay_bbox_raw, (list, dict)) else __import__("json").loads(overlay_bbox_raw)
        # Convert dict bbox to tuple if needed
        if isinstance(bbox, dict):
            bbox = (bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0))
        elif isinstance(bbox, list):
            bbox = tuple(bbox)
        segments.append(LayoutSegment(
            start_time=start_time or 0.0,
            end_time=end_time or 0.0,
            overlay_bbox=bbox,
            score=1.0,
            sample_count=1,
        ))

    return segments if segments else None, None


def inspect_segmentation(video_id: str) -> dict:
    """Run segmentation evaluation on a single video.

    Uses stored clips from video_clips table if available (fast path).
    Falls back to running the full segmenter (slow, minutes per video).
    """
    start_wall = time.monotonic()

    video_path = _get_video_path(video_id)
    if video_path is None:
        return {"video_id": video_id, "error": "Video file not found"}

    # Fast path: use stored clips if available
    stored_segments, _ = _load_stored_clips(video_id)
    if stored_segments is not None:
        logger.info(f"Using {len(stored_segments)} stored clips for {video_id}")
        segments = stored_segments
        gaps = []  # We don't have stored gaps, will infer from segment boundaries
    else:
        # Slow path: run segmenter from scratch
        logger.info(f"No stored clips for {video_id}, running segmenter...")
        segments, gaps = segment_video_layouts(video_path)

    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Infer gaps from segment boundaries if using stored clips
    if stored_segments is not None and not gaps and duration > 0:
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        if sorted_segs[0].start_time > 10:
            gaps.append((0.0, sorted_segs[0].start_time))
        for i in range(len(sorted_segs) - 1):
            gap_start = sorted_segs[i].end_time
            gap_end = sorted_segs[i + 1].start_time
            if gap_end - gap_start > 10:
                gaps.append((gap_start, gap_end))
        if sorted_segs[-1].end_time < duration - 10:
            gaps.append((sorted_segs[-1].end_time, duration))

    # ── Validate segments ──────────────────────────────────
    segment_results = []
    total_segment_frames = 0
    overlay_detected_count = 0
    pieces_readable_count = 0

    for seg in segments:
        seg_duration = seg.end_time - seg.start_time
        if seg_duration <= 0:
            continue

        # Sample 3 evenly-spaced times within [start_time, end_time]
        sample_times = [
            seg.start_time + seg_duration * i / 2 for i in range(3)
        ]
        middle_idx = 1  # index of the middle frame

        frame_validations = []
        middle_frame_b64 = None

        for idx, t in enumerate(sample_times):
            frame_no = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_validations.append({
                    "time": round(t, 2),
                    "overlay_detected": False,
                    "grid_found": False,
                    "pieces_readable": False,
                })
                total_segment_frames += 1
                continue

            # Save base64 thumbnail of middle frame only
            if idx == middle_idx:
                middle_frame_b64 = _frame_to_base64(frame)

            # Detect overlay
            det = detect_overlay_in_frame(frame)
            overlay_found = det.found
            grid_found = False
            pieces_readable = False

            if overlay_found and det.bbox is not None:
                x, y, w, h = det.bbox
                crop = frame[y : y + h, x : x + w]

                grid = detect_grid(crop)
                if grid is not None:
                    grid_found = True
                    try:
                        squares = grid.crop_squares(crop)
                        class_grid = classify_squares(squares)
                        if _is_valid_fen(class_grid):
                            pieces_readable = True
                    except Exception:
                        logger.debug(
                            "Piece classification failed for %s at t=%.1f",
                            video_id, t,
                        )

            if overlay_found:
                overlay_detected_count += 1
            if pieces_readable:
                pieces_readable_count += 1
            total_segment_frames += 1

            frame_validations.append({
                "time": round(t, 2),
                "overlay_detected": overlay_found,
                "grid_found": grid_found,
                "pieces_readable": pieces_readable,
            })

        segment_results.append({
            "start_time": round(seg.start_time, 2),
            "end_time": round(seg.end_time, 2),
            "overlay_bbox": seg.overlay_bbox,
            "score": round(seg.score, 4),
            "sample_count": seg.sample_count,
            "frames": frame_validations,
            "thumbnail_b64": middle_frame_b64,
        })

    # ── Validate gaps ──────────────────────────────────────
    gap_results = []
    total_gap_frames = 0
    gap_no_overlay_count = 0
    false_negative_count = 0

    for gap_start, gap_end in gaps:
        gap_duration = gap_end - gap_start
        if gap_duration <= 0:
            continue

        # Sample 2 frames
        sample_times = [
            gap_start + gap_duration * 0.25,
            gap_start + gap_duration * 0.75,
        ]

        frame_validations = []
        any_overlay = False

        for t in sample_times:
            frame_no = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_validations.append({
                    "time": round(t, 2),
                    "overlay_detected": False,
                })
                total_gap_frames += 1
                gap_no_overlay_count += 1
                continue

            det = detect_overlay_in_frame(frame)
            overlay_found = det.found

            if overlay_found:
                any_overlay = True
            else:
                gap_no_overlay_count += 1
            total_gap_frames += 1

            frame_validations.append({
                "time": round(t, 2),
                "overlay_detected": overlay_found,
            })

        if any_overlay:
            false_negative_count += 1

        gap_results.append({
            "start_time": round(gap_start, 2),
            "end_time": round(gap_end, 2),
            "frames": frame_validations,
            "has_overlay": any_overlay,
        })

    cap.release()

    # ── Compute metrics ────────────────────────────────────
    segment_consistency = (
        overlay_detected_count / total_segment_frames
        if total_segment_frames > 0
        else 0.0
    )
    gap_consistency = (
        gap_no_overlay_count / total_gap_frames
        if total_gap_frames > 0
        else 1.0
    )
    piece_readability = (
        pieces_readable_count / total_segment_frames
        if total_segment_frames > 0
        else 0.0
    )
    coverage_ratio = (
        sum(seg.end_time - seg.start_time for seg in segments) / duration
        if duration > 0
        else 0.0
    )

    elapsed_ms = round((time.monotonic() - start_wall) * 1000, 1)

    return {
        "video_id": video_id,
        "duration_sec": round(duration, 2),
        "resolution": f"{width}x{height}",
        "num_segments": len(segments),
        "num_gaps": len(gaps),
        "segments": segment_results,
        "gaps": gap_results,
        "metrics": {
            "segment_consistency": round(segment_consistency, 4),
            "gap_consistency": round(gap_consistency, 4),
            "piece_readability": round(piece_readability, 4),
            "false_negative_count": false_negative_count,
            "coverage_ratio": round(coverage_ratio, 4),
        },
        "elapsed_ms": elapsed_ms,
    }


# ── Evaluation persistence ──────────────────────────────────


def save_segmentation_eval(
    segment_consistency: float,
    gap_consistency: float,
    piece_readability: float,
    false_negative_rate: float,
    coverage_ratio: float,
    sample_size: int,
    notes: str | None = None,
) -> dict:
    """Save segmentation evaluation result to model_evaluations."""
    per_class_data = {
        "gap_consistency": gap_consistency,
        "piece_readability": piece_readability,
        "false_negative_rate": false_negative_rate,
        "coverage_ratio": coverage_ratio,
    }

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_evaluations
                   (model_name, sample_size, accuracy, notes, per_class)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, evaluated_at""",
                (
                    "segmenter",
                    sample_size,
                    segment_consistency,
                    notes,
                    json.dumps(per_class_data),
                ),
            )
            row = cur.fetchone()
            conn.commit()
    return {"id": row[0], "evaluated_at": str(row[1])}


# ── Session CRUD ────────────────────────────────────────────


def create_segmentation_eval_session(
    results: list[dict],
    segment_consistency: float,
    gap_consistency: float,
    piece_readability: float,
    false_negative_rate: float,
    coverage_ratio: float,
    sample_size: int,
    pin_state: dict | None = None,
    evaluation_id: int | None = None,
) -> dict:
    """Create and persist a segmentation eval session."""
    session_id = uuid.uuid4().hex[:12]

    # Strip heavy image data for storage
    lightweight = []
    for r in results:
        entry = dict(r)
        # Strip thumbnails from segments within each video result
        if "segments" in entry:
            entry["segments"] = [
                {k: v for k, v in seg.items() if k != "thumbnail_b64"}
                for seg in entry["segments"]
            ]
        lightweight.append(entry)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO segmentation_eval_sessions
                   (id, sample_size, segment_consistency, gap_consistency,
                    piece_readability, false_negative_rate, coverage_ratio,
                    results, pin_state, evaluation_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id,
                    sample_size,
                    segment_consistency,
                    gap_consistency,
                    piece_readability,
                    false_negative_rate,
                    coverage_ratio,
                    json.dumps(lightweight),
                    json.dumps(pin_state or {}),
                    evaluation_id,
                ),
            )
            conn.commit()
    return {"session_id": session_id}


def get_segmentation_eval_session(session_id: str) -> dict | None:
    """Fetch a segmentation eval session by ID."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, segment_consistency,
                          gap_consistency, piece_readability, false_negative_rate,
                          coverage_ratio, results, pin_state, evaluation_id
                   FROM segmentation_eval_sessions WHERE id = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "created_at": str(row[1]),
                "sample_size": row[2],
                "segment_consistency": row[3],
                "gap_consistency": row[4],
                "piece_readability": row[5],
                "false_negative_rate": row[6],
                "coverage_ratio": row[7],
                "results": row[8] if isinstance(row[8], list) else json.loads(row[8]),
                "pin_state": row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
                "evaluation_id": row[10],
            }


def list_segmentation_eval_sessions(limit: int = 20) -> list[dict]:
    """List recent segmentation eval sessions (lightweight, no results)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, segment_consistency,
                          gap_consistency, piece_readability, false_negative_rate,
                          coverage_ratio
                   FROM segmentation_eval_sessions
                   ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            return [
                {
                    "id": row[0],
                    "created_at": str(row[1]),
                    "sample_size": row[2],
                    "segment_consistency": row[3],
                    "gap_consistency": row[4],
                    "piece_readability": row[5],
                    "false_negative_rate": row[6],
                    "coverage_ratio": row[7],
                }
                for row in cur.fetchall()
            ]


def update_segmentation_eval_pins(session_id: str, pin_state: dict) -> dict:
    """Merge pin state updates into a segmentation eval session."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pin_state FROM segmentation_eval_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Session {session_id} not found"}

            existing = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
            existing.update(pin_state)

            cur.execute(
                "UPDATE segmentation_eval_sessions SET pin_state = %s WHERE id = %s",
                (json.dumps(existing), session_id),
            )
            conn.commit()
    return {"ok": True, "pin_state": existing}
