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
from pipeline.overlay.scanner import (
    MIN_LOW_VARIANCE_RATIO,
    compute_grid_regularity,
    detect_overlay_in_frame,
    fast_overlay_check,
)
from pipeline.overlay.segmenter import segment_video_layouts

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────


def _get_video_path(video_id: str) -> str | None:
    base_dirs = [
        "data/videos",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "videos"),
    ]
    for base in base_dirs:
        base = os.path.normpath(base)
        for ext in ("mp4", "mkv", "webm"):
            pattern = os.path.join(base, "*", f"{video_id}.{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
    return None


def _frame_to_base64(frame: np.ndarray, max_width: int = 200) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
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


def _validate_segment_frame(
    frame: np.ndarray,
    t: float,
    stored_bbox: tuple[int, int, int, int] | None,
    run_piece_classify: bool = True,
) -> dict:
    """Validate one segment frame: overlay detection + grid + piece classification.

    Strategy:
    - When a *stored_bbox* is available and detect_grid succeeds on that crop:
      skip the expensive full-frame sliding-window (detect_overlay_in_frame).
      The stored bbox already tells us where the board should be; we just
      verify it with the grid detector (fast, ~20ms) rather than re-scanning.
    - Otherwise, fall back to detect_overlay_in_frame (accurate but slower).
    - DINOv2 piece classification only runs when *run_piece_classify=True*
      (disabled for non-middle frames to avoid running it on all 5 per segment).
    """
    thumb = _frame_to_base64(frame)
    fh, fw = frame.shape[:2]
    overlay_found = False
    grid_found = False
    crop: np.ndarray | None = None
    grid = None  # detect_grid result; reused for piece classification

    # Also run fast_overlay_check to measure what the segmenter actually sees.
    fast_det = fast_overlay_check(frame)
    fast_check_found = fast_det.found

    if stored_bbox is not None:
        # Fast path: try detect_grid on stored bbox crop (no sliding window).
        x, y, w, h = stored_bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + w, fw), min(y + h, fh)
        if x2 > x1 and y2 > y1:
            candidate_crop = frame[y1:y2, x1:x2]
            grid = detect_grid(candidate_crop)
            if grid is not None:
                overlay_found = True
                grid_found = True
                crop = candidate_crop
        # If detect_grid failed on stored bbox, fall back to full frame scan
        if not overlay_found:
            det = detect_overlay_in_frame(frame)
            overlay_found = det.found
            if det.found and det.bbox is not None:
                bx, by, bw, bh = det.bbox
                bx1, by1 = max(0, bx), max(0, by)
                bx2, by2 = min(bx + bw, fw), min(by + bh, fh)
                if bx2 > bx1 and by2 > by1:
                    crop = frame[by1:by2, bx1:bx2]
                    grid = detect_grid(crop) if crop is not None else None
                    if grid is not None:
                        grid_found = True
    else:
        # No stored bbox: full sliding-window detection.
        det = detect_overlay_in_frame(frame)
        overlay_found = det.found
        if det.found and det.bbox is not None:
            bx, by, bw, bh = det.bbox
            bx1, by1 = max(0, bx), max(0, by)
            bx2, by2 = min(bx + bw, fw), min(by + bh, fh)
            if bx2 > bx1 and by2 > by1:
                crop = frame[by1:by2, bx1:bx2]
                grid = detect_grid(crop) if crop is not None else None
                if grid is not None:
                    grid_found = True

    pieces_readable = False
    if grid_found and run_piece_classify and crop is not None and crop.size > 0:
        # `grid` is already set from the detection step above — reuse it.
        if grid is not None:
            try:
                squares = grid.crop_squares(crop)
                class_grid = classify_squares(squares)
                if _is_valid_fen(class_grid):
                    pieces_readable = True
            except Exception:
                logger.debug("Piece classification failed at t=%.1f", t)

    return {
        "time": round(t, 2),
        "overlay_detected": overlay_found,
        "fast_check_found": fast_check_found,
        "grid_found": grid_found,
        "pieces_readable": pieces_readable,
        "thumbnail_b64": thumb,
    }


def _validate_gap_frame(frame: np.ndarray, t: float) -> dict:
    """Fast overlay presence check for one gap frame (no bbox needed).

    Uses fast_overlay_check: gap validation only needs overlay yes/no.
    """
    det = fast_overlay_check(frame)
    return {
        "time": round(t, 2),
        "overlay_detected": det.found,
        "thumbnail_b64": _frame_to_base64(frame),
    }


# ── Sampling ──────────────────────────────────────────────


def sample_downloaded_videos(
    limit: int = 10,
    exclude: list[str] | None = None,
) -> list[str]:
    """Return random sample of video IDs that have stored clips and local files.

    Only returns videos with entries in video_clips — avoids triggering the
    full segmenter (which takes minutes per video) during interactive evals.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT video_id FROM video_clips"
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

    Performance strategy:
    - All frame timestamps are sorted ascending so video seeking is
      monotonically forward — no expensive backward H264 seeks.
    - Segment frames use fast_overlay_check (not detect_overlay_in_frame)
      plus the stored bbox from video_clips for grid/piece cropping,
      avoiding the expensive 10-scale sliding window + expansion entirely.
    - Gap frames use fast_overlay_check (only need yes/no, not bbox).
    """
    t0 = time.monotonic()

    video_path = _get_video_path(video_id)
    if video_path is None:
        return {"video_id": video_id, "error": "Video file not found"}

    # Fast path: use stored clips if available
    stored_segments, _ = _load_stored_clips(video_id)
    if stored_segments is not None:
        logger.info(f"Using {len(stored_segments)} stored clips for {video_id}")
        segments = stored_segments
        gaps: list[tuple[float, float]] = []
    else:
        # Slow path: run segmenter from scratch
        logger.info(f"No stored clips for {video_id}, running segmenter...")
        segments, gaps = segment_video_layouts(video_path)

    # Get video metadata (quick single cap, released immediately)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    t_meta = time.monotonic()
    logger.info(
        f"[timing] {video_id}: metadata in {(t_meta - t0)*1000:.0f}ms, "
        f"{len(segments)} segs, duration={duration:.0f}s"
    )

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

    # ── Build per-frame task list ──────────────────────────
    # Each task: {t, mode, seg_idx/gap_idx, frame_idx, is_middle, stored_bbox}
    # Sort by timestamp so VideoCapture seeks are monotonically forward.

    all_tasks: list[dict] = []

    # DINOv2 budget: limit piece classification to at most 1 segment per video.
    # DINOv2 takes 3–4s per call on CPU; running it once per video is sufficient
    # for the eval metric while keeping the overall eval time manageable.
    piece_classify_budget = 1

    for i, seg in enumerate(segments):
        dur = seg.end_time - seg.start_time
        if dur <= 0:
            continue
        # 5 evenly-spaced frames at 0%, 25%, 50%, 75%, 100%
        for k in range(5):
            t = seg.start_time + dur * k / 4
            # Only allow DINOv2 for the middle frame of the first segment.
            is_mid = k == 2 and piece_classify_budget > 0
            if is_mid:
                piece_classify_budget -= 1
            all_tasks.append({
                "t": t,
                "mode": "segment",
                "seg_idx": i,
                "frame_idx": k,
                "is_middle": is_mid,
                "stored_bbox": seg.overlay_bbox,
            })

    for i, (gap_start, gap_end) in enumerate(gaps):
        dur = gap_end - gap_start
        if dur <= 0:
            continue
        # 4 evenly-spaced frames at 12.5%, 37.5%, 62.5%, 87.5%
        for k in range(4):
            t = gap_start + dur * (2 * k + 1) / 8
            all_tasks.append({
                "t": t,
                "mode": "gap",
                "gap_idx": i,
                "frame_idx": k,
                "is_middle": False,
                "stored_bbox": None,
            })

    # Sort ascending — forward-only seeks, much faster on H264
    all_tasks.sort(key=lambda x: x["t"])

    total_frames = len(all_tasks)
    logger.info(
        f"[timing] {video_id}: {sum(1 for t in all_tasks if t['mode']=='segment')} seg frames + "
        f"{sum(1 for t in all_tasks if t['mode']=='gap')} gap frames = {total_frames} total"
    )

    # ── Sequential frame processing ────────────────────────
    # One VideoCapture, monotonically forward seeks.

    t_proc_start = time.monotonic()

    seg_frame_results: dict[tuple[int, int], tuple[dict, bool]] = {}
    gap_frame_results: dict[tuple[int, int], dict] = {}

    cap = cv2.VideoCapture(video_path)
    for task in all_tasks:
        frame_no = int(task["t"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        t = task["t"]

        if task["mode"] == "segment":
            si, fi = task["seg_idx"], task["frame_idx"]
            is_mid = task["is_middle"]
            if not ret or frame is None:
                result: dict = {
                    "time": round(t, 2),
                    "overlay_detected": False,
                    "grid_found": False,
                    "pieces_readable": False,
                    "thumbnail_b64": None,
                }
            else:
                result = _validate_segment_frame(
                    frame, t, task["stored_bbox"],
                    run_piece_classify=task["is_middle"],
                )
            seg_frame_results[(si, fi)] = (result, is_mid)
        else:
            gi, fi = task["gap_idx"], task["frame_idx"]
            if not ret or frame is None:
                gap_result: dict = {
                    "time": round(t, 2),
                    "overlay_detected": False,
                    "thumbnail_b64": None,
                }
            else:
                gap_result = _validate_gap_frame(frame, t)
            gap_frame_results[(gi, fi)] = gap_result

    cap.release()

    t_proc_end = time.monotonic()
    logger.info(
        f"[timing] {video_id}: sequential frame processing in "
        f"{(t_proc_end - t_proc_start)*1000:.0f}ms "
        f"({total_frames} frames)"
    )

    # ── Aggregate segment results ──────────────────────────

    segment_results = []
    total_segment_frames = 0
    overlay_detected_count = 0
    fast_check_found_count = 0
    grid_found_count = 0
    pieces_readable_count = 0

    for i, seg in enumerate(segments):
        if seg.end_time - seg.start_time <= 0:
            continue

        frame_validations = []
        middle_frame_b64 = None
        seg_piece_eval_ran = False
        seg_piece_readable = False

        for k in range(5):
            key = (i, k)
            if key not in seg_frame_results:
                continue
            result, is_mid = seg_frame_results[key]
            frame_validations.append(result)
            if is_mid:
                middle_frame_b64 = result.get("thumbnail_b64")
                seg_piece_eval_ran = True
                seg_piece_readable = bool(result["pieces_readable"])
            total_segment_frames += 1
            if result["overlay_detected"]:
                overlay_detected_count += 1
            if result.get("fast_check_found"):
                fast_check_found_count += 1
            if result.get("grid_found"):
                grid_found_count += 1
            if result["pieces_readable"]:
                pieces_readable_count += 1

        # Sort by timestamp for display
        frame_validations.sort(key=lambda f: f["time"])

        segment_results.append({
            "start_time": round(seg.start_time, 2),
            "end_time": round(seg.end_time, 2),
            "overlay_bbox": seg.overlay_bbox,
            "score": round(seg.score, 4),
            "sample_count": seg.sample_count,
            "frames": frame_validations,
            "thumbnail_b64": middle_frame_b64,
            "piece_eval_ran": seg_piece_eval_ran,
            "piece_readable": seg_piece_readable,
        })

    # ── Aggregate gap results ──────────────────────────────

    gap_results = []
    total_gap_frames = 0
    gap_no_overlay_count = 0
    false_negative_count = 0

    for i, (gap_start, gap_end) in enumerate(gaps):
        if gap_end - gap_start <= 0:
            continue

        frame_validations = []
        any_overlay = False

        for k in range(4):
            key = (i, k)
            if key not in gap_frame_results:
                continue
            result = gap_frame_results[key]
            frame_validations.append(result)
            total_gap_frames += 1
            if result["overlay_detected"]:
                any_overlay = True
            else:
                gap_no_overlay_count += 1

        frame_validations.sort(key=lambda f: f["time"])

        if any_overlay:
            false_negative_count += 1

        gap_results.append({
            "start_time": round(gap_start, 2),
            "end_time": round(gap_end, 2),
            "frames": frame_validations,
            "has_overlay": any_overlay,
        })

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
    # Per-segment piece readability: fraction of segments where the DINOv2-evaluated
    # middle frame produced a valid FEN.  This is meaningful even with a budget cap
    # (1 DINOv2 call per video) because we always evaluate exactly the middle frame
    # of the first segment, so the ratio reflects real classification success rather
    # than being diluted by frames that never ran DINOv2.
    segments_with_piece_eval = sum(
        1 for seg in segment_results if seg.get("piece_eval_ran")
    )
    segments_piece_readable = sum(
        1 for seg in segment_results if seg.get("piece_readable")
    )
    piece_readability = (
        segments_piece_readable / segments_with_piece_eval
        if segments_with_piece_eval > 0
        else 0.0
    )
    coverage_ratio = (
        sum(seg.end_time - seg.start_time for seg in segments) / duration
        if duration > 0
        else 0.0
    )

    # fast_overlay_check consistency: what the segmenter would actually see
    fast_check_consistency = (
        fast_check_found_count / total_segment_frames
        if total_segment_frames > 0
        else 0.0
    )

    # Sub-step breakdown: where does the pipeline fail?
    overlay_miss_count = total_segment_frames - overlay_detected_count
    fast_check_miss_count = total_segment_frames - fast_check_found_count
    # grid_miss: overlay found but grid not found
    grid_miss_count = overlay_detected_count - grid_found_count
    # fen_miss: grid found but FEN invalid
    fen_miss_count = grid_found_count - pieces_readable_count

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        f"[timing] {video_id}: total={elapsed_ms}ms | "
        f"seg_consistency={segment_consistency:.2%} "
        f"fast_check={fast_check_consistency:.2%} "
        f"gap_consistency={gap_consistency:.2%} "
        f"piece_readability={piece_readability:.2%} | "
        f"overlay_miss={overlay_miss_count} fast_miss={fast_check_miss_count} "
        f"grid_miss={grid_miss_count} fen_miss={fen_miss_count}"
    )

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
            "fast_check_consistency": round(fast_check_consistency, 4),
            "gap_consistency": round(gap_consistency, 4),
            "piece_readability": round(piece_readability, 4),
            "false_negative_count": false_negative_count,
            "coverage_ratio": round(coverage_ratio, 4),
            "overlay_miss_count": overlay_miss_count,
            "fast_check_miss_count": fast_check_miss_count,
            "grid_miss_count": grid_miss_count,
            "fen_miss_count": fen_miss_count,
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
    """Save segmentation evaluation result to model_evaluations.

    Stores balanced_accuracy = (segment_consistency + gap_consistency) / 2
    in the accuracy column so it is consistent with the Screening and Overlay
    model performance charts.
    """
    balanced_accuracy = (segment_consistency + gap_consistency) / 2.0

    per_class_data = {
        "segment_consistency": segment_consistency,
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
                    balanced_accuracy,
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
