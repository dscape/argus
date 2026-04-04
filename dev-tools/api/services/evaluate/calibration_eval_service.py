"""Service layer for evaluating auto-calibration accuracy.

Runs downstream validators (overlay detection, grid detection, piece
classification, theme detection, orientation detection, camera bbox)
on calibrated clips and compares results to stored calibration data.
Follows the session CRUD pattern from overlay_test_service.py.
"""

import base64
import json
import logging
import time
import uuid

import chess
import cv2
import numpy as np
from pipeline.db.connection import get_conn
from pipeline.overlay.auto_calibration import (
    compute_camera_bbox,
    detect_board_orientation,
    detect_board_theme,
)
from pipeline.overlay.grid_detector import detect_grid
from pipeline.overlay.piece_classifier import CLASS_TO_PIECE, classify_squares
from pipeline.overlay.scanner import detect_overlay_in_frame

logger = logging.getLogger(__name__)


# -- Helpers -----------------------------------------------------------------


def _get_video_path(video_id: str) -> str | None:
    """Find the local path for a downloaded video, or None."""
    from pipeline.paths import find_video_file

    path = find_video_file(video_id)
    return str(path) if path is not None else None


def _bbox_iou(a, b) -> float:
    """IoU of two (x, y, w, h) bboxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _scale_bbox(
    bbox: tuple,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    """Scale bbox from one resolution to another."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    x, y, w, h = bbox
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def _frame_to_base64(frame: np.ndarray, max_width: int = 400) -> str:
    """Encode a frame as a base64 JPEG, resizing if wider than max_width."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_bboxes(
    frame: np.ndarray,
    bboxes_with_colors: list[tuple],
) -> np.ndarray:
    """Draw labeled rectangles on a frame copy.

    bboxes_with_colors is a list of (bbox, color_bgr, label) tuples
    where bbox is (x, y, w, h).
    """
    vis = frame.copy()
    for bbox, color, label in bboxes_with_colors:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4), font, font_scale, (255, 255, 255), thickness)
    return vis


def _is_valid_fen(class_grid: list[list[int]]) -> tuple[bool, str]:
    """Check validity and return (is_valid, fen_string).

    Valid = has exactly 1 white king + 1 black king, total pieces between 2 and 32.
    Builds the FEN assuming white-at-bottom (not flipped) orientation.
    """
    board = chess.Board(fen=None)
    for r in range(8):
        for c in range(8):
            piece = CLASS_TO_PIECE.get(class_grid[r][c])
            if piece is not None:
                sq = chess.square(c, 7 - r)
                board.set_piece_at(sq, piece)

    fen = board.board_fen()

    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    total_pieces = sum(1 for sq in chess.SQUARES if board.piece_at(sq) is not None)

    is_valid = white_kings == 1 and black_kings == 1 and 2 <= total_pieces <= 32
    return is_valid, fen


# -- Sampling ----------------------------------------------------------------


def sample_calibration_clips(
    limit: int = 10,
    exclude: list[int] | None = None,
) -> list[dict]:
    """Sample clips that have calibration data and are downloaded locally.

    Returns list of dicts with clip_id, video_id, clip_index, start_time,
    end_time, board_theme, board_flipped.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT vc.id, vc.video_id, vc.clip_index, vc.start_time, vc.end_time,
                       vc.board_theme, vc.board_flipped
                FROM video_clips vc
                JOIN youtube_videos yv ON yv.video_id = vc.video_id
                WHERE vc.overlay_bbox IS NOT NULL
                  AND vc.camera_bbox IS NOT NULL
                ORDER BY vc.created_at DESC
                LIMIT 200
                """
            )
            rows = cur.fetchall()

    # Filter to clips whose video is downloaded locally
    candidates = []
    exclude_set = set(exclude or [])
    for row in rows:
        clip_id, video_id, clip_index, start_time, end_time, board_theme, board_flipped = row
        if clip_id in exclude_set:
            continue
        if _get_video_path(video_id) is None:
            continue
        candidates.append({
            "clip_id": clip_id,
            "video_id": video_id,
            "clip_index": clip_index,
            "start_time": start_time,
            "end_time": end_time,
            "board_theme": board_theme,
            "board_flipped": board_flipped,
        })
        if len(candidates) >= limit:
            break

    return candidates


# -- Inspection --------------------------------------------------------------


def inspect_calibration(clip_id: int) -> dict:
    """Run downstream validators on a calibrated clip and report metrics.

    Extracts 5 evenly-spaced frames, runs overlay detection, grid detection,
    piece classification, theme detection, and orientation detection on each.
    Compares all results to the stored calibration.
    """
    start = time.monotonic()

    # 1. Load clip from DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT vc.id, vc.video_id, vc.clip_index, vc.start_time, vc.end_time,
                       vc.overlay_bbox, vc.camera_bbox, vc.board_theme, vc.board_flipped,
                       vc.ref_resolution
                FROM video_clips vc
                WHERE vc.id = %s
                """,
                (clip_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Clip {clip_id} not found")

    (
        _clip_id, video_id, clip_index, start_time, end_time,
        overlay_bbox_raw, camera_bbox_raw, stored_theme, stored_flipped,
        ref_resolution_raw,
    ) = row

    # Parse JSONB fields (may already be lists or may be JSON strings)
    def _parse_json(val):
        if isinstance(val, (list, dict)):
            return val
        return json.loads(val) if val else None

    stored_overlay_bbox = _parse_json(overlay_bbox_raw)
    stored_camera_bbox = _parse_json(camera_bbox_raw)
    ref_resolution = _parse_json(ref_resolution_raw)

    # 2. Find local video
    video_path = _get_video_path(video_id)
    if not video_path:
        raise ValueError(f"Video {video_id} is not downloaded locally")

    # 3. Open video and extract 5 frames evenly spaced within [start_time, end_time]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_end = end_time if end_time is not None else video_duration
    clip_end = min(clip_end, video_duration)
    clip_duration = clip_end - start_time

    num_frames = 5
    if clip_duration <= 0:
        cap.release()
        raise ValueError(f"Clip {clip_id} has zero or negative duration")

    # Evenly spaced timestamps
    if num_frames == 1:
        timestamps = [start_time + clip_duration / 2]
    else:
        step = clip_duration / (num_frames - 1)
        timestamps = [start_time + i * step for i in range(num_frames)]

    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, ts))
    cap.release()

    if not frames:
        raise ValueError(f"Could not extract any frames from clip {clip_id}")

    # 4. Scale stored bboxes from ref_resolution to actual frame resolution if needed
    ref_w = ref_resolution[0] if ref_resolution else frame_w
    ref_h = ref_resolution[1] if ref_resolution else frame_h

    if ref_w != frame_w or ref_h != frame_h:
        scaled_overlay_bbox = _scale_bbox(stored_overlay_bbox, ref_w, ref_h, frame_w, frame_h)
        scaled_camera_bbox = _scale_bbox(stored_camera_bbox, ref_w, ref_h, frame_w, frame_h)
    else:
        scaled_overlay_bbox = tuple(stored_overlay_bbox)
        scaled_camera_bbox = tuple(stored_camera_bbox)

    # 5. Validate each frame
    middle_idx = len(frames) // 2
    per_frame = []
    overlay_ious = []
    grid_found_count = 0
    valid_fen_count = 0
    theme_match_count = 0
    orientation_match_count = 0

    for i, (frame, ts) in enumerate(frames):
        frame_result: dict = {"timestamp": round(ts, 2)}

        # 5a. Overlay detection IoU
        detection = detect_overlay_in_frame(frame)
        if detection.found and detection.bbox:
            detected_bbox = detection.bbox
            iou = _bbox_iou(detected_bbox, scaled_overlay_bbox)
            frame_result["overlay_detected"] = True
            frame_result["overlay_iou"] = round(iou, 4)
            overlay_ious.append(iou)
        else:
            frame_result["overlay_detected"] = False
            frame_result["overlay_iou"] = 0.0
            overlay_ious.append(0.0)

        # 5b. Crop overlay using stored bbox, run grid detection
        ox, oy, ow, oh = scaled_overlay_bbox
        ox, oy, ow, oh = int(ox), int(oy), int(ow), int(oh)
        # Clamp to frame bounds
        ox = max(0, min(ox, frame_w - 1))
        oy = max(0, min(oy, frame_h - 1))
        ow = min(ow, frame_w - ox)
        oh = min(oh, frame_h - oy)
        crop = frame[oy : oy + oh, ox : ox + ow]

        grid = detect_grid(crop)
        frame_result["grid_found"] = grid is not None
        if grid is not None:
            grid_found_count += 1

        # 5c. Piece classification and FEN validity
        if grid is not None:
            squares = grid.crop_squares(crop)
            class_grid = classify_squares(squares)
            is_valid, fen_str = _is_valid_fen(class_grid)
            frame_result["fen"] = fen_str
            frame_result["fen_valid"] = is_valid
            if is_valid:
                valid_fen_count += 1
        else:
            frame_result["fen"] = None
            frame_result["fen_valid"] = False

        # 5d. Theme detection
        detected_theme, theme_conf = detect_board_theme(crop)
        frame_result["detected_theme"] = detected_theme
        frame_result["theme_confidence"] = round(theme_conf, 3)
        frame_result["theme_match"] = detected_theme == stored_theme
        if detected_theme == stored_theme:
            theme_match_count += 1

        # 5e. Orientation detection
        detected_flipped, orient_conf = detect_board_orientation(crop, stored_theme)
        frame_result["detected_flipped"] = detected_flipped
        frame_result["orientation_confidence"] = round(orient_conf, 3)
        frame_result["orientation_match"] = detected_flipped == stored_flipped
        if detected_flipped == stored_flipped:
            orientation_match_count += 1

        # 5f. Save images for the middle frame only
        if i == middle_idx:
            # Frame with bboxes drawn
            bboxes_to_draw = [
                (scaled_overlay_bbox, (0, 255, 0), "stored overlay"),
                (scaled_camera_bbox, (255, 0, 0), "stored camera"),
            ]
            if detection.found and detection.bbox:
                bboxes_to_draw.append((detection.bbox, (0, 255, 255), "detected overlay"))
            annotated = _draw_bboxes(frame, bboxes_to_draw)
            frame_result["frame_b64"] = _frame_to_base64(annotated, max_width=400)
            frame_result["crop_b64"] = _frame_to_base64(crop, max_width=300)

        per_frame.append(frame_result)

    # 6. Camera IoU: use first and last frames
    first_frame = frames[0][0]
    last_frame = frames[-1][0]
    fresh_camera_bbox = compute_camera_bbox([first_frame, last_frame], scaled_overlay_bbox)
    camera_iou = _bbox_iou(fresh_camera_bbox, scaled_camera_bbox)

    # 7. Compute per-clip aggregate metrics
    total = len(frames)
    overlay_iou_avg = sum(overlay_ious) / total if total > 0 else 0.0
    grid_success_rate = grid_found_count / total if total > 0 else 0.0
    fen_validity_rate = valid_fen_count / total if total > 0 else 0.0
    theme_accuracy = theme_match_count / total if total > 0 else 0.0
    orientation_accuracy = orientation_match_count / total if total > 0 else 0.0

    elapsed_ms = round((time.monotonic() - start) * 1000, 1)

    return {
        "clip_id": clip_id,
        "video_id": video_id,
        "clip_index": clip_index,
        "start_time": start_time,
        "end_time": end_time,
        "stored_calibration": {
            "overlay_bbox": stored_overlay_bbox,
            "camera_bbox": stored_camera_bbox,
            "ref_resolution": ref_resolution,
            "board_theme": stored_theme,
            "board_flipped": stored_flipped,
        },
        "validation": {
            "frames": per_frame,
            "fresh_camera_bbox": list(fresh_camera_bbox),
            "camera_iou": round(camera_iou, 4),
        },
        "metrics": {
            "overlay_iou": round(overlay_iou_avg, 4),
            "grid_success_rate": round(grid_success_rate, 4),
            "fen_validity_rate": round(fen_validity_rate, 4),
            "theme_accuracy": round(theme_accuracy, 4),
            "orientation_accuracy": round(orientation_accuracy, 4),
            "camera_iou": round(camera_iou, 4),
        },
        "elapsed_ms": elapsed_ms,
    }


# -- Evaluation persistence --------------------------------------------------


def save_calibration_eval(
    overlay_iou: float,
    grid_success_rate: float,
    fen_validity_rate: float,
    theme_accuracy: float,
    orientation_accuracy: float,
    camera_iou: float,
    sample_size: int,
    notes: str | None = None,
) -> dict:
    """Save calibration evaluation result to model_evaluations.

    accuracy is the mean of the 5 core metrics (overlay_iou,
    grid_success_rate, fen_validity_rate, theme_accuracy,
    orientation_accuracy), excluding camera_iou since it uses only 2 frames.
    """
    accuracy = (
        overlay_iou + grid_success_rate + fen_validity_rate
        + theme_accuracy + orientation_accuracy
    ) / 5.0

    per_class_data = {
        "overlay_iou": overlay_iou,
        "grid_success_rate": grid_success_rate,
        "fen_validity_rate": fen_validity_rate,
        "theme_accuracy": theme_accuracy,
        "orientation_accuracy": orientation_accuracy,
        "camera_iou": camera_iou,
    }

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_evaluations
                   (model_name, sample_size, accuracy, notes, per_class)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, evaluated_at""",
                (
                    "auto_calibration",
                    sample_size,
                    accuracy,
                    notes,
                    json.dumps(per_class_data),
                ),
            )
            row = cur.fetchone()
            conn.commit()
    return {"id": row[0], "evaluated_at": str(row[1])}


# -- Session CRUD ------------------------------------------------------------


def create_calibration_eval_session(
    results: list[dict],
    overlay_iou_avg: float,
    theme_accuracy: float,
    orientation_accuracy: float,
    grid_success_rate: float,
    fen_validity_rate: float,
    sample_size: int,
    pin_state: dict | None = None,
    evaluation_id: int | None = None,
) -> dict:
    """Create and persist a calibration eval session."""
    session_id = uuid.uuid4().hex[:12]

    # Strip heavy base64 image data before storing in DB
    lightweight = []
    for r in results:
        entry = dict(r)
        # Strip from top-level
        entry.pop("frame_b64", None)
        entry.pop("crop_b64", None)
        # Strip from nested validation.frames
        if "validation" in entry and "frames" in entry["validation"]:
            clean_frames = []
            for fr in entry["validation"]["frames"]:
                clean_fr = {k: v for k, v in fr.items() if k not in ("frame_b64", "crop_b64")}
                clean_frames.append(clean_fr)
            entry["validation"]["frames"] = clean_frames
        lightweight.append(entry)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO calibration_eval_sessions
                   (id, sample_size, overlay_iou_avg, theme_accuracy,
                    orientation_accuracy, grid_success_rate, fen_validity_rate,
                    results, pin_state, evaluation_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id,
                    sample_size,
                    overlay_iou_avg,
                    theme_accuracy,
                    orientation_accuracy,
                    grid_success_rate,
                    fen_validity_rate,
                    json.dumps(lightweight),
                    json.dumps(pin_state or {}),
                    evaluation_id,
                ),
            )
            conn.commit()
    return {"session_id": session_id}


def get_calibration_eval_session(session_id: str) -> dict | None:
    """Fetch a calibration eval session by ID."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, overlay_iou_avg,
                          theme_accuracy, orientation_accuracy,
                          grid_success_rate, fen_validity_rate,
                          results, pin_state, evaluation_id
                   FROM calibration_eval_sessions WHERE id = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "created_at": str(row[1]),
                "sample_size": row[2],
                "overlay_iou_avg": row[3],
                "theme_accuracy": row[4],
                "orientation_accuracy": row[5],
                "grid_success_rate": row[6],
                "fen_validity_rate": row[7],
                "results": row[8] if isinstance(row[8], list) else json.loads(row[8]),
                "pin_state": row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
                "evaluation_id": row[10],
            }


def list_calibration_eval_sessions(limit: int = 20) -> list[dict]:
    """List recent calibration eval sessions (lightweight, no results)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, overlay_iou_avg,
                          theme_accuracy, orientation_accuracy,
                          grid_success_rate, fen_validity_rate
                   FROM calibration_eval_sessions
                   ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            return [
                {
                    "id": row[0],
                    "created_at": str(row[1]),
                    "sample_size": row[2],
                    "overlay_iou_avg": row[3],
                    "theme_accuracy": row[4],
                    "orientation_accuracy": row[5],
                    "grid_success_rate": row[6],
                    "fen_validity_rate": row[7],
                }
                for row in cur.fetchall()
            ]


def update_calibration_eval_pins(session_id: str, pin_state: dict) -> dict:
    """Merge pin state updates into a calibration eval session."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pin_state FROM calibration_eval_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Session {session_id} not found"}

            existing = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
            existing.update(pin_state)

            cur.execute(
                "UPDATE calibration_eval_sessions SET pin_state = %s WHERE id = %s",
                (json.dumps(existing), session_id),
            )
            conn.commit()
    return {"ok": True, "pin_state": existing}
