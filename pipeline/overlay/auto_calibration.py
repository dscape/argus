"""Auto-propose layout calibration for new channels.

Uses the overlay bounding box detected during screening to automatically
determine board theme, board orientation, and camera region — so the human
only needs to confirm or adjust rather than draw from scratch.
"""

import logging
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from pipeline.overlay.calibration import BOARD_THEMES, hex_to_bgr
from pipeline.overlay.scanner import detect_overlay_in_frame
from pipeline.screen.frame_fetcher import fetch_youtube_frames

logger = logging.getLogger(__name__)

# Default reference resolution for calibrations.
DEFAULT_REF_RESOLUTION = (1920, 1080)


@dataclass
class CalibrationProposal:
    """Auto-detected calibration proposal for human review."""

    overlay: tuple[int, int, int, int]  # x, y, w, h at ref resolution
    camera: tuple[int, int, int, int]  # x, y, w, h at ref resolution
    ref_resolution: tuple[int, int]
    board_flipped: bool
    board_theme: str
    theme_confidence: float
    orientation_confidence: float


def detect_board_theme(overlay_crop: np.ndarray) -> tuple[str, float]:
    """Detect which board theme matches the overlay by sampling square colors.

    Divides the crop into an 8x8 grid, identifies empty squares (low pixel
    variance), separates them into light/dark by checkerboard position,
    computes median colors, and matches against known themes.

    Returns (theme_name, confidence).
    """
    h, w = overlay_crop.shape[:2]
    if h < 32 or w < 32:
        return "lichess_default", 0.0

    # Resize to clean multiple of 8
    canonical = 512
    crop = cv2.resize(overlay_crop, (canonical, canonical))
    cell = canonical // 8

    # Collect colors from empty cells (low variance = no piece)
    even_colors = []  # (row+col) % 2 == 0
    odd_colors = []   # (row+col) % 2 == 1

    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell, (row + 1) * cell
            x1, x2 = col * cell, (col + 1) * cell
            sq = crop[y1:y2, x1:x2]

            # Sample center 30% to avoid borders/pieces at edges
            margin = cell // 3
            inner = sq[margin:-margin, margin:-margin]
            if inner.size == 0:
                continue

            gray_inner = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            variance = float(np.var(gray_inner))

            # Only use low-variance cells (likely empty, no piece)
            if variance < 200:
                mean_color = np.mean(inner.reshape(-1, 3), axis=0)  # BGR
                if (row + col) % 2 == 0:
                    even_colors.append(mean_color)
                else:
                    odd_colors.append(mean_color)

    if len(even_colors) < 4 or len(odd_colors) < 4:
        return "lichess_default", 0.0

    even_median = np.median(even_colors, axis=0)  # BGR
    odd_median = np.median(odd_colors, axis=0)     # BGR

    # Determine which group is light vs dark
    even_bright = float(np.mean(even_median))
    odd_bright = float(np.mean(odd_median))

    if even_bright >= odd_bright:
        light_bgr, dark_bgr = even_median, odd_median
    else:
        light_bgr, dark_bgr = odd_median, even_median

    # Match against known themes
    best_theme = "lichess_default"
    best_dist = float("inf")

    for theme_name, colors in BOARD_THEMES.items():
        ref_light = np.array(hex_to_bgr(colors["light"]), dtype=np.float64)
        ref_dark = np.array(hex_to_bgr(colors["dark"]), dtype=np.float64)

        dist_light = float(np.linalg.norm(light_bgr - ref_light))
        dist_dark = float(np.linalg.norm(dark_bgr - ref_dark))
        total_dist = dist_light + dist_dark

        if total_dist < best_dist:
            best_dist = total_dist
            best_theme = theme_name

    # Confidence: inverse of distance, normalized
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + best_dist / 50.0)))

    return best_theme, confidence


def detect_board_orientation(
    overlay_crop: np.ndarray,
    theme: str,
) -> tuple[bool, float]:
    """Detect whether the board is flipped (Black at bottom).

    Uses the DINOv2 piece classifier to read the board in both orientations
    and checks which has the expected piece distribution.

    Returns (flipped, confidence).
    """
    from pipeline.overlay.grid_detector import detect_grid
    from pipeline.overlay.piece_classifier import CLASS_TO_PIECE, classify_squares

    grid = detect_grid(overlay_crop)
    if grid is None:
        return False, 0.0

    squares = grid.crop_squares(overlay_crop)
    class_grid = classify_squares(squares)

    def _build_board(flipped: bool) -> chess.Board:
        board = chess.Board(fen=None)
        for r in range(8):
            for c in range(8):
                piece = CLASS_TO_PIECE.get(class_grid[r][c])
                if piece is None:
                    continue
                if not flipped:
                    sq = chess.square(c, 7 - r)
                else:
                    sq = chess.square(7 - c, r)
                board.set_piece_at(sq, piece)
        return board

    board_normal = _build_board(False)
    board_flipped = _build_board(True)

    def material_score(board: chess.Board, ranks: range, color: bool) -> float:
        piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
        score = 0.0
        for rank in ranks:
            for f in range(8):
                piece = board.piece_at(chess.square(f, rank))
                if piece is not None and piece.color == color:
                    score += piece_values.get(piece.piece_type, 0)
        return score

    normal_score = material_score(board_normal, range(0, 2), chess.WHITE) + \
                   material_score(board_normal, range(6, 8), chess.BLACK)
    flipped_score = material_score(board_flipped, range(0, 2), chess.WHITE) + \
                    material_score(board_flipped, range(6, 8), chess.BLACK)

    total = normal_score + flipped_score
    if total < 1.0:
        return False, 0.0

    is_flipped = flipped_score > normal_score
    confidence = abs(flipped_score - normal_score) / total

    return is_flipped, confidence


def compute_camera_bbox(
    frames: list[np.ndarray],
    overlay_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Find the camera footage region using frame differencing.

    Static graphics (banners, logos, clocks) don't change between frames.
    Camera footage shows people who move. By comparing two frames taken at
    different timestamps we find which blocks outside the overlay actually
    change — those are the camera region.

    Args:
        frames: Two or more video frames at different timestamps.
        overlay_bbox: Detected overlay bounding box.
    """
    if len(frames) < 2:
        # Fallback if only one frame: use the largest non-overlay rectangle
        frame_h, frame_w = frames[0].shape[:2]
        ox, oy, ow, oh = overlay_bbox
        sides = [
            (0, 0, ox, frame_h),
            (ox + ow, 0, frame_w - ox - ow, frame_h),
            (0, 0, frame_w, oy),
            (0, oy + oh, frame_w, frame_h - oy - oh),
        ]
        best = max(sides, key=lambda s: s[2] * s[3] if s[2] > 0 and s[3] > 0 else 0)
        return best

    frame_a = frames[0]
    frame_b = frames[1]
    frame_h, frame_w = frame_a.shape[:2]
    ox, oy, ow, oh = overlay_bbox

    # Compute absolute difference between the two frames.
    # Resize frame_b to match frame_a if resolutions differ.
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray_b.shape != gray_a.shape:
        gray_b = cv2.resize(gray_b, (frame_w, frame_h)).astype(np.float32)
    diff = np.abs(gray_a - gray_b)

    # Zero out the overlay region (the board changes too, we only want camera)
    diff[oy : oy + oh, ox : ox + ow] = 0

    # Determine which side of the overlay has the most space
    sides = [
        ("left", 0, 0, ox, frame_h),
        ("right", ox + ow, 0, frame_w - ox - ow, frame_h),
        ("top", 0, 0, frame_w, oy),
        ("bottom", 0, oy + oh, frame_w, frame_h - oy - oh),
    ]
    sides.sort(key=lambda s: s[3] * s[4] if s[3] > 0 and s[4] > 0 else 0, reverse=True)
    _, rx, ry, rw, rh = sides[0]

    if rw <= 0 or rh <= 0:
        return (0, 0, frame_w, frame_h)

    # Divide the candidate region into blocks and check which have motion
    block_size = max(60, min(rw, rh) // 8)
    cols = max(1, rw // block_size)
    rows = max(1, rh // block_size)
    bw = rw // cols
    bh = rh // rows

    cam_min_x, cam_min_y = frame_w, frame_h
    cam_max_x, cam_max_y = 0, 0
    found_camera = False

    # Threshold: blocks with significant pixel change are camera footage
    for row in range(rows):
        for col in range(cols):
            bx = rx + col * bw
            by = ry + row * bh
            block_diff = diff[by : by + bh, bx : bx + bw]
            mean_diff = float(np.mean(block_diff))

            # Camera footage: people move, so diff > ~5 between frames
            # Static banners/logos: diff ~ 0
            if mean_diff > 3.0:
                cam_min_x = min(cam_min_x, bx)
                cam_min_y = min(cam_min_y, by)
                cam_max_x = max(cam_max_x, bx + bw)
                cam_max_y = max(cam_max_y, by + bh)
                found_camera = True

    if not found_camera:
        return (rx, ry, rw, rh)

    return (cam_min_x, cam_min_y, cam_max_x - cam_min_x, cam_max_y - cam_min_y)


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    src_w: int, src_h: int,
    dst_w: int, dst_h: int,
) -> tuple[int, int, int, int]:
    """Scale a bounding box from one resolution to another."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    x, y, w, h = bbox
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def _get_video_path(video_id: str) -> str | None:
    """Find the local path for a downloaded video, or None."""
    from pipeline.paths import find_video_file

    path = find_video_file(video_id)
    return str(path) if path is not None else None


def _extract_frames_from_video(
    video_path: str,
    timestamps_sec: list[int] = (60, 120, 300),
) -> list[tuple[np.ndarray, str]]:
    """Extract frames from a local video file at given timestamps."""
    import cv2 as _cv2

    cap = _cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    duration = cap.get(_cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(_cv2.CAP_PROP_FPS), 1)
    frames = []
    for ts in timestamps_sec:
        if ts > duration:
            continue
        cap.set(_cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, f"{ts}s"))

    cap.release()
    return frames


def propose_calibration(
    video_id: str,
    ref_resolution: tuple[int, int] = DEFAULT_REF_RESOLUTION,
) -> CalibrationProposal | None:
    """Auto-propose calibration for a single video.

    Prefers frames from the downloaded video file (full resolution,
    accurate overlay detection). Falls back to YouTube thumbnails if the
    video is not downloaded locally.
    """
    # Try local video first (much better quality than thumbnails)
    video_path = _get_video_path(video_id)
    if video_path:
        logger.info(f"Using local video: {video_path}")
        frames = _extract_frames_from_video(video_path)
    else:
        logger.info(f"No local video for {video_id}, falling back to YouTube thumbnails")
        frames = fetch_youtube_frames(video_id)

    if not frames:
        logger.warning(f"Could not get any frames for {video_id}")
        return None

    # Find the largest overlay detection across all frames.
    # Prefer the biggest expanded bbox — small seeds can expand differently
    # depending on board position, so the largest result is usually correct.
    best_detection = None
    best_frame = None

    for frame, label in frames:
        detection = detect_overlay_in_frame(frame)
        if detection.found:
            bbox_area = detection.bbox[2] * detection.bbox[3] if detection.bbox else 0
            best_area = best_detection.bbox[2] * best_detection.bbox[3] if best_detection and best_detection.bbox else 0
            if bbox_area > best_area:
                best_detection = detection
                best_frame = frame

    if best_detection is None or best_frame is None:
        logger.info(f"No overlay detected for {video_id}")
        return None

    frame_h, frame_w = best_frame.shape[:2]
    overlay_bbox = best_detection.bbox
    logger.info(
        f"Overlay detected: bbox={overlay_bbox} "
        f"({overlay_bbox[2]/frame_w*100:.0f}%w x {overlay_bbox[3]/frame_h*100:.0f}%h)"
    )

    # Crop the overlay region for theme + orientation detection
    ox, oy, ow, oh = overlay_bbox
    overlay_crop = best_frame[oy : oy + oh, ox : ox + ow]

    # Detect theme
    theme, theme_conf = detect_board_theme(overlay_crop)
    logger.info(f"Detected theme: {theme} (confidence={theme_conf:.2f})")

    # Detect orientation
    flipped, orient_conf = detect_board_orientation(overlay_crop, theme)
    logger.info(f"Detected orientation: flipped={flipped} (confidence={orient_conf:.2f})")

    # Compute camera region using frame differencing (distinguishes
    # moving camera footage from static banners/graphics).
    all_frames = [f for f, _ in frames]
    camera_bbox = compute_camera_bbox(all_frames, overlay_bbox)

    # If frames came from the video at native resolution, no scaling needed
    # if the video matches the reference resolution. Otherwise scale.
    ref_w, ref_h = ref_resolution
    if frame_w == ref_w and frame_h == ref_h:
        overlay_final = overlay_bbox
        camera_final = camera_bbox
    else:
        overlay_final = _scale_bbox(overlay_bbox, frame_w, frame_h, ref_w, ref_h)
        camera_final = _scale_bbox(camera_bbox, frame_w, frame_h, ref_w, ref_h)

    return CalibrationProposal(
        overlay=overlay_final,
        camera=camera_final,
        ref_resolution=ref_resolution,
        board_flipped=flipped,
        board_theme=theme,
        theme_confidence=theme_conf,
        orientation_confidence=orient_conf,
    )


def propose_calibration_for_channel(
    channel_handle: str,
    ref_resolution: tuple[int, int] = DEFAULT_REF_RESOLUTION,
) -> CalibrationProposal | None:
    """Propose calibration by analyzing multiple videos from a channel.

    Queries the DB for approved overlay videos, runs proposal on up to 5,
    and takes consensus on theme, orientation, and averaged bbox positions.
    """
    from pipeline.db.connection import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id
                FROM youtube_videos
                WHERE channel_handle = %s
                  AND screening_status = 'approved'
                  AND (layout_type IS NULL OR layout_type = 'overlay')
                ORDER BY published_at DESC
                LIMIT 5
                """,
                (channel_handle,),
            )
            rows = cur.fetchall()

    if not rows:
        logger.warning(f"No approved overlay videos for {channel_handle}")
        return None

    video_ids = [r[0] for r in rows]
    proposals: list[CalibrationProposal] = []

    for vid in video_ids:
        p = propose_calibration(vid, ref_resolution=ref_resolution)
        if p is not None:
            proposals.append(p)

    if not proposals:
        return None

    if len(proposals) == 1:
        return proposals[0]

    # Take consensus: most common theme and orientation
    from collections import Counter

    themes = Counter(p.board_theme for p in proposals)
    best_theme = themes.most_common(1)[0][0]

    flipped_votes = sum(1 for p in proposals if p.board_flipped)
    is_flipped = flipped_votes > len(proposals) / 2

    # Average overlay bboxes
    avg_overlay = tuple(
        int(np.mean([p.overlay[i] for p in proposals]))
        for i in range(4)
    )
    avg_camera = tuple(
        int(np.mean([p.camera[i] for p in proposals]))
        for i in range(4)
    )

    # Aggregate confidence
    avg_theme_conf = float(np.mean([p.theme_confidence for p in proposals]))
    avg_orient_conf = float(np.mean([p.orientation_confidence for p in proposals]))

    return CalibrationProposal(
        overlay=avg_overlay,
        camera=avg_camera,
        ref_resolution=ref_resolution,
        board_flipped=is_flipped,
        board_theme=best_theme,
        theme_confidence=avg_theme_conf,
        orientation_confidence=avg_orient_conf,
    )


def propose_calibration_for_clip(
    video_path: str,
    start_time: float,
    end_time: float | None,
    num_samples: int = 5,
    ref_resolution: tuple[int, int] = DEFAULT_REF_RESOLUTION,
) -> CalibrationProposal | None:
    """Auto-propose calibration for a specific clip time range.

    Unlike :func:`propose_calibration` (which samples fixed timestamps or
    thumbnails), this operates on a specific ``[start_time, end_time]``
    window of a downloaded video.

    Args:
        video_path: Path to the local video file.
        start_time: Clip start in seconds.
        end_time: Clip end in seconds (``None`` = end of video).
        num_samples: Number of frames to sample within the clip.
        ref_resolution: Target resolution for the returned bboxes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    clip_end = end_time if end_time is not None else duration
    clip_end = min(clip_end, duration)
    clip_duration = clip_end - start_time

    if clip_duration <= 0:
        cap.release()
        return None

    # Evenly spaced timestamps within the clip, avoiding the very edges
    margin = min(5.0, clip_duration * 0.05)
    effective_start = start_time + margin
    effective_end = clip_end - margin
    if effective_start >= effective_end:
        effective_start = start_time
        effective_end = clip_end

    if num_samples == 1:
        timestamps = [(effective_start + effective_end) / 2]
    else:
        step = (effective_end - effective_start) / (num_samples - 1)
        timestamps = [effective_start + i * step for i in range(num_samples)]

    frames: list[tuple[np.ndarray, str]] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, f"{ts:.1f}s"))

    cap.release()

    if not frames:
        logger.warning(f"No frames extracted from {video_path} [{start_time}-{clip_end}]")
        return None

    # Find the best overlay detection across sampled frames
    best_detection = None
    best_frame = None

    for frame, label in frames:
        detection = detect_overlay_in_frame(frame)
        if detection.found:
            bbox_area = detection.bbox[2] * detection.bbox[3] if detection.bbox else 0
            best_area = (
                best_detection.bbox[2] * best_detection.bbox[3]
                if best_detection and best_detection.bbox
                else 0
            )
            if bbox_area > best_area:
                best_detection = detection
                best_frame = frame

    if best_detection is None or best_frame is None:
        logger.info(f"No overlay detected in clip [{start_time:.0f}-{clip_end:.0f}]")
        return None

    frame_h, frame_w = best_frame.shape[:2]
    overlay_bbox = best_detection.bbox
    ox, oy, ow, oh = overlay_bbox
    overlay_crop = best_frame[oy : oy + oh, ox : ox + ow]

    theme, theme_conf = detect_board_theme(overlay_crop)
    flipped, orient_conf = detect_board_orientation(overlay_crop, theme)

    all_frames = [f for f, _ in frames]
    camera_bbox = compute_camera_bbox(all_frames, overlay_bbox)

    ref_w, ref_h = ref_resolution
    if frame_w == ref_w and frame_h == ref_h:
        overlay_final = overlay_bbox
        camera_final = camera_bbox
    else:
        overlay_final = _scale_bbox(overlay_bbox, frame_w, frame_h, ref_w, ref_h)
        camera_final = _scale_bbox(camera_bbox, frame_w, frame_h, ref_w, ref_h)

    return CalibrationProposal(
        overlay=overlay_final,
        camera=camera_final,
        ref_resolution=ref_resolution,
        board_flipped=flipped,
        board_theme=theme,
        theme_confidence=theme_conf,
        orientation_confidence=orient_conf,
    )
