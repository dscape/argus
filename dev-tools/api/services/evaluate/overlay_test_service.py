"""Service layer for overlay (piece classifier) accuracy testing.

Uses the overlay/positions dataset (400x400 board images with FEN in filenames)
to evaluate the DINOv2 piece classifier. Also supports real overlay crops
extracted from screening frames. Mirrors the screening inspector pattern:
sample → inspect → save session.
"""

import base64
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path

import chess
import cv2
import numpy as np
from pipeline.db.connection import get_conn
from pipeline.overlay.chess_positions_data import (
    BOARD_SIZE,
    SQ_SIZE,
    parse_fen_from_filename,
)
from pipeline.overlay.grid_detector import GridResult, detect_grid
from pipeline.overlay.piece_classifier import classify_squares, read_fen_with_grid
from pipeline.overlay.scanner import detect_overlay_fast, fast_overlay_check
from pipeline.paths import find_frame

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
CHESS_POSITIONS_TEST_DIR = _PROJECT_ROOT / "data" / "overlay" / "val"
# Real overlay crops extracted from video_clips
REAL_OVERLAY_TEST_DIR = _PROJECT_ROOT / "data" / "overlay" / "val_real"
# Prefix prepended to real sample filenames in the sample list so inspect_board
# can route to the right directory without path separators (which the board-image
# endpoint rejects for security).
_REAL_PREFIX = "real__"

_KNOWN_FRAME_NAMES = ("thumb_hires", "thumb_sd", "thumb", "25pct", "50pct", "75pct")
# For overlay extraction, only use auto-generated gameplay frames (25/50/75% of video).
# Custom thumbnails (thumb_hires, thumb_sd, thumb) are often promotional/title images
# that trigger false positives in the grid-regularity detector.
_EXTRACTION_FRAME_NAMES = ("25pct", "50pct", "75pct")
_APPROVED_OVERLAY_WHERE = """
    screening_status = 'approved'
    AND (layout_type = 'overlay' OR layout_type IS NULL)
"""

# Uniform grid for 400×400 chess-positions boards (50px squares)
_GRID = GridResult(
    v_lines=list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
    h_lines=list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
    sq_size=SQ_SIZE,
)


def _resolve_frame_path(video_id: str, frame_name: str) -> Path | None:
    """Return overlay frame path, preferring fullres then hires.

    Screening frames (480x360) are too low-res for reliable overlay detection.
    """
    for tier in ("fullres", "hires"):
        path = find_frame(video_id, tier, frame_name)  # type: ignore[arg-type]
        if path is not None:
            return path
    return None


def _video_has_frames(video_id: str) -> bool:
    """Check if a video has any extraction frames available."""
    for name in _EXTRACTION_FRAME_NAMES:
        if _resolve_frame_path(video_id, name) is not None:
            return True
    return False


def _video_has_unsaved_frames(video_id: str, saved_keys: set[str]) -> bool:
    """Return True when at least one extraction frame exists and is not saved."""
    for frame_name in _EXTRACTION_FRAME_NAMES:
        if f"{video_id}:{frame_name}" in saved_keys:
            continue
        if _resolve_frame_path(video_id, frame_name) is not None:
            return True
    return False


def _frame_to_base64(frame: np.ndarray, max_width: int = 400) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _fen_to_board(fen: str) -> chess.Board:
    if " " not in fen:
        fen = fen + " w - - 0 1"
    return chess.Board(fen)


def _diff_boards(predicted: chess.Board, expected: chess.Board) -> list[str]:
    """Return list of mismatch descriptions like 'e4: got P, expected .'."""
    mismatches: list[str] = []
    for sq in chess.SQUARES:
        p = predicted.piece_at(sq)
        e = expected.piece_at(sq)
        if p != e:
            sq_name = chess.square_name(sq)
            p_str = p.symbol() if p else "."
            e_str = e.symbol() if e else "."
            mismatches.append(f"{sq_name}: got {p_str}, expected {e_str}")
    return mismatches


# ── Sampling ────────────────────────────────────────────────


def sample_board_filenames(
    limit: int = 20,
    exclude: list[str] | None = None,
) -> list[str]:
    """Return random sample of board image filenames from the test set.

    Mixes synthetic (chess-positions) and real samples.  Real samples come
    from pre-extracted overlay crops in ``test_real/`` (``real__`` prefix).
    Up to 20% of the sample will be real when real data is available;
    if not enough real samples exist the quota is filled with synthetic.
    """
    if not CHESS_POSITIONS_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Chess positions test directory not found: {CHESS_POSITIONS_TEST_DIR}"
        )

    exclude_set = set(exclude or [])
    _IMG_EXTS = (".jpeg", ".jpg", ".png")

    synthetic = [
        f
        for f in os.listdir(CHESS_POSITIONS_TEST_DIR)
        if f.endswith(_IMG_EXTS) and f not in exclude_set
    ]

    # Gather real samples from pre-extracted overlay crops
    real: list[str] = []
    if REAL_OVERLAY_TEST_DIR.exists():
        real = [
            _REAL_PREFIX + f
            for f in os.listdir(REAL_OVERLAY_TEST_DIR)
            if f.endswith(_IMG_EXTS) and (_REAL_PREFIX + f) not in exclude_set
        ]

    real_quota = min(len(real), max(1, limit // 5))
    synth_quota = limit - real_quota

    chosen_real = random.sample(real, real_quota) if real_quota else []
    chosen_synth = random.sample(synthetic, min(synth_quota, len(synthetic)))

    combined = chosen_real + chosen_synth
    random.shuffle(combined)
    return combined


# ── Inspection ──────────────────────────────────────────────


def inspect_board(filename: str) -> dict:
    """Inspect a single board image: classify pieces and compare to ground truth.

    Handles two sample types:
    - Synthetic (chess-positions): fixed 50px grid, FEN from filename.
    - Real crops (``real__`` prefix): detected grid, FEN from filename.
    Both always produce ``match: bool`` and ``piece_accuracy: float``.
    """
    is_real = filename.startswith(_REAL_PREFIX)

    if is_real:
        actual_name = filename[len(_REAL_PREFIX):]
        image_path = REAL_OVERLAY_TEST_DIR / actual_name
        source = "real"
        stem = Path(actual_name).stem
        if stem.startswith("f_"):
            # Frame-based: f_{video_id}_{frame_name}_{fen_hyphenated}
            _, _, expected_fen = _parse_frame_filename(stem)
        else:
            # Legacy clip-based: r{clip_id}_{fen_hyphenated}
            fen_hyphenated = stem.split("_", 1)[1] if "_" in stem else stem
            expected_fen = fen_hyphenated.replace("-", "/")
        if not image_path.exists():
            raise FileNotFoundError(f"Board image not found: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    else:
        image_path = CHESS_POSITIONS_TEST_DIR / filename
        source = "synthetic"
        expected_fen = parse_fen_from_filename(filename)
        if not image_path.exists():
            raise FileNotFoundError(f"Board image not found: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read board image: {filename}")

    # Step 1: overlay detection timing
    t_overlay = time.monotonic()
    det = fast_overlay_check(image)
    overlay_detect_ms = round((time.monotonic() - t_overlay) * 1000, 1)

    # Step 2: grid detection timing
    t_grid = time.monotonic()
    detected_grid = detect_grid(image)
    grid_detect_ms = round((time.monotonic() - t_grid) * 1000, 1)

    if is_real:
        if detected_grid is None:
            elapsed_ms = overlay_detect_ms + grid_detect_ms
            return {
                "filename": filename,
                "source": source,
                "expected_fen": expected_fen,
                "predicted_fen": None,
                "match": False,
                "piece_accuracy": 0.0,
                "square_diffs": ["grid detection failed"],
                "board_image_b64": _frame_to_base64(image),
                "elapsed_ms": elapsed_ms,
                "overlay_detect_ms": overlay_detect_ms,
                "grid_detect_ms": grid_detect_ms,
                "piece_classify_ms": 0.0,
                "error": "grid detection failed",
            }
        grid = detected_grid
    else:
        if image.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"Invalid board image size: {filename}")
        grid = _GRID  # use fixed grid for accuracy on synthetic

    # Step 3: piece classification timing
    # detect_orientation=False for synthetic (always white-at-bottom);
    # True for real (orientation unknown).
    t_classify = time.monotonic()
    try:
        predicted_fen = read_fen_with_grid(
            image, grid, device="cpu", detect_orientation=is_real
        )
    except Exception as e:
        piece_classify_ms = round((time.monotonic() - t_classify) * 1000, 1)
        elapsed_ms = overlay_detect_ms + grid_detect_ms + piece_classify_ms
        return {
            "filename": filename,
            "source": source,
            "expected_fen": expected_fen,
            "predicted_fen": None,
            "match": False,
            "piece_accuracy": 0.0,
            "square_diffs": [str(e)],
            "board_image_b64": _frame_to_base64(image),
            "elapsed_ms": elapsed_ms,
            "overlay_detect_ms": overlay_detect_ms,
            "grid_detect_ms": grid_detect_ms,
            "piece_classify_ms": piece_classify_ms,
            "error": str(e),
        }

    piece_classify_ms = round((time.monotonic() - t_classify) * 1000, 1)
    elapsed_ms = overlay_detect_ms + grid_detect_ms + piece_classify_ms

    expected_board = _fen_to_board(expected_fen)
    predicted_board = _fen_to_board(predicted_fen)
    mismatches = _diff_boards(predicted_board, expected_board)

    correct_squares = 64 - len(mismatches)
    piece_accuracy = correct_squares / 64

    return {
        "filename": filename,
        "source": source,
        "expected_fen": expected_fen,
        "predicted_fen": predicted_fen,
        "match": len(mismatches) == 0,
        "piece_accuracy": round(piece_accuracy, 4),
        "square_diffs": mismatches,
        "board_image_b64": _frame_to_base64(image),
        "elapsed_ms": elapsed_ms,
        "overlay_detect_ms": overlay_detect_ms,
        "grid_detect_ms": grid_detect_ms,
        "piece_classify_ms": piece_classify_ms,
    }


# ── Real overlay extraction ─────────────────────────────────




def _get_video_path(video_id: str) -> str | None:
    """Find the local path for a downloaded video, or None."""
    from pipeline.paths import find_video_file

    path = find_video_file(video_id)
    return str(path) if path is not None else None




def _is_valid_fen_placement(fen: str) -> bool:
    """Return True if the FEN piece-placement string has both kings."""
    try:
        board = chess.Board(fen + " w - - 0 1" if " " not in fen else fen)
        return (
            board.king(chess.WHITE) is not None
            and board.king(chess.BLACK) is not None
        )
    except Exception:
        return False



def _build_fen_from_class_grid(class_grid: list[list[int]]) -> str:
    """Build a FEN piece-placement string from a classify_squares grid."""
    piece_map = {
        1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
        7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
    }
    fen_rows = []
    for row_squares in class_grid:
        empties = 0
        fen_row = ""
        for cls in row_squares:
            if cls == 0:
                empties += 1
            else:
                if empties:
                    fen_row += str(empties)
                    empties = 0
                fen_row += piece_map.get(cls, "?")
        if empties:
            fen_row += str(empties)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def _get_saved_clip_ids() -> set[int]:
    """Return clip IDs that already have files in test_real/.

    Filenames follow the pattern ``r{clip_id}_{fen}.jpg``.
    """
    saved: set[int] = set()
    if not REAL_OVERLAY_TEST_DIR.exists():
        return saved
    for fname in os.listdir(REAL_OVERLAY_TEST_DIR):
        if fname.startswith("r") and "_" in fname:
            try:
                saved.add(int(fname.split("_", 1)[0][1:]))
            except ValueError:
                continue
    return saved


def _parse_frame_filename(stem: str) -> tuple[str, str, str]:
    """Parse ``f_{video_id}_{frame_name}_{fen_hyphenated}`` into parts.

    Searches for known frame name markers to handle video IDs that contain
    underscores (e.g. ``ycitHs8_NY4``).

    Returns ``(video_id, frame_name, fen)`` where fen uses ``/`` separators.
    """
    for fname in _KNOWN_FRAME_NAMES:
        marker = f"_{fname}_"
        idx = stem.find(marker)
        if idx > 0:
            video_id = stem[2:idx]  # skip "f_" prefix
            fen_hyphenated = stem[idx + len(marker):]
            return video_id, fname, fen_hyphenated.replace("-", "/")
    raise ValueError(f"Cannot parse frame filename: {stem}")


def _get_saved_frame_keys() -> set[str]:
    """Return frame keys (``video_id:frame_name``) already saved in val_real/."""
    saved: set[str] = set()
    if not REAL_OVERLAY_TEST_DIR.exists():
        return saved
    for fname in os.listdir(REAL_OVERLAY_TEST_DIR):
        if not fname.startswith("f_"):
            continue
        try:
            vid, frame_name, _ = _parse_frame_filename(Path(fname).stem)
            saved.add(f"{vid}:{frame_name}")
        except ValueError:
            continue
    return saved


def get_extraction_candidates(
    limit: int = 200,
    video_ids: list[str] | None = None,
) -> list[str]:
    """Return video IDs that have extraction frames on disk and unsaved samples.

    Approved videos without an explicit ``layout_type`` are treated as overlay,
    matching the rest of the screening pipeline.
    """
    saved_keys = _get_saved_frame_keys()

    if video_ids:
        raw = video_ids
    else:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT video_id FROM youtube_videos
                       WHERE {_APPROVED_OVERLAY_WHERE}
                       ORDER BY random()
                       LIMIT %s""",
                    (limit * 3,),
                )
                raw = [row[0] for row in cur.fetchall()]

    result: list[str] = []
    for vid in raw:
        if len(result) >= limit:
            break
        if _video_has_unsaved_frames(vid, saved_keys):
            result.append(vid)
    return result


def extract_overlay_from_frames(video_id: str) -> dict:
    """Process one video's cached screening frames and return the best overlay.

    Tries frames (25pct, 50pct, 75pct), preferring hi-res overlay frames
    from fullres/hires tiers over low-res screening frames.

    Pipeline per frame: ``detect_overlay_fast`` (seed + expansion) →
    crop to bbox → ``detect_grid`` → ``classify_squares`` → FEN.

    Returns a dict with ``status`` of ``ok``, ``warning``, or ``no_overlay``.
    """
    saved_keys = _get_saved_frame_keys()

    if not _video_has_frames(video_id):
        return {"video_id": video_id, "status": "no_overlay"}

    for frame_name in _EXTRACTION_FRAME_NAMES:
        if f"{video_id}:{frame_name}" in saved_keys:
            continue
        frame_path = _resolve_frame_path(video_id, frame_name)
        if frame_path is None:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        det = detect_overlay_fast(frame)
        if not det.found or det.bbox is None:
            continue

        bx, by, bw, bh = det.bbox
        fh, fw = frame.shape[:2]
        crop = frame[max(0, by):min(fh, by + bh), max(0, bx):min(fw, bx + bw)]
        if crop.size == 0:
            continue

        grid = detect_grid(crop)
        if grid is None:
            continue

        # Tighten crop to just the chess grid
        gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
        gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
        board_crop = crop[gy1:gy2, gx1:gx2]

        try:
            squares = grid.crop_squares(crop)
            class_grid = classify_squares(squares)
            fen = _build_fen_from_class_grid(class_grid)
        except Exception:
            continue

        piece_count = sum(c != 0 for row in class_grid for c in row)
        if piece_count < 2:
            continue

        frame_key = f"{video_id}:{frame_name}"
        entry: dict = {
            "frame_key": frame_key,
            "video_id": video_id,
            "frame_name": frame_name,
            "image_b64": _frame_to_base64(board_crop, max_width=400),
            "predicted_fen": fen,
        }

        if not _is_valid_fen_placement(fen):
            entry["status"] = "warning"
            entry["warning"] = "FEN may be invalid (missing king)"
        else:
            entry["status"] = "ok"

        return entry

    return {"video_id": video_id, "status": "no_overlay"}


def detect_overlay_from_frames(video_id: str) -> dict:
    """Fast phase: detect overlay + grid, return crop image without FEN.

    Uses ``detect_overlay_fast`` which expands the initial seed bbox via
    grid detection to find pixel-perfect board boundaries.  Returns
    immediately so the UI can show the overlay crop while FEN
    classification happens asynchronously.
    """
    saved_keys = _get_saved_frame_keys()

    if not _video_has_frames(video_id):
        logger.info("detect_overlay: %s — no frames on disk", video_id)
        return {"video_id": video_id, "status": "no_overlay"}

    for frame_name in _EXTRACTION_FRAME_NAMES:
        if f"{video_id}:{frame_name}" in saved_keys:
            continue
        frame_path = _resolve_frame_path(video_id, frame_name)
        if frame_path is None:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        det = detect_overlay_fast(frame)
        if not det.found or det.bbox is None:
            continue

        bx, by, bw, bh = det.bbox
        fh, fw = frame.shape[:2]
        crop = frame[max(0, by):min(fh, by + bh), max(0, bx):min(fw, bx + bw)]
        if crop.size == 0:
            continue

        grid = detect_grid(crop)
        if grid is None:
            continue

        gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
        gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
        board_crop = crop[gy1:gy2, gx1:gx2]

        frame_key = f"{video_id}:{frame_name}"
        logger.info("detect_overlay: %s — found in %s", video_id, frame_name)
        return {
            "frame_key": frame_key,
            "video_id": video_id,
            "frame_name": frame_name,
            "image_b64": _frame_to_base64(board_crop, max_width=400),
            "status": "detected",
        }

    logger.info("detect_overlay: %s — no overlay in any frame", video_id)
    return {"video_id": video_id, "status": "no_overlay"}


def classify_overlay_fen(video_id: str, frame_name: str) -> dict:
    """Slow phase: run DINOv2 piece classification on a detected overlay.

    Re-loads the frame and re-detects overlay + grid (~40ms overhead),
    then runs ``classify_squares`` to produce FEN (~500ms+ on CPU).
    """
    frame_path = _resolve_frame_path(video_id, frame_name)
    if frame_path is None:
        return {"status": "error", "warning": "Frame not found on disk"}

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return {"status": "error", "warning": "Could not read frame"}

    det = detect_overlay_fast(frame)
    if not det.found or det.bbox is None:
        return {"status": "error", "warning": "Overlay not detected"}

    bx, by, bw, bh = det.bbox
    fh, fw = frame.shape[:2]
    crop = frame[max(0, by):min(fh, by + bh), max(0, bx):min(fw, bx + bw)]
    if crop.size == 0:
        return {"status": "error", "warning": "Empty crop"}

    grid = detect_grid(crop)
    if grid is None:
        return {"status": "error", "warning": "Grid detection failed"}

    try:
        squares = grid.crop_squares(crop)
        class_grid = classify_squares(squares)
        fen = _build_fen_from_class_grid(class_grid)
    except Exception as e:
        return {"status": "error", "warning": str(e)}

    piece_count = sum(c != 0 for row in class_grid for c in row)
    if piece_count < 2:
        return {"status": "error", "warning": "Too few pieces detected"}

    result: dict = {"predicted_fen": fen}

    if not _is_valid_fen_placement(fen):
        result["status"] = "warning"
        result["warning"] = "FEN may be invalid (missing king)"
    else:
        result["status"] = "ok"

    return result


def save_confirmed_frame_extractions(confirmations: list[dict]) -> dict:
    """Save user-confirmed frame overlay extractions to val_real/.

    Each confirmation should have: video_id, frame_name, fen.
    Re-reads the frame from disk, re-detects overlay, and saves the
    grid-tightened crop as ``f_{video_id}_{frame_name}_{fen_hyphenated}.jpg``.
    """
    REAL_OVERLAY_TEST_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    errors: list[str] = []

    for conf in confirmations:
        video_id = conf.get("video_id", "")
        frame_name = conf.get("frame_name", "")
        fen = conf.get("fen", "")
        label = f"{video_id}:{frame_name}"

        frame_path = _resolve_frame_path(video_id, frame_name)
        if frame_path is None:
            errors.append(f"{label}: frame not found on disk")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            errors.append(f"{label}: could not read frame")
            continue

        det = detect_overlay_fast(frame)
        if not det.found or det.bbox is None:
            errors.append(f"{label}: overlay not detected")
            continue

        bx, by, bw, bh = det.bbox
        fh, fw = frame.shape[:2]
        crop = frame[max(0, by):min(fh, by + bh), max(0, bx):min(fw, bx + bw)]
        if crop.size == 0:
            errors.append(f"{label}: empty crop")
            continue

        grid = detect_grid(crop)
        if grid is None:
            errors.append(f"{label}: grid detection failed")
            continue

        gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
        gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
        board_crop = crop[gy1:gy2, gx1:gx2]
        if board_crop.size == 0:
            errors.append(f"{label}: empty board crop")
            continue

        fen_hyphenated = fen.replace("/", "-")
        out_filename = f"f_{video_id}_{frame_name}_{fen_hyphenated}.jpg"
        out_path = REAL_OVERLAY_TEST_DIR / out_filename

        _, buf = cv2.imencode(".jpg", board_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())

        saved += 1
        logger.info(f"Saved frame sample: {out_filename}")

    return {"saved": saved, "errors": errors}


# ── Overlay Detection Validation ───────────────────────────


def validate_overlay_detection(limit: int = 100) -> dict:
    """Validate fast_overlay_check accuracy on real video frames.

    Samples frames from downloaded overlay videos across diverse channels.
    For each frame, runs fast_overlay_check and records whether the overlay
    was detected, the score, and the time taken.

    Args:
        limit: Target number of frames to test.

    Returns:
        Dict with per-frame results and aggregate metrics.
    """
    t0 = time.monotonic()

    # Find downloaded overlay videos across diverse channels
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT video_id, channel_handle
                FROM youtube_videos
                WHERE {_APPROVED_OVERLAY_WHERE}
                  AND channel_handle IS NOT NULL
                ORDER BY random()
                """
            )
            all_videos = cur.fetchall()

    # Find videos that are actually downloaded
    available: list[tuple[str, str, str]] = []  # (video_id, channel, path)
    for video_id, channel_handle in all_videos:
        path = _get_video_path(video_id)
        if path is not None:
            available.append((video_id, channel_handle, path))

    if not available:
        return {"error": "No downloaded overlay videos found", "results": []}

    # Sample 2-3 frames per video, distribute across channels
    frames_per_video = max(2, min(3, limit // max(1, len(available))))
    rng = random.Random(42)

    results: list[dict] = []
    channel_stats: dict[str, dict] = {}

    for video_id, channel_handle, video_path in available:
        if len(results) >= limit:
            break

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if duration < 90:
            cap.release()
            continue

        # Sample random timestamps, skipping first/last 30s
        safe_start = 30.0
        safe_end = max(safe_start + 10, duration - 30.0)
        timestamps = sorted(
            rng.uniform(safe_start, safe_end) for _ in range(frames_per_video)
        )

        for ts in timestamps:
            if len(results) >= limit:
                break

            frame_no = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fh, fw = frame.shape[:2]

            t_start = time.monotonic()
            det = fast_overlay_check(frame)
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

            result = {
                "video_id": video_id,
                "channel": channel_handle,
                "timestamp": round(ts, 1),
                "found": det.found,
                "score": round(det.score, 4) if det.found else 0.0,
                "bbox": det.bbox if det.found else None,
                "time_ms": elapsed_ms,
                "resolution": f"{fw}x{fh}",
            }
            results.append(result)

            # Track per-channel stats
            ch = channel_handle
            if ch not in channel_stats:
                channel_stats[ch] = {"total": 0, "found": 0, "times": []}
            channel_stats[ch]["total"] += 1
            if det.found:
                channel_stats[ch]["found"] += 1
            channel_stats[ch]["times"].append(elapsed_ms)

        cap.release()

    # Compute aggregate metrics
    total = len(results)
    found_count = sum(1 for r in results if r["found"])
    detection_rate = found_count / total if total > 0 else 0.0
    all_times = [r["time_ms"] for r in results]
    avg_time = sum(all_times) / len(all_times) if all_times else 0.0
    sorted_times = sorted(all_times)
    p95_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0.0

    # Per-channel summary
    per_channel = []
    for ch, stats in sorted(channel_stats.items(), key=lambda x: x[0]):
        rate = stats["found"] / stats["total"] if stats["total"] > 0 else 0.0
        ch_times = stats["times"]
        per_channel.append({
            "channel": ch,
            "total": stats["total"],
            "found": stats["found"],
            "detection_rate": round(rate, 4),
            "avg_time_ms": round(sum(ch_times) / len(ch_times), 1),
        })

    elapsed_total = round((time.monotonic() - t0) * 1000, 1)

    return {
        "total_frames": total,
        "detected": found_count,
        "detection_rate": round(detection_rate, 4),
        "avg_time_ms": round(avg_time, 1),
        "p95_time_ms": round(p95_time, 1),
        "per_channel": per_channel,
        "num_channels": len(channel_stats),
        "num_videos": len({r["video_id"] for r in results}),
        "elapsed_ms": elapsed_total,
        "results": results,
    }


# ── Evaluation persistence ──────────────────────────────────


def save_overlay_eval(
    accuracy: float,
    sample_size: int,
    piece_accuracy: float | None = None,
    images_per_minute: int | None = None,
    notes: str | None = None,
) -> dict:
    """Save overlay evaluation result to model_evaluations."""
    per_class_data: dict = {}
    if piece_accuracy is not None:
        per_class_data["piece_accuracy"] = piece_accuracy
    if images_per_minute is not None:
        per_class_data["images_per_minute"] = images_per_minute

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_evaluations
                   (model_name, sample_size, accuracy, notes, per_class)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, evaluated_at""",
                (
                    "overlay",
                    sample_size,
                    accuracy,
                    notes,
                    json.dumps(per_class_data) if per_class_data else None,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    return {"id": row[0], "evaluated_at": str(row[1])}


# ── Session CRUD ────────────────────────────────────────────


def create_overlay_test_session(
    results: list[dict],
    accuracy: float | None,
    sample_size: int,
    piece_accuracy: float | None = None,
    pin_state: dict | None = None,
    evaluation_id: int | None = None,
) -> dict:
    """Create and persist an overlay test session."""
    session_id = uuid.uuid4().hex[:12]

    # Strip heavy image data for storage
    lightweight = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "board_image_b64"}
        lightweight.append(entry)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO overlay_test_sessions
                   (id, sample_size, accuracy, piece_accuracy, results, pin_state, evaluation_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id,
                    sample_size,
                    accuracy,
                    piece_accuracy,
                    json.dumps(lightweight),
                    json.dumps(pin_state or {}),
                    evaluation_id,
                ),
            )
            conn.commit()
    return {"session_id": session_id}


def get_overlay_test_session(session_id: str) -> dict | None:
    """Fetch an overlay test session by ID."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, accuracy, piece_accuracy,
                          results, pin_state, evaluation_id
                   FROM overlay_test_sessions WHERE id = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "created_at": str(row[1]),
                "sample_size": row[2],
                "accuracy": row[3],
                "piece_accuracy": row[4],
                "results": row[5] if isinstance(row[5], list) else json.loads(row[5]),
                "pin_state": row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}"),
                "evaluation_id": row[7],
            }


def list_overlay_test_sessions(limit: int = 20) -> list[dict]:
    """List recent overlay test sessions (lightweight, no results)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, accuracy, piece_accuracy
                   FROM overlay_test_sessions
                   ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            return [
                {
                    "id": row[0],
                    "created_at": str(row[1]),
                    "sample_size": row[2],
                    "accuracy": row[3],
                    "piece_accuracy": row[4],
                }
                for row in cur.fetchall()
            ]


def update_overlay_session_pins(session_id: str, pin_state: dict) -> dict:
    """Merge pin state updates into an overlay test session."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pin_state FROM overlay_test_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Session {session_id} not found"}

            existing = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
            existing.update(pin_state)

            cur.execute(
                "UPDATE overlay_test_sessions SET pin_state = %s WHERE id = %s",
                (json.dumps(existing), session_id),
            )
            conn.commit()
    return {"ok": True, "pin_state": existing}
