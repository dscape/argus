"""Service layer for overlay (piece classifier) accuracy testing.

Uses the overlay/positions dataset (400x400 board images with FEN in filenames)
to evaluate the DINOv2 piece classifier. Also supports real video overlay crops
extracted from stored video_clips. Mirrors the screening inspector pattern:
sample → inspect → save session.
"""

import base64
import glob as glob_mod
import logging
import os
import random
import time
import urllib.request
import uuid
from pathlib import Path

import chess
import cv2
import json
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.overlay.scanner import OverlayDetection, detect_overlay_in_frame, fast_overlay_check
from pipeline.overlay.chess_positions_data import (
    BOARD_SIZE,
    SQ_SIZE,
    parse_fen_from_filename,
)
from pipeline.overlay.grid_detector import GridResult, detect_grid
from pipeline.overlay.piece_classifier import classify_squares, read_fen_with_grid

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
CHESS_POSITIONS_TEST_DIR = _PROJECT_ROOT / "data" / "overlay" / "val"
# Real overlay crops extracted from video_clips
REAL_OVERLAY_TEST_DIR = _PROJECT_ROOT / "data" / "overlay" / "val_real"
# Prefix prepended to real sample filenames in the sample list so inspect_board
# can route to the right directory without path separators (which the board-image
# endpoint rejects for security).
_REAL_PREFIX = "real__"

# Uniform grid for 400×400 chess-positions boards (50px squares)
_GRID = GridResult(
    v_lines=list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
    h_lines=list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
    sq_size=SQ_SIZE,
)


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
        # FEN encoded in filename as "r{clip_id}_{fen_hyphenated}.ext"
        stem = Path(actual_name).stem
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


def _fetch_youtube_thumbnail(video_id: str, index: int) -> np.ndarray | None:
    """Fetch YouTube timeline thumbnail {index} (0-3) for video_id.

    YouTube provides 4 timeline thumbnails at different timestamps via
    img.youtube.com/vi/{id}/{0-3}.jpg — no video download needed.
    Returns None if the fetch fails or the image cannot be decoded.
    """
    url = f"https://img.youtube.com/vi/{video_id}/{index}.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "argus/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"Thumbnail {index} for {video_id} unavailable: {e}")
        return None


def _find_overlay_thumbnail(
    video_id: str,
) -> tuple[np.ndarray | None, OverlayDetection | None]:
    """Try all 4 YouTube thumbnails and return the first one where overlay is detected.

    Uses detect_overlay_in_frame (with grid-line validation) so that false
    positives from non-chess content in thumbnails (faces, flags, scoreboards)
    are rejected.  Stops early once a confirmed overlay is found.  Falls back
    to returning the first available thumbnail (with found=False) if none of
    the 4 has an overlay.
    """
    first_frame: np.ndarray | None = None
    first_det: OverlayDetection | None = None

    for index in range(4):
        frame = _fetch_youtube_thumbnail(video_id, index)
        if frame is None:
            continue
        if first_frame is None:
            first_frame = frame

        det = detect_overlay_in_frame(frame)
        if det.found:
            return frame, det

        if first_det is None:
            first_det = det

    return first_frame, first_det


def _get_video_path(video_id: str) -> str | None:
    """Find the local path for a downloaded video, or None."""
    base_dirs = [
        "data/videos",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "videos"),
    ]
    for base in base_dirs:
        base = os.path.normpath(base)
        for ext in ("mp4", "mkv", "webm"):
            pattern = os.path.join(base, "*", f"{video_id}.{ext}")
            matches = glob_mod.glob(pattern)
            if matches:
                return matches[0]
    return None


def _detect_from_video(
    video_path: str,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray | None, OverlayDetection | None]:
    """Read frames at several timestamps within [start_time, end_time] and
    return the best frame where detect_overlay_in_frame succeeds.

    Uses fast_overlay_check (~50ms each) to scan 4 timestamps spread across
    the clip range, then runs detect_overlay_in_frame (~1.5s) only on the
    best candidate.

    If no timestamp passes fast_overlay_check, returns (fallback_frame,
    not-found) to avoid false positives from running expensive detection
    on frames without an overlay.
    """
    cap = cv2.VideoCapture(video_path)
    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    span = end_time - start_time

    # Fast pre-scan: find the timestamp most likely to have an overlay.
    best_frame: np.ndarray | None = None
    best_fast_score = -1.0
    fallback_frame: np.ndarray | None = None

    # Sample timestamps spread across the clip range.
    timestamps = [
        start_time + span * f
        for f in [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
    ]
    # Collect all fast-check hits, sorted by area (largest first).
    candidates: list[tuple[int, np.ndarray, float]] = []  # (area, frame, score)
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if fallback_frame is None:
            fallback_frame = frame
        fast = fast_overlay_check(frame)
        if fast.found and fast.bbox is not None:
            area = fast.bbox[2] * fast.bbox[3]
            candidates.append((area, frame, fast.score))

    cap.release()

    # Try candidates largest-first — run detect_overlay_in_frame and
    # accept the first that produces a valid expanded detection.
    candidates.sort(key=lambda c: c[0], reverse=True)
    for _, cand_frame, _ in candidates[:3]:
        det = detect_overlay_in_frame(cand_frame)
        if det.found and det.bbox is not None:
            # Verify expansion produced a reasonable board — if the
            # expanded bbox is barely larger than the seed, the seed
            # was likely a false positive (OTB camera view, scoreboard).
            if det.seed_bbox is not None:
                seed_area = det.seed_bbox[2] * det.seed_bbox[3]
                exp_area = det.bbox[2] * det.bbox[3]
                if exp_area > seed_area * 1.3:
                    return cand_frame, det
            else:
                return cand_frame, det

    # No timestamp passed fast check — return fallback with not-found
    # to avoid false positives from detecting non-overlay content.
    if fallback_frame is not None:
        fh, fw = fallback_frame.shape[:2]
        return fallback_frame, OverlayDetection(
            found=False, frame_resolution=(fw, fh),
        )
    return None, None


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


def _parse_bbox(raw) -> tuple[int, int, int, int]:
    """Parse overlay_bbox from either list [x,y,w,h] or dict {x,y,w,h}."""
    if isinstance(raw, (list, tuple)):
        return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
    if isinstance(raw, dict):
        return (
            int(raw.get("x", 0)),
            int(raw.get("y", 0)),
            int(raw.get("w", 0)),
            int(raw.get("h", 0)),
        )
    parsed = json.loads(raw)
    return _parse_bbox(parsed)


def _parse_ref_res(raw) -> tuple[int, int]:
    """Parse ref_resolution from either list [w,h] or dict {width,height}."""
    if isinstance(raw, (list, tuple)):
        return int(raw[0]), int(raw[1])
    if isinstance(raw, dict):
        return int(raw.get("width", 1920)), int(raw.get("height", 1080))
    parsed = json.loads(raw)
    return _parse_ref_res(parsed)


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


def preview_real_overlay_extractions(
    limit: int = 200,
    video_ids: list[str] | None = None,
) -> list[dict]:
    """Extract and preview overlay crops from video_clips WITHOUT saving.

    Returns a list of dicts with clip_id, video_id, crop as base64,
    auto-labeled FEN, and status information for user review.

    For each clip, reads the local video at start_time (giving a distinct
    frame per clip — no duplicate stills).  Falls back to YouTube thumbnails
    (img.youtube.com/vi/{id}/{0-3}.jpg, all 4 tried, first with overlay wins)
    when the local video is unavailable.  Stored overlay_bbox values are
    intentionally ignored so scanner improvements are picked up automatically.

    When *video_ids* is provided only clips belonging to those videos are
    returned — this is much faster than processing the entire table.
    """
    saved_ids = _get_saved_clip_ids()

    with get_conn() as conn:
        with conn.cursor() as cur:
            conditions: list[str] = []
            params: list = []

            if video_ids:
                conditions.append("video_id = ANY(%s)")
                params.append(video_ids)
            if saved_ids:
                conditions.append("id != ALL(%s)")
                params.append(list(saved_ids))

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)

            cur.execute(
                f"""SELECT id, video_id, clip_index, start_time, end_time
                    FROM video_clips
                    {where}
                    ORDER BY random()
                    LIMIT %s""",
                params,
            )
            rows = cur.fetchall()

    results: list[dict] = []
    # Thumbnail fallback cache: only used when no local video is available.
    # All clips from the same video share the same thumbnail result.
    video_thumb_cache: dict[str, tuple[np.ndarray | None, OverlayDetection | None]] = {}

    for row in rows:
        clip_id, video_id, clip_index, start_time, end_time = row
        entry: dict = {
            "clip_id": clip_id,
            "video_id": video_id,
            "start_time": start_time,
            "status": "pending",
        }

        frame: np.ndarray | None = None
        det: OverlayDetection | None = None

        # Primary: local video, trying a few timestamps within the clip range
        # so a bad transition frame at start_time doesn't ruin the preview.
        video_path = _get_video_path(video_id)
        if video_path is not None:
            frame, det = _detect_from_video(video_path, start_time, end_time)

        # Fallback: YouTube thumbnails (cached per video to avoid redundant requests).
        if frame is None:
            if video_id not in video_thumb_cache:
                video_thumb_cache[video_id] = _find_overlay_thumbnail(video_id)
            frame, det = video_thumb_cache[video_id]

        if frame is None:
            entry["status"] = "error"
            entry["error"] = "no video source available (local video missing, thumbnails failed)"
            results.append(entry)
            continue

        if not det.found or det.bbox is None:
            # Show the full thumbnail so the user can see what the frame looks like.
            entry["image_b64"] = _frame_to_base64(frame, max_width=400)
            entry["status"] = "no_overlay"
            results.append(entry)
            continue

        bx, by, bw, bh = det.bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(fw, bx + bw)
        y2 = min(fh, by + bh)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            entry["status"] = "error"
            entry["error"] = "empty crop after detection"
            results.append(entry)
            continue

        grid = detect_grid(crop)
        if grid is None:
            entry["status"] = "error"
            entry["error"] = "grid detection failed on detected crop"
            results.append(entry)
            continue

        # Tighten crop to just the chess grid (exclude player names,
        # clocks, tournament headers, etc.).
        gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
        gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
        board_crop = crop[gy1:gy2, gx1:gx2]
        entry["image_b64"] = _frame_to_base64(board_crop, max_width=400)

        try:
            squares = grid.crop_squares(crop)
            class_grid = classify_squares(squares)
            fen = _build_fen_from_class_grid(class_grid)
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = f"classification failed: {e}"
            results.append(entry)
            continue

        # Fewer than 2 classified pieces means the crop is almost certainly
        # a false positive (flag, scoreboard) — real boards always have kings.
        piece_count = sum(c != 0 for row in class_grid for c in row)
        if piece_count < 2:
            entry["image_b64"] = _frame_to_base64(frame, max_width=400)
            entry["status"] = "no_overlay"
            results.append(entry)
            continue

        if not _is_valid_fen_placement(fen):
            entry["status"] = "warning"
            entry["warning"] = "FEN may be invalid (missing king)"
            entry["predicted_fen"] = fen
            results.append(entry)
            continue

        entry["status"] = "ok"
        entry["predicted_fen"] = fen
        results.append(entry)

    return results


def save_confirmed_extractions(confirmations: list[dict]) -> dict:
    """Save confirmed overlay extractions to test_real/ directory.

    Each confirmation should have: clip_id, video_id, fen (confirmed by user).
    Re-detects the overlay from the video (same as preview) and saves the crop.
    """
    REAL_OVERLAY_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Look up clip data
    clip_ids = [c["clip_id"] for c in confirmations]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, video_id, start_time, end_time
                   FROM video_clips
                   WHERE id = ANY(%s)""",
                (clip_ids,),
            )
            rows = {row[0]: row for row in cur.fetchall()}

    fen_lookup = {c["clip_id"]: c["fen"] for c in confirmations}
    saved = 0
    errors: list[str] = []

    for clip_id, fen in fen_lookup.items():
        row = rows.get(clip_id)
        if row is None:
            errors.append(f"clip {clip_id}: not found in DB")
            continue

        _, video_id, start_time, end_time = row

        video_path = _get_video_path(video_id)
        if video_path is None:
            errors.append(f"clip {clip_id}: video not found")
            continue

        frame, det = _detect_from_video(video_path, start_time, end_time)
        if frame is None or det is None or not det.found or det.bbox is None:
            errors.append(f"clip {clip_id}: overlay not detected")
            continue

        bx, by, bw, bh = det.bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(fw, bx + bw)
        y2 = min(fh, by + bh)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            errors.append(f"clip {clip_id}: empty crop")
            continue

        # Tighten crop to just the chess grid.
        grid = detect_grid(crop)
        if grid is None:
            errors.append(f"clip {clip_id}: grid detection failed")
            continue
        gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
        gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
        board_crop = crop[gy1:gy2, gx1:gx2]
        if board_crop.size == 0:
            errors.append(f"clip {clip_id}: empty board crop")
            continue

        # Filename uses FEN with / replaced by - (same pattern as synthetic)
        fen_hyphenated = fen.replace("/", "-")
        out_filename = f"r{clip_id}_{fen_hyphenated}.jpg"
        out_path = REAL_OVERLAY_TEST_DIR / out_filename

        _, buf = cv2.imencode(".jpg", board_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())

        saved += 1
        logger.info(f"Saved real sample: {out_filename}")

    return {"saved": saved, "errors": errors}


def extract_real_overlay_samples(limit: int = 200) -> dict:
    """Extract real board crops from stored video_clips and save as test samples.

    For each video_clip:
    1. Opens the local video file at the clip mid-point.
    2. Crops the overlay region using the stored overlay_bbox.
    3. Runs detect_grid() + classify_squares() to produce a pseudo-label FEN.
    4. Saves the crop to REAL_OVERLAY_TEST_DIR as r{clip_id}_{fen_hyphenated}.jpg.

    Returns a dict with counts of processed / saved / skipped clips.
    """
    REAL_OVERLAY_TEST_DIR.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, video_id, start_time, end_time,
                          overlay_bbox, ref_resolution
                   FROM video_clips
                   ORDER BY id
                   LIMIT %s""",
                (limit,),
            )
            rows = cur.fetchall()

    processed = 0
    saved = 0
    skipped = 0

    for row in rows:
        clip_id, video_id, start_time, end_time, overlay_bbox_raw, ref_res_raw = row
        t0 = time.monotonic()

        try:
            overlay_bbox = (
                overlay_bbox_raw
                if isinstance(overlay_bbox_raw, dict)
                else json.loads(overlay_bbox_raw)
            )
            ref_res = (
                ref_res_raw
                if isinstance(ref_res_raw, dict)
                else json.loads(ref_res_raw)
            )
        except Exception:
            skipped += 1
            continue

        video_path = _get_video_path(video_id)
        if video_path is None:
            skipped += 1
            continue

        # Sample at the clip mid-point
        mid_time = (start_time + (end_time or start_time + 30)) / 2

        cap = cv2.VideoCapture(video_path)
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_time * fps))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            skipped += 1
            continue

        processed += 1

        # Scale bbox from reference resolution to actual frame size
        fh, fw = frame.shape[:2]
        ref_w = ref_res.get("width", fw)
        ref_h = ref_res.get("height", fh)
        sx = fw / ref_w
        sy = fh / ref_h

        ox = int(overlay_bbox.get("x", 0) * sx)
        oy = int(overlay_bbox.get("y", 0) * sy)
        ow = int(overlay_bbox.get("w", fw) * sx)
        oh = int(overlay_bbox.get("h", fh) * sy)
        x1, y1 = max(0, ox), max(0, oy)
        x2, y2 = min(fw, ox + ow), min(fh, oy + oh)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            skipped += 1
            continue

        # Detect grid and classify pieces for pseudo-label
        grid = detect_grid(crop)
        if grid is None:
            skipped += 1
            continue

        try:
            squares = grid.crop_squares(crop)
            class_grid = classify_squares(squares)
            # Build FEN piece-placement from class grid
            fen_rows = []
            for row_squares in class_grid:
                empties = 0
                fen_row = ""
                piece_map = {
                    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
                    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
                }
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
            fen = "/".join(fen_rows)
        except Exception:
            skipped += 1
            continue

        if not _is_valid_fen_placement(fen):
            skipped += 1
            continue

        fen_hyphenated = fen.replace("/", "-")
        out_filename = f"r{clip_id}_{fen_hyphenated}.jpg"
        out_path = REAL_OVERLAY_TEST_DIR / out_filename

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())

        saved += 1
        logger.debug(
            f"Saved real sample clip={clip_id} video={video_id} "
            f"in {(time.monotonic() - t0)*1000:.0f}ms → {out_filename}"
        )

    logger.info(
        f"extract_real_overlay_samples: processed={processed} saved={saved} skipped={skipped}"
    )
    return {"processed": processed, "saved": saved, "skipped": skipped}


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
                """
                SELECT video_id, channel_handle
                FROM youtube_videos
                WHERE layout_type = 'overlay'
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
