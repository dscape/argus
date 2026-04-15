"""Service layer for overlay (piece classifier) accuracy testing.

Uses local overlay board datasets when available and falls back to committed
test fixtures so the evaluation UIs remain usable in fresh clones. Also
supports real overlay crops extracted from screening frames. Mirrors the
screening inspector pattern: sample → inspect → save session.
"""

import base64
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
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
from pipeline.overlay.piece_classifier import class_grid_to_fen, read_fen_with_grid
from pipeline.overlay.real_board_data import parse_real_board_fen
from pipeline.overlay.scanner import (
    compute_alternation_strength,
    compute_axis_aligned_periodicity,
    compute_grid_regularity,
    detect_overlay_runtime,
    runtime_overlay_check,
)
from pipeline.paths import find_frame

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
CHESS_POSITIONS_TEST_DIR = _PROJECT_ROOT / "data" / "overlay" / "val"
BOARD_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "boards"
FIXTURE_FRAMES_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "frames"
FIXTURE_FRAME_GROUND_TRUTH_PATH = FIXTURE_FRAMES_DIR / "ground_truth.json"
# Real overlay crops extracted from video_clips
REAL_OVERLAY_TEST_DIR = _PROJECT_ROOT / "data" / "overlay" / "val_real"
# Prefix prepended to real sample filenames in the sample list so inspect_board
# can route to the right directory without path separators (which the board-image
# endpoint rejects for security).
_REAL_PREFIX = "real__"
_IMAGE_EXTS = (".jpeg", ".jpg", ".png")

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
    """Return the best available frame path from local cache or committed fixtures."""
    for tier in ("fullres", "hires", "lores"):
        path = find_frame(video_id, tier, frame_name)  # type: ignore[arg-type]
        if path is not None:
            return path

    fixture_path = FIXTURE_FRAMES_DIR / video_id / f"{frame_name}.jpg"
    if fixture_path.exists():
        return fixture_path
    return None


def _fixture_frame_annotations_for_video(video_id: str) -> dict[str, dict[str, object]]:
    ground_truth = _load_fixture_frame_ground_truth()
    result: dict[str, dict[str, object]] = {}
    prefix = f"{video_id}/"
    for key, annotation in ground_truth.items():
        if not key.startswith(prefix):
            continue
        frame_name = key.split("/", 1)[1]
        result[frame_name] = annotation
    return result


def _fixture_overlay_frame_names(video_id: str) -> list[str]:
    annotations = _fixture_frame_annotations_for_video(video_id)
    return [
        frame_name
        for frame_name in _EXTRACTION_FRAME_NAMES
        if bool(annotations.get(frame_name, {}).get("has_overlay"))
    ]


def _fixture_overlay_bbox(
    video_id: str,
    frame_name: str,
) -> tuple[int, int, int, int] | None:
    annotation = _fixture_frame_annotations_for_video(video_id).get(frame_name)
    if annotation is None or not bool(annotation.get("has_overlay")):
        return None
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return tuple(int(value) for value in bbox)


def _overlay_eval_frame_names(video_id: str) -> list[str]:
    annotated = [
        frame_name
        for frame_name in _fixture_overlay_frame_names(video_id)
        if _resolve_frame_path(video_id, frame_name) is not None
    ]
    if annotated:
        return annotated
    return [
        frame_name
        for frame_name in _EXTRACTION_FRAME_NAMES
        if _resolve_frame_path(video_id, frame_name) is not None
    ]


def _video_has_frames(video_id: str) -> bool:
    """Check if a video has any overlay-eval candidate frames available."""
    return bool(_overlay_eval_frame_names(video_id))


def _video_has_unsaved_frames(video_id: str, saved_keys: set[str]) -> bool:
    """Return True when at least one candidate frame exists and is not saved."""
    return any(
        f"{video_id}:{frame_name}" not in saved_keys
        for frame_name in _overlay_eval_frame_names(video_id)
    )


def _frame_to_base64(frame: np.ndarray, max_width: int | None = 400) -> str:
    h, w = frame.shape[:2]
    if max_width is not None and w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _list_image_filenames(directory: Path, *, exclude: set[str]) -> list[str]:
    if not directory.exists():
        return []
    return [
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS and path.name not in exclude
    ]


def _resolve_synthetic_board_path(filename: str) -> Path | None:
    for root in (CHESS_POSITIONS_TEST_DIR, BOARD_FIXTURES_DIR):
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def resolve_board_image_path(filename: str) -> Path:
    if filename.startswith(_REAL_PREFIX):
        actual_name = filename[len(_REAL_PREFIX):]
        image_path = REAL_OVERLAY_TEST_DIR / actual_name
    else:
        image_path = _resolve_synthetic_board_path(filename)

    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Board image not found: {filename}")
    return image_path


def _load_fixture_frame_ground_truth() -> dict[str, dict[str, object]]:
    if not FIXTURE_FRAME_GROUND_TRUTH_PATH.exists():
        return {}
    data = json.loads(FIXTURE_FRAME_GROUND_TRUTH_PATH.read_text())
    return data if isinstance(data, dict) else {}


def _fixture_candidate_video_ids(saved_keys: set[str], limit: int) -> list[str]:
    del saved_keys
    ground_truth = _load_fixture_frame_ground_truth()
    video_ids = list(
        {
            key.split("/", 1)[0]
            for key, annotation in ground_truth.items()
            if bool(annotation.get("has_overlay"))
        }
    )
    random.shuffle(video_ids)

    result: list[str] = []
    for video_id in video_ids:
        if _video_has_frames(video_id):
            result.append(video_id)
            if len(result) >= limit:
                break
    return result


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


@dataclass(frozen=True)
class BoardCropResult:
    board_crop: np.ndarray
    overlay_detect_ms: float
    grid_detect_ms: float
    detector_found: bool = True
    warning: str | None = None


def _decode_image_b64(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def _uniform_grid_for_image(image: np.ndarray) -> GridResult:
    h, w = image.shape[:2]
    sq_w = w / 8
    sq_h = h / 8
    sq_size = int(round((sq_w + sq_h) / 2))
    return GridResult(
        v_lines=[int(round(c * sq_w)) for c in range(9)],
        h_lines=[int(round(r * sq_h)) for r in range(9)],
        sq_size=sq_size,
    )


def _board_alignment_score(image: np.ndarray) -> float:
    regularity = compute_grid_regularity(image)
    frac, contrast = compute_alternation_strength(image)
    periodicity = compute_axis_aligned_periodicity(image)
    return regularity * frac * contrast * periodicity


def _square_bbox_from_grid(
    image_shape: tuple[int, ...],
    grid: GridResult,
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
    gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
    side = max(gx2 - gx1, gy2 - gy1)
    x = max(0, min(gx1, w - side))
    y = max(0, min(gy1, h - side))
    side = min(side, w - x, h - y)
    return x, y, side, side


def _initial_board_crop_seeds(
    image: np.ndarray,
    grid: GridResult | None,
) -> list[tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    side = min(h, w)
    seeds = {
        (0, 0, side, side),
        (max(0, (w - side) // 2), max(0, (h - side) // 2), side, side),
    }
    if grid is not None:
        seeds.add(_square_bbox_from_grid(image.shape, grid))
    return list(seeds)


def _refine_square_bbox(
    image: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    seed_x, seed_y, seed_w, seed_h = seed_bbox
    seed_side = min(seed_w, seed_h, h, w)
    seed_x = max(0, min(seed_x, w - seed_side))
    seed_y = max(0, min(seed_y, h - seed_side))

    cell = max(1, seed_side // 8)
    step = max(2, cell // 6)
    size_slack = max(2, cell // 2)
    side_min = max(64, seed_side - size_slack)
    side_max = min(min(h, w), seed_side + size_slack)

    best_bbox = (seed_x, seed_y, seed_side, seed_side)
    best_score = _board_alignment_score(
        image[seed_y : seed_y + seed_side, seed_x : seed_x + seed_side]
    )

    for side in range(side_min, side_max + 1, step):
        min_x = max(0, seed_x - cell)
        max_x = min(w - side, seed_x + cell)
        min_y = max(0, seed_y - cell)
        max_y = min(h - side, seed_y + cell)

        for x in range(min_x, max_x + 1, step):
            for y in range(min_y, max_y + 1, step):
                score = _board_alignment_score(image[y : y + side, x : x + side])
                if score > best_score:
                    best_bbox = (x, y, side, side)
                    best_score = score

    return best_bbox


def _extract_board_crop_from_overlay_crop(overlay_crop: np.ndarray) -> BoardCropResult | None:
    if overlay_crop.size == 0:
        return None

    t_grid = time.monotonic()
    grid = detect_grid(overlay_crop, allow_uniform=False)
    grid_detect_ms = round((time.monotonic() - t_grid) * 1000, 1)

    best_bbox: tuple[int, int, int, int] | None = None
    best_score = -1.0
    for seed in _initial_board_crop_seeds(overlay_crop, grid):
        bbox = _refine_square_bbox(overlay_crop, seed)
        x, y, side, _ = bbox
        score = _board_alignment_score(overlay_crop[y : y + side, x : x + side])
        if score > best_score:
            best_bbox = bbox
            best_score = score

    if best_bbox is None:
        return None

    x, y, side, _ = best_bbox
    board_crop = overlay_crop[y : y + side, x : x + side]
    if board_crop.size == 0:
        return None

    return BoardCropResult(
        board_crop=board_crop,
        overlay_detect_ms=0.0,
        grid_detect_ms=grid_detect_ms,
    )


def _extract_board_crop_from_frame(frame: np.ndarray) -> BoardCropResult | None:
    t_overlay = time.monotonic()
    det = detect_overlay_runtime(frame)
    overlay_detect_ms = round((time.monotonic() - t_overlay) * 1000, 1)
    if not det.found or det.bbox is None:
        return None

    bx, by, bw, bh = det.bbox
    fh, fw = frame.shape[:2]
    overlay_crop = frame[max(0, by):min(fh, by + bh), max(0, bx):min(fw, bx + bw)]
    result = _extract_board_crop_from_overlay_crop(overlay_crop)
    if result is None:
        return None

    return BoardCropResult(
        board_crop=result.board_crop,
        overlay_detect_ms=overlay_detect_ms,
        grid_detect_ms=result.grid_detect_ms,
    )


def _read_board_crop_fen(board_crop: np.ndarray) -> str:
    grid = _uniform_grid_for_image(board_crop)
    return read_fen_with_grid(
        board_crop,
        grid,
        device="cpu",
        detect_orientation=True,
    )


def _piece_count_from_fen(fen: str) -> int:
    return sum(1 for ch in fen if ch.isalpha())


# ── Sampling ────────────────────────────────────────────────


def sample_board_filenames(
    limit: int = 20,
    exclude: list[str] | None = None,
) -> list[str]:
    """Return random sample of committed board fixtures for FEN inspection.

    ``/evaluate/fen`` is meant to isolate piece-classifier accuracy from the
    runtime overlay-cropping path. Real full-frame extraction is exercised in
    ``/evaluate/overlay`` instead, so this sampler sticks to curated board
    images with stable geometry.
    """
    if not CHESS_POSITIONS_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Chess positions test directory not found: {CHESS_POSITIONS_TEST_DIR}"
        )

    exclude_set = set(exclude or [])

    synthetic = _list_image_filenames(CHESS_POSITIONS_TEST_DIR, exclude=exclude_set)
    if not synthetic:
        synthetic = _list_image_filenames(BOARD_FIXTURES_DIR, exclude=exclude_set)

    chosen = random.sample(synthetic, min(limit, len(synthetic)))
    random.shuffle(chosen)
    return chosen


# ── Inspection ──────────────────────────────────────────────


def inspect_board(filename: str) -> dict:
    """Inspect a single board image: classify pieces and compare to ground truth."""
    is_real = filename.startswith(_REAL_PREFIX)

    if is_real:
        actual_name = filename[len(_REAL_PREFIX):]
        image_path = resolve_board_image_path(filename)
        source = "real"
        expected_fen = parse_real_board_fen(actual_name)
    else:
        image_path = resolve_board_image_path(filename)
        source = "synthetic"
        expected_fen = parse_fen_from_filename(filename)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read board image: {filename}")

    t_overlay = time.monotonic()
    runtime_overlay_check(image)
    overlay_detect_ms = round((time.monotonic() - t_overlay) * 1000, 1)

    if is_real:
        crop_result = _extract_board_crop_from_overlay_crop(image)
        if crop_result is None:
            elapsed_ms = overlay_detect_ms
            return {
                "filename": filename,
                "source": source,
                "expected_fen": expected_fen,
                "predicted_fen": None,
                "match": False,
                "piece_accuracy": 0.0,
                "square_diffs": ["board crop refinement failed"],
                "board_image_b64": _frame_to_base64(image),
                "elapsed_ms": elapsed_ms,
                "overlay_detect_ms": overlay_detect_ms,
                "grid_detect_ms": 0.0,
                "piece_classify_ms": 0.0,
                "error": "board crop refinement failed",
            }
        board_image = crop_result.board_crop
        grid_detect_ms = crop_result.grid_detect_ms
    else:
        if image.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"Invalid board image size: {filename}")
        board_image = image
        grid_detect_ms = 0.0

    t_classify = time.monotonic()
    try:
        if is_real:
            predicted_fen = _read_board_crop_fen(board_image)
        else:
            predicted_fen = read_fen_with_grid(
                board_image,
                _GRID,
                device="cpu",
                detect_orientation=False,
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
            "board_image_b64": _frame_to_base64(board_image),
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
        "board_image_b64": _frame_to_base64(board_image),
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
    """Build a piece-placement FEN from classifier class ids."""
    fen, _ = class_grid_to_fen(class_grid, detect_orientation=False)
    return fen


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


def _is_saved_frame(video_id: str, frame_name: str) -> bool:
    return f"{video_id}:{frame_name}" in _get_saved_frame_keys()


def get_extraction_candidates(
    limit: int = 200,
    video_ids: list[str] | None = None,
) -> list[str]:
    """Return video IDs that have overlay-eval candidate frames on disk.

    Approved videos without an explicit ``layout_type`` are treated as overlay,
    matching the rest of the screening pipeline.
    """
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
    seen: set[str] = set()
    for vid in raw:
        if len(result) >= limit:
            break
        if vid in seen:
            continue
        if _video_has_frames(vid):
            result.append(vid)
            seen.add(vid)

    if len(result) < limit and not video_ids:
        for vid in _fixture_candidate_video_ids(set(), limit - len(result)):
            if vid in seen:
                continue
            result.append(vid)
            seen.add(vid)
            if len(result) >= limit:
                break

    return result


def _iter_extractable_frame_names(video_id: str) -> list[str]:
    return _overlay_eval_frame_names(video_id)


def _load_board_crop_for_video_frame(
    video_id: str,
    frame_name: str,
) -> BoardCropResult | None:
    frame_path = _resolve_frame_path(video_id, frame_name)
    if frame_path is None:
        return None

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None

    runtime_crop = _extract_board_crop_from_frame(frame)
    if runtime_crop is not None:
        return runtime_crop

    fallback_bbox = _fixture_overlay_bbox(video_id, frame_name)
    if fallback_bbox is None:
        return None

    x, y, w, h = fallback_bbox
    fh, fw = frame.shape[:2]
    overlay_crop = frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    if overlay_crop.size == 0:
        return None

    fallback_crop = _extract_board_crop_from_overlay_crop(overlay_crop)
    board_crop = fallback_crop.board_crop if fallback_crop is not None else overlay_crop
    grid_detect_ms = fallback_crop.grid_detect_ms if fallback_crop is not None else 0.0
    return BoardCropResult(
        board_crop=board_crop,
        overlay_detect_ms=0.0,
        grid_detect_ms=grid_detect_ms,
        detector_found=False,
        warning="Runtime detector missed a known overlay; showing the fixture crop for review.",
    )


def extract_overlay_from_frames(video_id: str) -> dict:
    """Process one video's cached screening frames and return the best overlay."""
    if not _video_has_frames(video_id):
        return {"video_id": video_id, "status": "no_overlay"}

    for frame_name in _iter_extractable_frame_names(video_id):
        crop_result = _load_board_crop_for_video_frame(video_id, frame_name)
        if crop_result is None:
            continue

        t_classify = time.monotonic()
        try:
            fen = _read_board_crop_fen(crop_result.board_crop)
        except Exception:
            continue
        piece_classify_ms = round((time.monotonic() - t_classify) * 1000, 1)

        piece_count = _piece_count_from_fen(fen)
        if piece_count < 2:
            continue

        frame_key = f"{video_id}:{frame_name}"
        entry: dict = {
            "frame_key": frame_key,
            "video_id": video_id,
            "frame_name": frame_name,
            "image_b64": _frame_to_base64(crop_result.board_crop, max_width=None),
            "predicted_fen": fen,
            "overlay_detect_ms": crop_result.overlay_detect_ms,
            "grid_detect_ms": crop_result.grid_detect_ms,
            "piece_classify_ms": piece_classify_ms,
            "detector_found": crop_result.detector_found,
            "already_saved": _is_saved_frame(video_id, frame_name),
        }

        warnings: list[str] = []
        if crop_result.warning:
            warnings.append(crop_result.warning)
        if not _is_valid_fen_placement(fen):
            warnings.append("FEN may be invalid (missing king)")

        if warnings:
            entry["status"] = "warning"
            entry["warning"] = " ".join(warnings)
        else:
            entry["status"] = "ok"

        return entry

    return {"video_id": video_id, "status": "no_overlay"}


def detect_overlay_from_frames(video_id: str) -> dict:
    """Fast phase: detect overlay and return the exact board crop shown to the UI."""
    if not _video_has_frames(video_id):
        logger.info("detect_overlay: %s — no frames on disk", video_id)
        return {"video_id": video_id, "status": "no_overlay"}

    for frame_name in _iter_extractable_frame_names(video_id):
        crop_result = _load_board_crop_for_video_frame(video_id, frame_name)
        if crop_result is None:
            continue

        frame_key = f"{video_id}:{frame_name}"
        logger.info("detect_overlay: %s — found in %s", video_id, frame_name)
        result = {
            "frame_key": frame_key,
            "video_id": video_id,
            "frame_name": frame_name,
            "image_b64": _frame_to_base64(crop_result.board_crop, max_width=None),
            "overlay_detect_ms": crop_result.overlay_detect_ms,
            "grid_detect_ms": crop_result.grid_detect_ms,
            "detector_found": crop_result.detector_found,
            "already_saved": _is_saved_frame(video_id, frame_name),
            "status": "detected" if crop_result.detector_found else "warning",
        }
        if crop_result.warning:
            result["warning"] = crop_result.warning
        return result

    logger.info("detect_overlay: %s — no overlay in any frame", video_id)
    return {"video_id": video_id, "status": "no_overlay"}


def classify_overlay_fen(
    video_id: str,
    frame_name: str,
    image_b64: str | None = None,
) -> dict:
    """Slow phase: classify the exact board crop shown in the detect phase."""
    if image_b64:
        try:
            board_crop = _decode_image_b64(image_b64)
        except ValueError as e:
            return {"status": "error", "warning": str(e)}
    else:
        crop_result = _load_board_crop_for_video_frame(video_id, frame_name)
        if crop_result is None:
            return {"status": "error", "warning": "Overlay crop not available"}
        board_crop = crop_result.board_crop

    t_classify = time.monotonic()
    try:
        fen = _read_board_crop_fen(board_crop)
    except Exception as e:
        return {"status": "error", "warning": str(e)}

    piece_classify_ms = round((time.monotonic() - t_classify) * 1000, 1)
    piece_count = _piece_count_from_fen(fen)
    if piece_count < 2:
        return {
            "status": "error",
            "warning": "Too few pieces detected",
            "piece_classify_ms": piece_classify_ms,
        }

    result: dict = {
        "predicted_fen": fen,
        "piece_classify_ms": piece_classify_ms,
    }

    if not _is_valid_fen_placement(fen):
        result["status"] = "warning"
        result["warning"] = "FEN may be invalid (missing king)"
    else:
        result["status"] = "ok"

    return result


def save_confirmed_frame_extractions(confirmations: list[dict]) -> dict:
    """Save user-confirmed frame overlay extractions to val_real/.

    Supports two payload shapes:
    - screening-frame samples via ``video_id`` + ``frame_name``
    - custom board crops via ``sample_id`` + ``image_b64``
    """
    REAL_OVERLAY_TEST_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    errors: list[str] = []

    for conf in confirmations:
        sample_id_raw = str(conf.get("sample_id", "")).strip()
        sample_id = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in sample_id_raw)
        while "--" in sample_id:
            sample_id = sample_id.replace("--", "-")
        sample_id = sample_id.strip("-")

        video_id = conf.get("video_id", "")
        frame_name = conf.get("frame_name", "")
        fen = conf.get("fen", "")
        image_b64 = conf.get("image_b64")
        label = sample_id or f"{video_id}:{frame_name}"

        if image_b64:
            try:
                board_crop = _decode_image_b64(image_b64)
            except ValueError as e:
                errors.append(f"{label}: {e}")
                continue
        else:
            crop_result = _load_board_crop_for_video_frame(video_id, frame_name)
            if crop_result is None:
                errors.append(f"{label}: overlay crop not available")
                continue
            board_crop = crop_result.board_crop

        if board_crop.size == 0:
            errors.append(f"{label}: empty board crop")
            continue

        fen_hyphenated = fen.replace("/", "-")
        if sample_id:
            out_filename = f"r{sample_id}_{fen_hyphenated}.jpg"
        else:
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
    """Validate runtime_overlay_check accuracy on real video frames.

    Samples frames from downloaded overlay videos across diverse channels.
    For each frame, runs runtime_overlay_check and records whether the overlay
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
            det = runtime_overlay_check(frame)
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
    piece_classify_ms_avg: float | None = None,
    notes: str | None = None,
) -> dict:
    """Save overlay evaluation result to model_evaluations."""
    per_class_data: dict = {}
    if piece_accuracy is not None:
        per_class_data["piece_accuracy"] = piece_accuracy
    if images_per_minute is not None:
        per_class_data["images_per_minute"] = images_per_minute
    if piece_classify_ms_avg is not None:
        per_class_data["piece_classify_ms_avg"] = round(piece_classify_ms_avg, 3)

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


# ── Overlay Detection Evaluation Sessions ─────────────────


def save_overlay_detection_eval(
    detection_rate: float,
    fen_success_rate: float,
    sample_size: int,
    images_per_minute: int | None = None,
    notes: str | None = None,
) -> dict:
    """Save an overlay detection evaluation to model_evaluations."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_evaluations
                   (model_name, accuracy, sample_size, per_class, notes)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    "overlay_detection",
                    detection_rate,
                    sample_size,
                    json.dumps({
                        "fen_success_rate": fen_success_rate,
                        "images_per_minute": images_per_minute,
                    }),
                    notes,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return {"id": row[0]} if row else {}


def create_overlay_eval_session(
    results: list[dict],
    detection_rate: float | None,
    fen_success_rate: float | None,
    sample_size: int,
    pin_state: dict | None = None,
    evaluation_id: int | None = None,
) -> dict:
    """Create and persist an overlay detection evaluation session."""
    session_id = uuid.uuid4().hex[:12]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO overlay_eval_sessions
                   (id, sample_size, detection_rate,
                    fen_success_rate, results,
                    pin_state, evaluation_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id,
                    sample_size,
                    detection_rate,
                    fen_success_rate,
                    json.dumps(results),
                    json.dumps(pin_state or {}),
                    evaluation_id,
                ),
            )
            conn.commit()
    return {"session_id": session_id}


def _overlay_eval_metrics_from_results(results: list[dict]) -> tuple[int, float | None, float | None]:
    """Derive overlay-eval summary metrics from the stored per-sample results."""
    sample_size = len(results)
    if sample_size == 0:
        return 0, None, None

    detected = sum(1 for result in results if result.get("status") != "no_overlay")
    fen_success = sum(1 for result in results if result.get("status") == "ok")
    return sample_size, detected / sample_size, fen_success / sample_size


def get_overlay_eval_session(session_id: str) -> dict | None:
    """Fetch an overlay detection evaluation session by ID."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, detection_rate, fen_success_rate,
                          results, pin_state, evaluation_id
                   FROM overlay_eval_sessions WHERE id = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "created_at": str(row[1]),
                "sample_size": row[2],
                "detection_rate": row[3],
                "fen_success_rate": row[4],
                "results": row[5] if isinstance(row[5], list) else json.loads(row[5]),
                "pin_state": row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}"),
                "evaluation_id": row[7],
            }


def list_overlay_eval_sessions(limit: int = 20) -> list[dict]:
    """List recent overlay detection evaluation sessions (lightweight, no results)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, sample_size, detection_rate, fen_success_rate
                   FROM overlay_eval_sessions
                   ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            return [
                {
                    "id": row[0],
                    "created_at": str(row[1]),
                    "sample_size": row[2],
                    "detection_rate": row[3],
                    "fen_success_rate": row[4],
                }
                for row in cur.fetchall()
            ]


def update_overlay_eval_results(session_id: str, results: list[dict]) -> dict:
    """Replace an overlay detection session's results and recompute its metrics."""
    sample_size, detection_rate, fen_success_rate = _overlay_eval_metrics_from_results(results)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE overlay_eval_sessions
                   SET results = %s,
                       sample_size = %s,
                       detection_rate = %s,
                       fen_success_rate = %s
                   WHERE id = %s
                   RETURNING id""",
                (
                    json.dumps(results),
                    sample_size,
                    detection_rate,
                    fen_success_rate,
                    session_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Session {session_id} not found"}
            conn.commit()

    return {
        "ok": True,
        "sample_size": sample_size,
        "detection_rate": detection_rate,
        "fen_success_rate": fen_success_rate,
        "results": results,
    }


def update_overlay_eval_pins(session_id: str, pin_state: dict) -> dict:
    """Merge pin state updates into an overlay detection evaluation session."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pin_state FROM overlay_eval_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Session {session_id} not found"}

            existing = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
            existing.update(pin_state)

            cur.execute(
                "UPDATE overlay_eval_sessions SET pin_state = %s WHERE id = %s",
                (json.dumps(existing), session_id),
            )
            conn.commit()
    return {"ok": True, "pin_state": existing}
