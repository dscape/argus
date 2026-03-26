"""Service layer for overlay (piece classifier) accuracy testing.

Uses the chess_positions dataset (400×400 board images with FEN in filenames)
to evaluate the DINOv2 piece classifier. Mirrors the screening inspector
pattern: sample → inspect → save session.
"""

import base64
import logging
import os
import random
import time
import uuid
from pathlib import Path

import chess
import cv2
import json
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.overlay.chess_positions_data import (
    BOARD_SIZE,
    SQ_SIZE,
    parse_fen_from_filename,
)
from pipeline.overlay.grid_detector import GridResult
from pipeline.overlay.piece_classifier import read_fen_with_grid

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHESS_POSITIONS_TEST_DIR = _PROJECT_ROOT / "data" / "chess_positions" / "test"

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
    """Return random sample of board image filenames from test set."""
    if not CHESS_POSITIONS_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Chess positions test directory not found: {CHESS_POSITIONS_TEST_DIR}"
        )

    files = [
        f
        for f in os.listdir(CHESS_POSITIONS_TEST_DIR)
        if f.endswith((".jpeg", ".jpg", ".png"))
    ]

    if exclude:
        exclude_set = set(exclude)
        files = [f for f in files if f not in exclude_set]

    if limit < len(files):
        files = random.sample(files, limit)

    return files


# ── Inspection ──────────────────────────────────────────────


def inspect_board(filename: str) -> dict:
    """Inspect a single board image: classify pieces and compare to ground truth."""
    image_path = CHESS_POSITIONS_TEST_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(f"Board image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Invalid board image: {filename}")

    expected_fen = parse_fen_from_filename(filename)

    start = time.monotonic()
    try:
        predicted_fen = read_fen_with_grid(
            image, _GRID, device="cpu", detect_orientation=False
        )
    except Exception as e:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return {
            "filename": filename,
            "expected_fen": expected_fen,
            "predicted_fen": None,
            "match": False,
            "piece_accuracy": 0.0,
            "square_diffs": [str(e)],
            "board_image_b64": _frame_to_base64(image),
            "elapsed_ms": elapsed_ms,
            "error": str(e),
        }

    elapsed_ms = round((time.monotonic() - start) * 1000, 1)

    expected_board = _fen_to_board(expected_fen)
    predicted_board = _fen_to_board(predicted_fen)
    mismatches = _diff_boards(predicted_board, expected_board)

    correct_squares = 64 - len(mismatches)
    piece_accuracy = correct_squares / 64

    return {
        "filename": filename,
        "expected_fen": expected_fen,
        "predicted_fen": predicted_fen,
        "match": len(mismatches) == 0,
        "piece_accuracy": round(piece_accuracy, 4),
        "square_diffs": mismatches,
        "board_image_b64": _frame_to_base64(image),
        "elapsed_ms": elapsed_ms,
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
                    "piece_classifier",
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
