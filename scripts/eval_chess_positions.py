#!/usr/bin/env python3
"""Evaluate the piece classifier on chess-positions boards.

Classifies boards using the known 50×50 uniform grid, reports accuracy,
and optionally copies failing boards into the test fixtures directory so
they become regression tests.

Usage:
    .venv/bin/python scripts/eval_chess_positions.py data/overlay/val
    .venv/bin/python scripts/eval_chess_positions.py data/overlay/val --add-failures
    .venv/bin/python scripts/eval_chess_positions.py data/overlay/val --add-fixtures 20
    .venv/bin/python scripts/eval_chess_positions.py data/overlay/val --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import chess
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.chess_positions_data import (
    BOARD_SIZE,
    SQ_SIZE,
    parse_fen_from_filename,
)
from pipeline.overlay.grid_detector import GridResult
from pipeline.overlay.piece_classifier import read_fen_with_grid

from argus.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "boards"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

# Uniform grid for chess-positions boards (400×400, 50px squares)
_GRID = GridResult(
    v_lines=list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
    h_lines=list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
    sq_size=SQ_SIZE,
)


def _fen_to_board(fen: str) -> chess.Board:
    if " " not in fen:
        fen = fen + " w - - 0 1"
    return chess.Board(fen)


def _diff_boards(predicted: chess.Board, expected: chess.Board) -> list[str]:
    """Return list of mismatch descriptions."""
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


def evaluate(
    data_dir: Path,
    *,
    limit: int | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Evaluate classifier on chess-positions boards.

    Returns (passes, failures) where each entry is a dict with
    'filename', 'fen', 'predicted_fen', and 'mismatches'.
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith((".jpeg", ".jpg", ".png")))
    if limit and limit < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, limit)

    passes: list[dict] = []
    failures: list[dict] = []

    for i, fname in enumerate(files):
        image = cv2.imread(str(data_dir / fname), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
            continue

        expected_fen = parse_fen_from_filename(fname)
        try:
            predicted_fen = read_fen_with_grid(
                image,
                _GRID,
                device=device,
                detect_orientation=False,
            )
        except Exception as e:
            logger.warning("Error on %s: %s", fname, e)
            failures.append(
                {
                    "filename": fname,
                    "fen": expected_fen,
                    "predicted_fen": None,
                    "mismatches": [str(e)],
                }
            )
            continue

        expected_board = _fen_to_board(expected_fen)
        predicted_board = _fen_to_board(predicted_fen)
        mismatches = _diff_boards(predicted_board, expected_board)

        entry = {
            "filename": fname,
            "fen": expected_fen,
            "predicted_fen": predicted_fen,
            "mismatches": mismatches,
        }

        if mismatches:
            failures.append(entry)
        else:
            passes.append(entry)

        if (i + 1) % 200 == 0:
            logger.info(
                "Progress: %d/%d (%.1f%% accurate so far)",
                i + 1,
                len(files),
                len(passes) / max(len(passes) + len(failures), 1) * 100,
            )

    return passes, failures


def add_to_fixtures(entries: list[dict], data_dir: Path) -> int:
    """Copy board images to test fixtures and update ground_truth.json.

    Returns number of boards added.
    """
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing ground truth
    gt: dict = {}
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH) as f:
            gt = json.load(f)

    added = 0
    for entry in entries:
        fname = entry["filename"]
        # Use filename stem as key (truncated for readability)
        key = Path(fname).stem[:40]
        if key in gt:
            continue

        # Copy image
        src = data_dir / fname
        dst = FIXTURES_DIR / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        gt[key] = {
            "image": fname,
            "fen": entry["fen"],
            "grid": {
                "v_lines": list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
                "h_lines": list(range(0, BOARD_SIZE + 1, SQ_SIZE)),
            },
        }
        added += 1

    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(gt, f, indent=2)

    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate piece classifier on chess-positions")
    parser.add_argument("data_dir", type=str, help="Path to chess-positions test/ directory")
    parser.add_argument("--limit", type=int, default=None, help="Max boards to evaluate")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--add-failures",
        action="store_true",
        help="Copy failing boards to test fixtures",
    )
    parser.add_argument(
        "--add-fixtures",
        type=int,
        default=0,
        metavar="N",
        help="Also add N random passing boards as regression fixtures",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Directory not found: %s", data_dir)
        sys.exit(1)

    passes, failures = evaluate(
        data_dir,
        limit=args.limit,
        device=resolve_device(args.device),
        seed=args.seed,
    )

    total = len(passes) + len(failures)
    acc = len(passes) / max(total, 1) * 100
    logger.info("")
    logger.info("=== Results ===")
    logger.info(
        "Total: %d boards, %d passed, %d failed (%.1f%% accuracy)",
        total,
        len(passes),
        len(failures),
        acc,
    )

    if failures:
        logger.info("")
        logger.info("Failures (first 20):")
        for entry in failures[:20]:
            n_wrong = len(entry["mismatches"])
            logger.info("  %s: %d wrong squares", entry["filename"], n_wrong)
            for m in entry["mismatches"][:5]:
                logger.info("    %s", m)

    # Add failing boards to test fixtures
    if args.add_failures and failures:
        added = add_to_fixtures(failures, data_dir)
        logger.info("")
        logger.info("Added %d failing boards to %s", added, FIXTURES_DIR)

    # Add passing boards as regression fixtures
    if args.add_fixtures > 0 and passes:
        rng = random.Random(args.seed)
        sample = rng.sample(passes, min(args.add_fixtures, len(passes)))
        added = add_to_fixtures(sample, data_dir)
        logger.info("Added %d passing boards as regression fixtures to %s", added, FIXTURES_DIR)

    if args.add_failures or args.add_fixtures > 0:
        logger.info("Run `make test` to verify all fixtures pass.")


if __name__ == "__main__":
    main()
