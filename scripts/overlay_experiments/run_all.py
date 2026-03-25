#!/usr/bin/env python3
"""Run all 5 overlay reader strategies on reference videos and compare results."""

import os
import sys
import time
import traceback

import chess
import numpy as np

# Ensure we can import sibling modules
sys.path.insert(0, os.path.dirname(__file__))

# Strategy modules
import strategy_1_area_ranking as s1
import strategy_2_edge_empty as s2
import strategy_3_self_bootstrap as s3
import strategy_4_move_tracking as s4
import strategy_5_intra_clustering as s5
from shared import (
    GROUND_TRUTH,
    VIDEO_DIR,
    VIDEO_IDS,
    evaluate,
    extract_frame,
)

STRATEGIES = [
    ("S1: Area Rank", s1),
    ("S2: Edge+Bright", s2),
    ("S3: Self-Boot", s3),
    ("S4: Move Track", s4),
    ("S5: Clustering", s5),
]

TIMESTAMPS = [30, 60]  # seconds
VIDEO_PREFIXES = {
    "O8ZwstOxG_A": "O8Z",
    "7RaBQag34Hk": "7Ra",
    "2wWUKmCBr6A": "2wW",
    "Ov8PXnJp1PU": "Ov8",
}


def main():
    print("=" * 80)
    print("Overlay Reader Strategy Comparison")
    print("=" * 80)

    # Collect results
    all_results: dict[str, list[dict]] = {name: [] for name, _ in STRATEGIES}

    for vid_id in VIDEO_IDS:
        prefix = VIDEO_PREFIXES[vid_id]
        video_path = str(VIDEO_DIR / f"{vid_id}.mp4")
        if not os.path.exists(video_path):
            print(f"\n  SKIP {vid_id}: video file not found")
            continue

        print(f"\n{'─' * 60}")
        print(f"Video: {vid_id} ({prefix})")
        print(f"{'─' * 60}")

        # Reset stateful strategies for each video
        s3._reader = None
        s4._tracker = None

        for ts in TIMESTAMPS:
            gt_key = (prefix, ts)
            if gt_key not in GROUND_TRUTH:
                continue

            gt_fen = GROUND_TRUTH[gt_key]
            gt_board = chess.Board(gt_fen)

            frame = extract_frame(video_path, ts)
            if frame is None:
                print(f"  t={ts}s: could not extract frame")
                continue

            print(f"\n  t={ts}s  GT: {gt_board.board_fen()}")

            for name, mod in STRATEGIES:
                t0 = time.time()
                try:
                    # Strategies 3 & 4 need video_path for bootstrapping
                    if hasattr(mod, "read_position"):
                        sig = mod.read_position.__code__.co_varnames
                        if "video_path" in sig:
                            pred_fen = mod.read_position(frame, video_path=video_path)
                        else:
                            pred_fen = mod.read_position(frame)
                    else:
                        pred_fen = None
                except Exception as e:
                    pred_fen = None
                    print(f"    {name}: ERROR — {e}")
                    traceback.print_exc()
                    continue

                elapsed = time.time() - t0
                metrics = evaluate(pred_fen, gt_fen)
                all_results[name].append(metrics)

                status = "✓" if metrics["exact_match"] else " "
                print(
                    f"    {name:20s} [{elapsed:5.2f}s] "
                    f"empty={metrics['empty_acc']:.0%} "
                    f"presence={metrics['presence_acc']:.0%} "
                    f"color={metrics['color_acc']:.0%} "
                    f"type={metrics['type_acc']:.0%} "
                    f"{status}"
                )
                if pred_fen:
                    print(f"      Pred: {pred_fen}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY (averaged across all test frames)")
    print(f"{'=' * 80}")
    print(
        f"{'Strategy':20s} {'Empty':>8s} {'Presence':>10s} {'Color':>8s} {'Type':>8s} {'Exact':>8s} {'N':>4s}"
    )
    print("─" * 70)

    for name, _ in STRATEGIES:
        results = all_results[name]
        if not results:
            print(f"{name:20s} {'—':>8s} {'—':>10s} {'—':>8s} {'—':>8s} {'—':>8s} {0:4d}")
            continue

        avg = {
            k: np.mean([r[k] for r in results])
            for k in ["empty_acc", "presence_acc", "color_acc", "type_acc"]
        }
        exact = sum(1 for r in results if r["exact_match"])
        n = len(results)

        print(
            f"{name:20s} "
            f"{avg['empty_acc']:7.0%} "
            f"{avg['presence_acc']:9.0%} "
            f"{avg['color_acc']:7.0%} "
            f"{avg['type_acc']:7.0%} "
            f"{exact}/{n:>3d} "
            f"{n:4d}"
        )

    print()


if __name__ == "__main__":
    main()
