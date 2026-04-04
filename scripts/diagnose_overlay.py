"""Diagnostic script for overlay detection alignment issues.

Runs fast_overlay_check internals with verbose logging on both test fixtures
and real frames to compare scores and identify why detection fails.

Usage:
    docker exec argus-dev-api python3 scripts/diagnose_overlay.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pipeline.overlay.scanner import (
    FAST_CHECK_MAX_DIM,
    FAST_SCAN_SCALES,
    FAST_SCAN_STEP_FRACTION,
    MAX_CHECKERBOARD_STD,
    MIN_ALTERNATION_CONTRAST,
    MIN_ALTERNATION_FRAC,
    _checkerboard_std,
    check_checkerboard_consistency,
    compute_alternation_strength,
    compute_grid_regularity,
    fast_overlay_check,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Test fixtures
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "frames"
FIXTURES_GT = FIXTURES_DIR / "ground_truth.json"

# Real frames - prefer hires from new layout, fall back to legacy overlay dir
REAL_FRAMES_DIR = PROJECT_ROOT / "data" / "videos"
_LEGACY_OVERLAY_DIR = PROJECT_ROOT / "data" / "overlay" / "dataset" / "frames"


def scan_with_logging(frame: np.ndarray) -> dict:
    """Replicate fast_overlay_check logic but log all scores."""
    orig_h, orig_w = frame.shape[:2]

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gray_full = cv2.GaussianBlur(gray_full, (3, 3), 0)

    scale_factor = 1.0
    if max(orig_h, orig_w) > FAST_CHECK_MAX_DIM:
        scale_factor = FAST_CHECK_MAX_DIM / max(orig_h, orig_w)
        small = cv2.resize(
            frame,
            (int(orig_w * scale_factor), int(orig_h * scale_factor)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame
    h, w = small.shape[:2] if len(small.shape) == 3 else small.shape

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    best_frac = 0.0
    best_contrast = 0.0
    best_regularity = 0.0
    best_scale = 0.0
    best_window = (0, 0, 0, 0)
    total_windows = 0

    # Also try sub-cell offsets to see what scores WOULD be
    best_frac_with_offsets = 0.0
    best_contrast_with_offsets = 0.0

    for scale in FAST_SCAN_SCALES:
        win_size = int(min(h, w) * scale)
        if win_size < 64:
            continue

        step = max(1, int(win_size * FAST_SCAN_STEP_FRACTION))
        cell = win_size // 8
        offsets = [0, cell // 3, 2 * cell // 3]

        for y in range(0, h - win_size + 1, step):
            for x in range(0, w - win_size + 1, step):
                total_windows += 1
                region = gray[y : y + win_size, x : x + win_size]

                frac, contrast = compute_alternation_strength(region)
                if frac > best_frac or (frac == best_frac and contrast > best_contrast):
                    best_frac = frac
                    best_contrast = contrast
                    best_scale = scale
                    best_window = (x, y, win_size, win_size)

                regularity = compute_grid_regularity(region)
                if regularity > best_regularity:
                    best_regularity = regularity

                # Try sub-cell offsets
                for dy in offsets:
                    for dx in offsets:
                        if dx == 0 and dy == 0:
                            continue
                        rx, ry = x + dx, y + dy
                        if rx + win_size > w or ry + win_size > h:
                            continue
                        region2 = gray[ry : ry + win_size, rx : rx + win_size]
                        f2, c2 = compute_alternation_strength(region2)
                        if f2 > best_frac_with_offsets or (
                            f2 == best_frac_with_offsets and c2 > best_contrast_with_offsets
                        ):
                            best_frac_with_offsets = f2
                            best_contrast_with_offsets = c2

    # Take the overall best including offsets
    best_frac_with_offsets = max(best_frac, best_frac_with_offsets)
    best_contrast_with_offsets = max(best_contrast, best_contrast_with_offsets)

    # Check checkerboard std at best window mapped to full res
    inv = 1.0 / scale_factor if scale_factor < 1.0 else 1.0
    bx, by, bw, bh = best_window
    if scale_factor < 1.0:
        full_bbox = (int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv))
    else:
        full_bbox = best_window
    fx, fy, fw, fh_box = full_bbox
    fx = max(0, min(fx, orig_w - fw))
    fy = max(0, min(fy, orig_h - fh_box))

    if fw > 0 and fh_box > 0 and fx + fw <= orig_w and fy + fh_box <= orig_h:
        cb_std = _checkerboard_std(gray_full[fy : fy + fh_box, fx : fx + fw])
        cb_pass = check_checkerboard_consistency(gray_full, (fx, fy, fw, fh_box))
    else:
        cb_std = 999.0
        cb_pass = False

    # Run actual detection
    det = fast_overlay_check(frame)

    return {
        "detected": det.found,
        "det_score": det.score,
        "det_bbox": list(det.bbox) if det.bbox else None,
        "best_frac": round(best_frac, 4),
        "best_contrast": round(best_contrast, 2),
        "best_frac_offsets": round(best_frac_with_offsets, 4),
        "best_contrast_offsets": round(best_contrast_with_offsets, 2),
        "best_regularity": round(best_regularity, 4),
        "best_scale": best_scale,
        "checkerboard_std": round(cb_std, 2),
        "checkerboard_pass": cb_pass,
        "total_windows": total_windows,
        "passes_p1_frac": best_frac >= MIN_ALTERNATION_FRAC,
        "passes_p1_contrast": best_contrast >= MIN_ALTERNATION_CONTRAST,
        "passes_p1_with_offsets": (
            best_frac_with_offsets >= MIN_ALTERNATION_FRAC
            and best_contrast_with_offsets >= MIN_ALTERNATION_CONTRAST
        ),
    }


def main():
    print("=" * 90)
    print("OVERLAY DETECTION DIAGNOSTIC")
    print(
        f"Thresholds: frac>={MIN_ALTERNATION_FRAC}, contrast>={MIN_ALTERNATION_CONTRAST}, "
        f"cb_std<={MAX_CHECKERBOARD_STD}"
    )
    print("=" * 90)

    # Load test fixtures
    fixtures = []
    if FIXTURES_GT.exists():
        gt = json.loads(FIXTURES_GT.read_text())
        for key, entry in gt.items():
            if not entry.get("has_overlay"):
                continue
            video_id, label = key.split("/", 1)
            path = FIXTURES_DIR / video_id / f"{label}.jpg"
            if path.exists():
                fixtures.append((f"FIXTURE:{key}", str(path)))

    # Load real frames (sample 20 videos, 1 frame each)
    # Check hires tier in new layout, fall back to legacy overlay dir
    real_frames = []
    search_dirs = []
    if REAL_FRAMES_DIR.exists():
        for vd in sorted(REAL_FRAMES_DIR.iterdir()):
            hires = vd / "hires"
            if hires.is_dir():
                search_dirs.append((vd.name, hires))
    if not search_dirs and _LEGACY_OVERLAY_DIR.exists():
        for vd in sorted(_LEGACY_OVERLAY_DIR.iterdir()):
            if vd.is_dir():
                search_dirs.append((vd.name, vd))
    count = 0
    for vid_name, fdir in search_dirs:
        fp = fdir / "50pct.jpg"
        if not fp.exists():
            fp = fdir / "25pct.jpg"
        if fp.exists():
            real_frames.append((f"REAL:{vid_name}/50pct", str(fp)))
            count += 1
            if count >= 20:
                break

    all_frames = fixtures + real_frames

    if not all_frames:
        print("No frames found!")
        sys.exit(1)

    print(f"\nAnalyzing {len(fixtures)} test fixtures + {len(real_frames)} real frames\n")

    header = (
        f"{'Frame':<45} {'Det':>3} {'Frac':>6} {'Cntr':>6} "
        f"{'F+Off':>6} {'C+Off':>6} {'Reg':>6} {'CB_Std':>7} {'CB':>3} "
        f"{'Scale':>5} {'P1F':>3} {'P1C':>3} {'P1+O':>4}"
    )
    print(header)
    print("-" * len(header))

    fixture_detected = 0
    real_detected = 0
    real_would_pass_with_offsets = 0

    for label, path in all_frames:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{label:<45} COULD NOT LOAD")
            continue

        result = scan_with_logging(frame)

        det_str = "YES" if result["detected"] else " no"
        p1f = "YES" if result["passes_p1_frac"] else " no"
        p1c = "YES" if result["passes_p1_contrast"] else " no"
        p1o = "YES" if result["passes_p1_with_offsets"] else "  no"

        print(
            f"{label:<45} {det_str:>3} {result['best_frac']:>6.3f} {result['best_contrast']:>6.1f} "
            f"{result['best_frac_offsets']:>6.3f} {result['best_contrast_offsets']:>6.1f} "
            f"{result['best_regularity']:>6.3f} {result['checkerboard_std']:>7.1f} "
            f"{'Y' if result['checkerboard_pass'] else 'N':>3} "
            f"{result['best_scale']:>5.2f} {p1f:>3} {p1c:>3} {p1o:>4}"
        )

        if label.startswith("FIXTURE:"):
            if result["detected"]:
                fixture_detected += 1
        else:
            if result["detected"]:
                real_detected += 1
            if result["passes_p1_with_offsets"]:
                real_would_pass_with_offsets += 1

    print("\n" + "=" * 90)
    print("SUMMARY")
    print(f"  Test fixtures:  {fixture_detected}/{len(fixtures)} detected")
    print(f"  Real frames:    {real_detected}/{len(real_frames)} detected")
    print(f"  Real w/offsets: {real_would_pass_with_offsets}/{len(real_frames)} would pass Phase 1")
    print("=" * 90)


if __name__ == "__main__":
    main()
