"""Render overlay test fixtures with predicted and ground-truth boxes.

This makes the bbox tests inspectable: every frame is saved with
ground-truth and detector outputs overlaid, plus a contact sheet and a JSON
summary with IoU/pass-fail details.

Examples:
    .venv/bin/python scripts/visualize_overlay_tests.py
    .venv/bin/python scripts/visualize_overlay_tests.py \\
        --ground-truth data/videos/ground_truth.json \\
        --images-root data/videos --tier lores
    .venv/bin/python scripts/visualize_overlay_tests.py \\
        --glob 'data/videos/*/lores/50pct.jpg'
    .venv/bin/python scripts/visualize_overlay_tests.py \\
        --detector precise --limit 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.overlay.scanner import detect_overlay_fast, fast_overlay_check  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "frames"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "overlay_test_viz"
FAST_IOU_THRESHOLD = 0.40
PRECISE_UNDERCOVERAGE_TOLERANCE_PX = 8

COLOR_GT = (0, 220, 0)
COLOR_FAST = (0, 215, 255)
COLOR_PRECISE = (255, 200, 0)
COLOR_PASS = (40, 190, 40)
COLOR_FAIL = (40, 40, 220)
COLOR_TEXT = (240, 240, 240)
COLOR_MUTED = (180, 180, 180)
COLOR_BG = (24, 24, 24)


@dataclass
class FrameCase:
    key: str
    path: Path
    has_overlay: bool | None
    gt_bbox: tuple[int, int, int, int] | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=GROUND_TRUTH_PATH,
        help="Ground-truth JSON to visualize. Ignored when --glob is used.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=FIXTURES_DIR,
        help="Root directory used to resolve image paths from the ground-truth file.",
    )
    parser.add_argument(
        "--tier",
        choices=["auto", "lores", "hires", "fullres"],
        default="auto",
        help="When GT image paths omit the frame tier, resolve via this tier or auto-detect.",
    )
    parser.add_argument(
        "--glob",
        help="Optional glob of images to visualize instead of test fixtures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of frames to render. 0 means no limit.",
    )
    parser.add_argument(
        "--detector",
        choices=["fast", "precise", "both"],
        default="both",
        help="Which detector outputs to draw.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=640,
        help="Width of each panel in the contact sheet.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of contact-sheet columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for annotated images and summary outputs.",
    )
    return parser.parse_args()


def _compute_iou(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> float | None:
    if a is None or b is None:
        return None

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox_undercoverage(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if a is None or b is None:
        return None

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_right = ax + aw
    a_bottom = ay + ah
    b_right = bx + bw
    b_bottom = by + bh
    return (
        max(0, ax - bx),
        max(0, ay - by),
        max(0, b_right - a_right),
        max(0, b_bottom - a_bottom),
    )


def _image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    return w, h


def _resolve_gt_image(
    images_root: Path,
    image_rel: Path,
    frame_width: int | None,
    frame_height: int | None,
    tier: str,
) -> Path:
    direct = images_root / image_rel
    if direct.exists():
        return direct

    video_id = image_rel.parts[0]
    filename = image_rel.name

    tier_order = ["hires", "fullres", "lores"] if tier == "auto" else [tier]
    candidates: list[Path] = []

    for tier_name in tier_order:
        candidate = images_root / video_id / tier_name / filename
        if candidate.exists():
            candidates.append(candidate)

    fallback = images_root / video_id / filename
    if fallback.exists():
        candidates.append(fallback)

    if not candidates:
        return direct

    if frame_width is not None and frame_height is not None:
        for candidate in candidates:
            size = _image_size(candidate)
            if size == (frame_width, frame_height):
                return candidate

    return candidates[0]


def _load_cases(
    glob_pattern: str | None,
    limit: int,
    ground_truth_path: Path,
    images_root: Path,
    tier: str,
) -> list[FrameCase]:
    cases: list[FrameCase] = []

    if glob_pattern:
        for path in sorted(PROJECT_ROOT.glob(glob_pattern)):
            if path.is_file():
                cases.append(
                    FrameCase(
                        key=str(path.relative_to(PROJECT_ROOT)),
                        path=path,
                        has_overlay=None,
                        gt_bbox=None,
                    )
                )
    else:
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Missing ground truth file: {ground_truth_path}")
        gt = json.loads(ground_truth_path.read_text())
        for key, entry in sorted(gt.items()):
            img_path = _resolve_gt_image(
                images_root=images_root,
                image_rel=Path(entry["image"]),
                frame_width=entry.get("frame_width"),
                frame_height=entry.get("frame_height"),
                tier=tier,
            )
            bbox = entry.get("bbox")
            cases.append(
                FrameCase(
                    key=key,
                    path=img_path,
                    has_overlay=bool(entry.get("has_overlay")),
                    gt_bbox=tuple(bbox) if bbox else None,
                )
            )

    if limit > 0:
        cases = cases[:limit]
    return cases


def _draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y, w, h = bbox
    thickness = max(2, min(image.shape[0], image.shape[1]) // 320)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(image.shape[0], image.shape[1]) / 1400)
    label_thickness = max(1, thickness - 1)
    (tw, th), _ = cv2.getTextSize(label, font, scale, label_thickness)
    box_y1 = max(0, y - th - 12)
    box_y2 = box_y1 + th + 10
    box_x2 = min(image.shape[1] - 1, x + tw + 10)
    cv2.rectangle(image, (x, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        image,
        label,
        (x + 5, box_y2 - 6),
        font,
        scale,
        (0, 0, 0),
        label_thickness,
        lineType=cv2.LINE_AA,
    )


def _case_verdict(
    expected_overlay: bool | None,
    gt_bbox: tuple[int, int, int, int] | None,
    det_bbox: tuple[int, int, int, int] | None,
    detector_name: str,
) -> str:
    if expected_overlay is None:
        return "n/a"
    if not expected_overlay:
        return "pass" if det_bbox is None else "fail"
    if det_bbox is None or gt_bbox is None:
        return "fail"
    if detector_name == "fast":
        iou = _compute_iou(det_bbox, gt_bbox) or 0.0
        return "pass" if iou >= FAST_IOU_THRESHOLD else "fail"
    under = _bbox_undercoverage(det_bbox, gt_bbox)
    assert under is not None
    return "pass" if max(under) <= PRECISE_UNDERCOVERAGE_TOLERANCE_PX else "fail"


def _info_strip(
    width: int,
    lines: list[tuple[str, tuple[int, int, int]]],
    footer_height: int = 122,
) -> np.ndarray:
    strip = np.full((footer_height, width, 3), COLOR_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(strip, lines[0][0], (14, 28), font, 0.72, lines[0][1], 2, cv2.LINE_AA)
    y = 56
    for text, color in lines[1:]:
        cv2.putText(strip, text, (14, y), font, 0.52, color, 1, cv2.LINE_AA)
        y += 22
    return strip


def _render_panel(
    case: FrameCase,
    image: np.ndarray,
    draw_fast: bool,
    draw_precise: bool,
) -> tuple[np.ndarray, dict]:
    fast = fast_overlay_check(image)
    precise = detect_overlay_fast(image)
    annotated = image.copy()

    if case.gt_bbox is not None:
        _draw_bbox(annotated, case.gt_bbox, COLOR_GT, "GT")
    if draw_fast and fast.found and fast.bbox is not None:
        _draw_bbox(annotated, fast.bbox, COLOR_FAST, "FAST")
    if draw_precise and precise.found and precise.bbox is not None:
        _draw_bbox(annotated, precise.bbox, COLOR_PRECISE, "PRECISE")

    target_w = max(240, int(args.panel_width))
    scale = target_w / annotated.shape[1]
    resized = cv2.resize(
        annotated,
        (target_w, int(round(annotated.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )

    fast_iou = _compute_iou(fast.bbox if fast.found else None, case.gt_bbox)
    precise_iou = _compute_iou(precise.bbox if precise.found else None, case.gt_bbox)
    precise_under = _bbox_undercoverage(precise.bbox if precise.found else None, case.gt_bbox)

    gt_text = (
        f"GT: overlay {list(case.gt_bbox)}"
        if case.gt_bbox is not None
        else ("GT: no overlay" if case.has_overlay is False else "GT: none")
    )
    fast_verdict = _case_verdict(
        case.has_overlay, case.gt_bbox, fast.bbox if fast.found else None, "fast"
    )
    precise_verdict = _case_verdict(
        case.has_overlay, case.gt_bbox, precise.bbox if precise.found else None, "precise"
    )

    lines = [
        (case.key, COLOR_TEXT),
        (gt_text, COLOR_MUTED),
        (
            f"fast: {'found' if fast.found else 'miss'}"
            + (f"  IoU={fast_iou:.3f}" if fast_iou is not None else "")
            + f"  score={fast.score:.3f}"
            + f"  {fast_verdict.upper()}",
            COLOR_PASS
            if fast_verdict == "pass"
            else (COLOR_FAIL if fast_verdict == "fail" else COLOR_MUTED),
        ),
        (
            f"precise: {'found' if precise.found else 'miss'}"
            + (f"  IoU={precise_iou:.3f}" if precise_iou is not None else "")
            + (f"  clip={precise_under}" if precise_under is not None else "")
            + f"  score={precise.score:.3f}"
            + f"  {precise_verdict.upper()}",
            COLOR_PASS
            if precise_verdict == "pass"
            else (COLOR_FAIL if precise_verdict == "fail" else COLOR_MUTED),
        ),
    ]
    panel = np.vstack([resized, _info_strip(target_w, lines)])

    summary = {
        "key": case.key,
        "image_path": str(case.path),
        "has_overlay": case.has_overlay,
        "ground_truth_bbox": list(case.gt_bbox) if case.gt_bbox is not None else None,
        "fast": {
            "found": bool(fast.found),
            "bbox": list(fast.bbox) if fast.bbox is not None else None,
            "score": round(float(fast.score), 4),
            "iou": round(float(fast_iou), 4) if fast_iou is not None else None,
            "verdict": fast_verdict,
        },
        "precise": {
            "found": bool(precise.found),
            "bbox": list(precise.bbox) if precise.bbox is not None else None,
            "score": round(float(precise.score), 4),
            "iou": round(float(precise_iou), 4) if precise_iou is not None else None,
            "undercoverage_px": list(precise_under) if precise_under is not None else None,
            "verdict": precise_verdict,
        },
    }
    return panel, summary


def _build_contact_sheet(panels: list[np.ndarray], columns: int) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to render")

    columns = max(1, columns)
    rows = math.ceil(len(panels) / columns)
    panel_h = max(panel.shape[0] for panel in panels)
    panel_w = max(panel.shape[1] for panel in panels)
    gap = 16

    sheet_h = rows * panel_h + gap * (rows + 1)
    sheet_w = columns * panel_w + gap * (columns + 1)
    sheet = np.full((sheet_h, sheet_w, 3), (8, 8, 8), dtype=np.uint8)

    for idx, panel in enumerate(panels):
        row = idx // columns
        col = idx % columns
        y = gap + row * (panel_h + gap)
        x = gap + col * (panel_w + gap)
        sheet[y : y + panel.shape[0], x : x + panel.shape[1]] = panel

    return sheet


def main() -> None:
    cases = _load_cases(
        args.glob,
        args.limit,
        args.ground_truth,
        args.images_root,
        args.tier,
    )
    if not cases:
        raise SystemExit("No images matched.")

    out_dir = args.out_dir
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    draw_fast = args.detector in {"fast", "both"}
    draw_precise = args.detector in {"precise", "both"}

    panels: list[np.ndarray] = []
    summary_rows: list[dict] = []

    for case in cases:
        image = cv2.imread(str(case.path))
        if image is None:
            print(f"Skipping unreadable image: {case.path}")
            continue

        panel, summary = _render_panel(case, image, draw_fast, draw_precise)
        panels.append(panel)
        summary_rows.append(summary)

        safe_name = case.key.replace("/", "__").replace(":", "__")
        panel_path = panels_dir / f"{safe_name}.jpg"
        cv2.imwrite(str(panel_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

    if not panels:
        raise SystemExit("No panels were rendered.")

    sheet = _build_contact_sheet(panels, args.columns)
    sheet_path = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    summary = {
        "count": len(summary_rows),
        "detector": args.detector,
        "fast_iou_threshold": FAST_IOU_THRESHOLD,
        "rows": summary_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    fast_failures = sum(1 for row in summary_rows if row["fast"]["verdict"] == "fail")
    precise_failures = sum(1 for row in summary_rows if row["precise"]["verdict"] == "fail")

    print(f"Wrote {len(summary_rows)} annotated panels to {panels_dir}")
    print(f"Contact sheet: {sheet_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Fast failures: {fast_failures}")
    print(f"Precise failures: {precise_failures}")


if __name__ == "__main__":
    args = _parse_args()
    main()
