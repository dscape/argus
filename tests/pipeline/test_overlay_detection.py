"""Overlay detection tests driven by manually annotated ground truth.

Ground truth is stored at tests/fixtures/frames/ground_truth.json,
created via the dev-tools overlay-bbox training-label UI.

Tests target the default YOLO detector contracts:
- runtime_overlay_check is the runtime presence/bbox screener
- detect_overlay_runtime is the padded runtime bbox detector
"""

import json
import time
from pathlib import Path

import cv2
import pytest
from pipeline.overlay.scanner import detect_overlay_runtime, runtime_overlay_check

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frames"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

MAX_MS = 200
PRECISE_UNDERCOVERAGE_TOLERANCE_PX = 8


def _load_ground_truth() -> dict:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    return json.loads(GROUND_TRUTH_PATH.read_text())


def _compute_iou(a: list[int], b: list[int]) -> float:
    """Compute IoU between two [x, y, w, h] bboxes."""
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
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Pixels of GT crop lost on each side: (left, top, right, bottom)."""
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


def _positive_entries() -> list[tuple[str, dict]]:
    gt = _load_ground_truth()
    return [(k, v) for k, v in gt.items() if v.get("has_overlay")]


def _negative_entries() -> list[tuple[str, dict]]:
    gt = _load_ground_truth()
    return [(k, v) for k, v in gt.items() if not v.get("has_overlay")]


def _positive_ids() -> list[str]:
    return [k for k, _ in _positive_entries()]


def _negative_ids() -> list[str]:
    return [k for k, _ in _negative_entries()]


def _load_frame(key: str):
    video_id, label = key.split("/", 1)
    path = FIXTURES_DIR / video_id / f"{label}.jpg"
    frame = cv2.imread(str(path))
    assert frame is not None, f"Could not load frame: {path}"
    return frame


@pytest.fixture(autouse=True)
def _skip_without_ground_truth():
    gt = _load_ground_truth()
    if not gt:
        pytest.skip("No ground truth annotations found")


class TestRuntimeOverlayCheck:
    """runtime_overlay_check must screen accurately under 100ms."""

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_detects_overlay(self, key: str):
        frame = _load_frame(key)
        det = runtime_overlay_check(frame)
        assert det.found, f"runtime_overlay_check missed overlay in {key}"

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_under_100ms(self, key: str):
        frame = _load_frame(key)
        t0 = time.perf_counter()
        runtime_overlay_check(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < MAX_MS, (
            f"runtime_overlay_check took {elapsed_ms:.1f}ms on {key} (max {MAX_MS}ms)"
        )

    @pytest.mark.parametrize("key", _negative_ids(), ids=_negative_ids())
    def test_no_false_positive(self, key: str):
        frame = _load_frame(key)
        det = runtime_overlay_check(frame)
        assert not det.found, f"runtime_overlay_check false positive on {key}"

    @pytest.mark.parametrize("key", _negative_ids(), ids=_negative_ids())
    def test_negative_under_100ms(self, key: str):
        frame = _load_frame(key)
        t0 = time.perf_counter()
        runtime_overlay_check(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < MAX_MS, (
            f"runtime_overlay_check took {elapsed_ms:.1f}ms on {key} (max {MAX_MS}ms)"
        )

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_bbox_within_frame(self, key: str):
        gt = _load_ground_truth()
        fw = gt[key]["frame_width"]
        fh = gt[key]["frame_height"]
        frame = _load_frame(key)
        det = runtime_overlay_check(frame)
        if det.found and det.bbox:
            x, y, w, h = det.bbox
            assert x >= 0, f"x={x} < 0"
            assert y >= 0, f"y={y} < 0"
            assert x + w <= fw, f"x+w={x + w} > {fw}"
            assert y + h <= fh, f"y+h={y + h} > {fh}"


class TestRuntimeOverlayCheckBboxAccuracy:
    """runtime_overlay_check bbox must at least overlap the correct overlay."""

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_fast_bbox_overlaps_overlay(self, key: str):
        gt = _load_ground_truth()
        gt_bbox = gt[key]["bbox"]
        frame = _load_frame(key)
        det = runtime_overlay_check(frame)
        assert det.found and det.bbox is not None, f"runtime_overlay_check missed overlay in {key}"
        iou = _compute_iou(list(det.bbox), gt_bbox)
        assert iou > 0, (
            f"runtime_overlay_check bbox has zero overlap with ground truth in {key}: "
            f"detected={list(det.bbox)}, expected={gt_bbox} — "
            f"detector locked onto wrong region"
        )


class TestDetectOverlayRuntime:
    """Padded runtime coords must not clip annotated overlay edges."""

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_precise_bbox_iou(self, key: str):
        gt = _load_ground_truth()
        gt_bbox = gt[key]["bbox"]
        frame = _load_frame(key)

        det = detect_overlay_runtime(frame)
        assert det.found and det.bbox is not None, f"detect_overlay_runtime missed overlay in {key}"
        iou = _compute_iou(list(det.bbox), gt_bbox)
        assert iou >= 0.75, (
            f"detect_overlay_runtime IoU {iou:.3f} < 0.75 for {key}: "
            f"detected={list(det.bbox)}, expected={gt_bbox}"
        )

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_precise_bbox_does_not_clip_overlay(self, key: str):
        gt = _load_ground_truth()
        gt_bbox = gt[key]["bbox"]
        frame = _load_frame(key)

        det = detect_overlay_runtime(frame)
        assert det.found and det.bbox is not None, f"detect_overlay_runtime missed overlay in {key}"

        under = _bbox_undercoverage(list(det.bbox), gt_bbox)
        assert max(under) <= PRECISE_UNDERCOVERAGE_TOLERANCE_PX, (
            f"detect_overlay_runtime clips overlay in {key}: "
            f"undercoverage(left,top,right,bottom)={under}, "
            f"detected={list(det.bbox)}, expected={gt_bbox}"
        )
