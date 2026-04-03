"""Overlay detection tests driven by manually annotated ground truth.

Ground truth is stored at tests/fixtures/frames/ground_truth.json,
created via the dev-tools Overlay BBox annotation UI.

Tests target fast_overlay_check which must:
- Detect overlays within reasonable IoU of ground truth
- Run in under 100ms per frame
- Not produce false positives on negative frames
"""

import json
import time
from pathlib import Path

import cv2
import pytest
from pipeline.overlay.scanner import fast_overlay_check

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "frames"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

MAX_MS = 200
IOU_THRESHOLD = 0.40


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


class TestFastOverlayCheck:
    """fast_overlay_check must detect overlays accurately under 100ms."""

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_detects_overlay(self, key: str):
        frame = _load_frame(key)
        det = fast_overlay_check(frame)
        assert det.found, f"fast_overlay_check missed overlay in {key}"

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_bbox_iou(self, key: str):
        gt = _load_ground_truth()
        gt_bbox = gt[key]["bbox"]
        frame = _load_frame(key)
        det = fast_overlay_check(frame)
        assert det.found and det.bbox is not None
        iou = _compute_iou(list(det.bbox), gt_bbox)
        assert iou >= IOU_THRESHOLD, (
            f"IoU {iou:.3f} < {IOU_THRESHOLD} for {key}: "
            f"detected={list(det.bbox)}, expected={gt_bbox}"
        )

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_under_100ms(self, key: str):
        frame = _load_frame(key)
        t0 = time.perf_counter()
        fast_overlay_check(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < MAX_MS, (
            f"fast_overlay_check took {elapsed_ms:.1f}ms on {key} (max {MAX_MS}ms)"
        )

    @pytest.mark.parametrize("key", _negative_ids(), ids=_negative_ids())
    def test_no_false_positive(self, key: str):
        frame = _load_frame(key)
        det = fast_overlay_check(frame)
        assert not det.found, f"fast_overlay_check false positive on {key}"

    @pytest.mark.parametrize("key", _negative_ids(), ids=_negative_ids())
    def test_negative_under_100ms(self, key: str):
        frame = _load_frame(key)
        t0 = time.perf_counter()
        fast_overlay_check(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < MAX_MS, (
            f"fast_overlay_check took {elapsed_ms:.1f}ms on {key} (max {MAX_MS}ms)"
        )

    @pytest.mark.parametrize("key", _positive_ids(), ids=_positive_ids())
    def test_bbox_within_frame(self, key: str):
        gt = _load_ground_truth()
        fw = gt[key]["frame_width"]
        fh = gt[key]["frame_height"]
        frame = _load_frame(key)
        det = fast_overlay_check(frame)
        if det.found and det.bbox:
            x, y, w, h = det.bbox
            assert x >= 0, f"x={x} < 0"
            assert y >= 0, f"y={y} < 0"
            assert x + w <= fw, f"x+w={x + w} > {fw}"
            assert y + h <= fh, f"y+h={y + h} > {fh}"
