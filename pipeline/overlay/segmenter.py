"""Auto-segment a video into layout regions based on overlay detection.

Samples frames at regular intervals and groups consecutive frames with
similar overlay positions into segments.  Each segment becomes a
``video_clip`` entry with a preliminary overlay bbox that the calibration
step later refines.
"""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.overlay.scanner import detect_overlay_in_frame

logger = logging.getLogger(__name__)


@dataclass
class LayoutSegment:
    """A contiguous region of a video with a consistent overlay layout."""

    start_time: float
    end_time: float
    overlay_bbox: tuple[int, int, int, int] | None  # consensus (x, y, w, h), None = gap
    score: float  # average detection score
    sample_count: int


def _bbox_relative_shift(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Max relative shift between two bboxes (x, y, w, h).

    Returns a value in [0, ∞).  Values < ~0.15 indicate the overlay
    hasn't meaningfully moved.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ref_w = max(aw, bw, 1)
    ref_h = max(ah, bh, 1)
    return max(
        abs(ax - bx) / ref_w,
        abs(ay - by) / ref_h,
        abs(aw - bw) / ref_w,
        abs(ah - bh) / ref_h,
    )


def _median_bbox(
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Compute the element-wise median of a list of bboxes."""
    arr = np.array(bboxes)
    med = np.median(arr, axis=0).astype(int)
    return (int(med[0]), int(med[1]), int(med[2]), int(med[3]))


def segment_video_layouts(
    video_path: str,
    sample_interval_sec: float = 30.0,
    bbox_shift_threshold: float = 0.15,
    min_overlay_fraction: float = 0.55,
) -> tuple[list[LayoutSegment], list[tuple[float, float]]]:
    """Segment a video into layout regions by sampling overlay detection.

    Args:
        video_path: Path to the local video file.
        sample_interval_sec: Seconds between sampled frames (default 30).
        bbox_shift_threshold: Max relative bbox shift to consider "same layout".
        min_overlay_fraction: Minimum overlay height as a fraction of frame
            height.  Segments with smaller overlays are treated as gaps
            (intros, outros, transitions).

    Returns:
        (segments, gaps) where *segments* have overlay detected and *gaps*
        are time ranges with no overlay.
    """
    t0 = time.monotonic()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video: fps={fps}, frames={total_frames}")

    duration = total_frames / fps
    logger.info(
        f"Segmenting {video_path}: {duration:.0f}s, {width}x{height}, "
        f"sampling every {sample_interval_sec}s"
    )

    # ── Sample frames and run overlay detection ──────────────
    detections: list[tuple[float, tuple[int, int, int, int] | None, float]] = []
    ts = 0.0
    while ts < duration:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            ts += sample_interval_sec
            continue

        det = detect_overlay_in_frame(frame)
        bbox = det.bbox if det.found else None
        score = det.score if det.found else 0.0
        detections.append((ts, bbox, score))

        ts += sample_interval_sec

    cap.release()

    if not detections:
        return [], []

    logger.info(f"Sampled {len(detections)} frames in {time.monotonic() - t0:.1f}s")

    # ── Group into raw segments ──────────────────────────────
    raw_segments: list[dict] = []  # {start, end, bboxes, scores, has_overlay}

    for ts_val, bbox, score in detections:
        has_overlay = bbox is not None
        end_time = min(ts_val + sample_interval_sec, duration)

        if not raw_segments:
            raw_segments.append({
                "start": ts_val,
                "end": end_time,
                "bboxes": [bbox] if bbox else [],
                "scores": [score] if bbox else [],
                "has_overlay": has_overlay,
            })
            continue

        prev = raw_segments[-1]
        same_type = prev["has_overlay"] == has_overlay

        if same_type and has_overlay and bbox is not None and prev["bboxes"]:
            # Both have overlay — check if bbox shifted
            shift = _bbox_relative_shift(prev["bboxes"][-1], bbox)
            if shift < bbox_shift_threshold:
                prev["end"] = end_time
                prev["bboxes"].append(bbox)
                prev["scores"].append(score)
                continue

        if same_type and not has_overlay:
            # Both gaps — extend
            prev["end"] = end_time
            continue

        # Layout change — start new segment
        raw_segments.append({
            "start": ts_val,
            "end": end_time,
            "bboxes": [bbox] if bbox else [],
            "scores": [score] if bbox else [],
            "has_overlay": has_overlay,
        })

    # ── Merge tiny segments (< 2 samples) into neighbors ────
    merged: list[dict] = []
    for seg in raw_segments:
        n_samples = len(seg["bboxes"]) if seg["has_overlay"] else max(1, int((seg["end"] - seg["start"]) / sample_interval_sec))
        if n_samples < 2 and merged:
            # Absorb into previous segment
            merged[-1]["end"] = seg["end"]
            if seg["has_overlay"]:
                merged[-1]["bboxes"].extend(seg["bboxes"])
                merged[-1]["scores"].extend(seg["scores"])
                if not merged[-1]["has_overlay"] and seg["bboxes"]:
                    merged[-1]["has_overlay"] = True
        else:
            merged.append(seg)

    # ── Build output ─────────────────────────────────────────
    min_overlay_px = int(height * min_overlay_fraction)
    segments: list[LayoutSegment] = []
    gaps: list[tuple[float, float]] = []

    for seg in merged:
        if seg["has_overlay"] and seg["bboxes"]:
            bbox = _median_bbox(seg["bboxes"])
            # Skip segments where the overlay is too small — likely an
            # intro, outro, or transition rather than actual gameplay.
            if bbox[2] < min_overlay_px or bbox[3] < min_overlay_px:
                logger.info(
                    f"Skipping segment {seg['start']:.0f}-{seg['end']:.0f}s: "
                    f"overlay {bbox[2]}x{bbox[3]} below minimum {min_overlay_px}px"
                )
                gaps.append((seg["start"], seg["end"]))
                continue
            segments.append(LayoutSegment(
                start_time=seg["start"],
                end_time=seg["end"],
                overlay_bbox=bbox,
                score=float(np.mean(seg["scores"])),
                sample_count=len(seg["bboxes"]),
            ))
        else:
            gaps.append((seg["start"], seg["end"]))

    elapsed = time.monotonic() - t0
    logger.info(
        f"Segmentation complete: {len(segments)} segment(s), "
        f"{len(gaps)} gap(s), {elapsed:.1f}s"
    )

    return segments, gaps
