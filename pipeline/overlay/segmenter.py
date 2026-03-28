"""Auto-segment a video into layout regions using fast uniform sampling.

Samples frames at regular intervals and runs a lightweight overlay check
on each.  Consecutive frames with the same overlay layout are merged into
``LayoutSegment`` objects that become ``video_clip`` entries.

A binary-search boundary refinement pass then tightens the segment edges
to within ~1 second of the true transition.
"""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.overlay.scanner import fast_overlay_check

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

    Returns a value in [0, inf).  Values < ~0.15 indicate the overlay
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


def _read_frame_at(cap: cv2.VideoCapture, t: float, fps: float) -> np.ndarray | None:
    """Seek to time *t* and read a frame.  Returns None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    return frame if ret and frame is not None else None


def _refine_boundary(
    cap: cv2.VideoCapture,
    fps: float,
    t_has: float,
    t_not: float,
    iterations: int = 5,
) -> float:
    """Binary search for the precise overlay boundary between two times.

    *t_has* is a timestamp known to have overlay, *t_not* is a timestamp
    known to NOT have overlay.  Returns the refined boundary time.
    """
    for _ in range(iterations):
        mid = (t_has + t_not) / 2.0
        frame = _read_frame_at(cap, mid, fps)
        if frame is None:
            break
        det = fast_overlay_check(frame)
        if det.found:
            t_has = mid
        else:
            t_not = mid
    # Return the midpoint between the last known overlay and non-overlay
    return (t_has + t_not) / 2.0


def segment_video_layouts(
    video_path: str,
    sample_interval_sec: float = 5.0,
    min_overlay_fraction: float = 0.55,
) -> tuple[list[LayoutSegment], list[tuple[float, float]]]:
    """Segment a video into overlay and gap regions.

    Uses dense uniform sampling with ``fast_overlay_check`` (which
    internally downscales to 540p for speed).  Then refines segment
    boundaries with a binary search to get ~1-second precision.

    Args:
        video_path: Path to the local video file.
        sample_interval_sec: Seconds between sampled frames.
        min_overlay_fraction: Minimum overlay height as a fraction of
            frame height.  Segments with smaller overlays are treated
            as gaps (intros, outros, transitions).

    Returns:
        (segments, gaps) where *segments* have overlay detected and
        *gaps* are time ranges with no overlay.
    """
    t0 = time.monotonic()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video: fps={fps}, frames={total_frames}")

    duration = total_frames / fps
    logger.info(
        f"Segmenting {video_path}: {duration:.0f}s, {width}x{height}, "
        f"interval={sample_interval_sec}s"
    )

    # ── Phase 1: Uniform sampling ────────────────────────────
    # Sample at regular intervals.  fast_overlay_check downscales
    # internally so each call is ~5-10ms even on 1080p+ input.
    samples: list[dict] = []
    ts = 0.0

    while ts < duration:
        frame = _read_frame_at(cap, ts, fps)
        if frame is not None:
            det = fast_overlay_check(frame)
            samples.append({
                "time": ts,
                "found": det.found,
                "bbox": det.bbox if det.found else None,
                "score": det.score if det.found else 0.0,
            })
        else:
            samples.append({
                "time": ts,
                "found": False,
                "bbox": None,
                "score": 0.0,
            })
        ts += sample_interval_sec

    t_sample = time.monotonic()
    found_count = sum(1 for s in samples if s["found"])
    logger.info(
        f"Sampled {len(samples)} frames in {(t_sample - t0)*1000:.0f}ms: "
        f"{found_count} with overlay"
    )

    if not samples:
        cap.release()
        return [], []

    # ── Phase 2: Group consecutive samples into raw segments ──
    raw_segments: list[dict] = []

    for sample in samples:
        has_overlay = sample["found"]

        if not raw_segments:
            raw_segments.append({
                "start": sample["time"],
                "end": sample["time"] + sample_interval_sec,
                "bboxes": [sample["bbox"]] if sample["bbox"] else [],
                "scores": [sample["score"]] if sample["bbox"] else [],
                "has_overlay": has_overlay,
            })
            continue

        prev = raw_segments[-1]
        if prev["has_overlay"] == has_overlay:
            # Extend current segment
            prev["end"] = sample["time"] + sample_interval_sec
            if has_overlay and sample["bbox"]:
                prev["bboxes"].append(sample["bbox"])
                prev["scores"].append(sample["score"])
        else:
            # Start new segment
            raw_segments.append({
                "start": sample["time"],
                "end": sample["time"] + sample_interval_sec,
                "bboxes": [sample["bbox"]] if sample["bbox"] else [],
                "scores": [sample["score"]] if sample["bbox"] else [],
                "has_overlay": has_overlay,
            })

    # Clamp last segment end to actual duration
    if raw_segments:
        raw_segments[-1]["end"] = min(raw_segments[-1]["end"], duration)

    # ── Phase 3: Binary-search boundary refinement ────────────
    # Refine the transition points between overlay and gap segments
    # for ~1-second precision.
    for i in range(len(raw_segments) - 1):
        curr = raw_segments[i]
        nxt = raw_segments[i + 1]

        if curr["has_overlay"] != nxt["has_overlay"]:
            # There's a transition between curr.end and nxt.start
            # (which may overlap by sample_interval_sec)
            if curr["has_overlay"]:
                # overlay → gap: find where overlay ends
                t_has = curr["end"] - sample_interval_sec
                t_not = nxt["start"] + sample_interval_sec
            else:
                # gap → overlay: find where overlay starts
                t_not = curr["end"] - sample_interval_sec
                t_has = nxt["start"] + sample_interval_sec

            # Ensure t_has and t_not are in valid range
            t_has = max(0, min(t_has, duration))
            t_not = max(0, min(t_not, duration))

            if abs(t_has - t_not) > 1.0:  # Only refine if gap > 1s
                boundary = _refine_boundary(cap, fps, t_has, t_not)
                curr["end"] = boundary
                nxt["start"] = boundary

    cap.release()

    t_refine = time.monotonic()
    logger.info(
        f"Boundary refinement in {(t_refine - t_sample)*1000:.0f}ms"
    )

    # ── Phase 4: Build output ─────────────────────────────────
    # fast_overlay_check skips bbox expansion, so its bboxes are scan
    # windows (sub-regions of the actual overlay).  Use a lower size
    # threshold — any grid-regularity-passing detection is real.
    min_overlay_px = int(height * min_overlay_fraction * 0.5)
    segments: list[LayoutSegment] = []
    gaps: list[tuple[float, float]] = []

    for seg in raw_segments:
        if seg["has_overlay"] and seg["bboxes"]:
            bbox = _median_bbox(seg["bboxes"])
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
