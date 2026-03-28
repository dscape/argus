"""Auto-segment a video into layout regions using scene detection.

Uses TransNetV2 for fast shot boundary detection, then runs a lightweight
overlay check on one representative frame per scene.  Consecutive scenes
with the same overlay layout are merged into ``LayoutSegment`` objects
that become ``video_clip`` entries.
"""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.overlay.scanner import fast_overlay_check
from pipeline.overlay.scene_detector import detect_scenes

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


def _classify_scenes(
    video_path: str,
    scenes: list,
    fps: float,
) -> list[dict]:
    """Run fast overlay check on one representative frame per scene.

    Returns a list of dicts with keys: start_time, end_time, bbox, score,
    has_overlay.
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    found_count = 0

    for i, scene in enumerate(scenes):
        mid_frame = (scene.start_frame + scene.end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()

        if not ret or frame is None:
            logger.debug(
                f"Scene {i}: {scene.start_time:.1f}-{scene.end_time:.1f}s "
                f"frame {mid_frame} unreadable"
            )
            results.append({
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "bbox": None,
                "score": 0.0,
                "has_overlay": False,
            })
            continue

        det = fast_overlay_check(frame)
        if det.found:
            found_count += 1
        logger.debug(
            f"Scene {i}: {scene.start_time:.1f}-{scene.end_time:.1f}s "
            f"overlay={'YES' if det.found else 'no'} "
            f"score={det.score:.3f} bbox={det.bbox}"
        )
        results.append({
            "start_time": scene.start_time,
            "end_time": scene.end_time,
            "bbox": det.bbox if det.found else None,
            "score": det.score if det.found else 0.0,
            "has_overlay": det.found,
        })

    cap.release()
    logger.info(
        f"Classified {len(scenes)} scenes: {found_count} with overlay, "
        f"{len(scenes) - found_count} without"
    )
    return results


def _fallback_classify(
    video_path: str,
    duration: float,
    fps: float,
    sample_interval_sec: float,
) -> list[dict]:
    """Fallback: uniform sampling with fast_overlay_check when scene detection fails."""
    cap = cv2.VideoCapture(video_path)
    results = []
    ts = 0.0

    while ts < duration:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        end_time = min(ts + sample_interval_sec, duration)

        if not ret or frame is None:
            results.append({
                "start_time": ts,
                "end_time": end_time,
                "bbox": None,
                "score": 0.0,
                "has_overlay": False,
            })
            ts += sample_interval_sec
            continue

        det = fast_overlay_check(frame)
        results.append({
            "start_time": ts,
            "end_time": end_time,
            "bbox": det.bbox if det.found else None,
            "score": det.score if det.found else 0.0,
            "has_overlay": det.found,
        })
        ts += sample_interval_sec

    cap.release()
    return results


def segment_video_layouts(
    video_path: str,
    sample_interval_sec: float = 30.0,
    bbox_shift_threshold: float = 0.15,
    min_overlay_fraction: float = 0.55,
) -> tuple[list[LayoutSegment], list[tuple[float, float]]]:
    """Segment a video into layout regions using scene detection + fast overlay check.

    Uses TransNetV2 to find shot boundaries, then classifies each scene
    with a lightweight overlay detector.  Falls back to uniform sampling
    if scene detection fails.

    Args:
        video_path: Path to the local video file.
        sample_interval_sec: Seconds between sampled frames (fallback only).
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
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid video: fps={fps}, frames={total_frames}")

    duration = total_frames / fps
    logger.info(
        f"Segmenting {video_path}: {duration:.0f}s, {width}x{height}"
    )

    # ── Phase 1: Scene detection ──────────────────────────────
    try:
        scenes = detect_scenes(video_path)
        logger.info(f"TransNetV2 detected {len(scenes)} scene(s)")
        classified = _classify_scenes(video_path, scenes, fps)
    except Exception as e:
        logger.warning(f"Scene detection failed, using fallback sampling: {e}")
        classified = _fallback_classify(video_path, duration, fps, sample_interval_sec)

    if not classified:
        return [], []

    logger.info(
        f"Classified {len(classified)} scene(s) in {time.monotonic() - t0:.1f}s"
    )

    # ── Phase 2: Merge consecutive same-type scenes ───────────
    raw_segments: list[dict] = []

    for scene in classified:
        has_overlay = scene["has_overlay"]
        bbox = scene["bbox"]
        score = scene["score"]

        if not raw_segments:
            raw_segments.append({
                "start": scene["start_time"],
                "end": scene["end_time"],
                "bboxes": [bbox] if bbox else [],
                "scores": [score] if bbox else [],
                "has_overlay": has_overlay,
            })
            continue

        prev = raw_segments[-1]
        same_type = prev["has_overlay"] == has_overlay

        if same_type and has_overlay and bbox is not None:
            # Merge consecutive overlay scenes unconditionally.
            # fast_overlay_check returns approximate sub-region bboxes
            # that vary between scenes; the calibration step refines
            # the actual overlay position later.
            prev["end"] = scene["end_time"]
            prev["bboxes"].append(bbox)
            prev["scores"].append(score)
            continue

        if same_type and not has_overlay:
            prev["end"] = scene["end_time"]
            continue

        raw_segments.append({
            "start": scene["start_time"],
            "end": scene["end_time"],
            "bboxes": [bbox] if bbox else [],
            "scores": [score] if bbox else [],
            "has_overlay": has_overlay,
        })

    # ── Phase 3: Build output ─────────────────────────────────
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
