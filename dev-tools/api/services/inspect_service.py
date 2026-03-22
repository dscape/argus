"""Video frame inspection service — fetch YouTube frames, detect overlay + OTB, auto-classify.

Fetches 4 YouTube auto-generated frame thumbnails (0.jpg–3.jpg) for each video,
runs overlay + OTB detection on each, and aggregates results.

Background job system for batch inspection with polling support.
"""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from uuid import uuid4

import cv2
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.overlay.scanner import detect_overlay_in_frame
from pipeline.screen.dual_region_detector import (
    detect_otb_region,
    overlay_bbox_to_json,
)

logger = logging.getLogger(__name__)

# ── Face detection (loaded once, bundled with OpenCV) ─────

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── In-memory job store (dev tool only) ────────────────────

_jobs: dict[str, dict] = {}

# YouTube auto-generated frame URLs (publicly accessible, no API quota)
# 0.jpg = default thumbnail, 1.jpg ≈ 25%, 2.jpg ≈ 50%, 3.jpg ≈ 75%
_FRAME_URLS = [
    ("https://img.youtube.com/vi/{video_id}/0.jpg", "thumb"),
    ("https://img.youtube.com/vi/{video_id}/1.jpg", "25%"),
    ("https://img.youtube.com/vi/{video_id}/2.jpg", "50%"),
    ("https://img.youtube.com/vi/{video_id}/3.jpg", "75%"),
]

# Minimum resolution to accept a frame (YouTube 1/2/3.jpg are 120x90)
_MIN_FRAME_WIDTH = 100


def _fetch_single_frame(url: str) -> np.ndarray | None:
    """Fetch a single image URL and return as numpy array."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None and frame.shape[1] >= _MIN_FRAME_WIDTH:
                return frame
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        pass
    return None


def _fetch_youtube_frames(video_id: str) -> list[tuple[np.ndarray, str]]:
    """Fetch all 4 YouTube auto-generated frames in parallel.

    Returns list of (frame, label) tuples.
    """
    results: list[tuple[np.ndarray, str]] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _fetch_single_frame, url_template.format(video_id=video_id)
            ): label
            for url_template, label in _FRAME_URLS
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                frame = future.result()
                if frame is not None:
                    results.append((frame, label))
            except Exception as e:
                logger.warning(f"Failed to fetch frame {label}: {e}")

    # Sort to maintain consistent order: thumb, 25%, 50%, 75%
    label_order = {label: i for i, (_, label) in enumerate(_FRAME_URLS)}
    results.sort(key=lambda x: label_order.get(x[1], 99))
    return results


def _frame_to_base64(frame: np.ndarray, max_width: int = 640) -> str:
    """Encode a BGR frame as a JPEG base64 string, optionally resizing."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _detect_person(frame: np.ndarray) -> tuple[bool, int]:
    """Detect faces/people in frame using Haar cascade. Returns (found, count)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15)
    )
    count = len(faces)
    return count > 0, count


def _analyze_frame(
    frame: np.ndarray, label: str
) -> dict:
    """Run overlay + OTB + person detection on a single frame."""
    overlay_det = detect_overlay_in_frame(frame)

    otb_found = False
    otb_confidence = 0.0
    if overlay_det.found and overlay_det.bbox:
        otb_det = detect_otb_region(frame, overlay_det.bbox)
        otb_found = otb_det.found
        otb_confidence = otb_det.confidence

    has_person, person_count = _detect_person(frame)

    return {
        "label": label,
        "overlay_found": overlay_det.found,
        "overlay_score": round(overlay_det.score, 3),
        "overlay_bbox": list(overlay_det.bbox) if overlay_det.bbox else None,
        "otb_found": otb_found,
        "otb_confidence": round(otb_confidence, 3),
        "has_person": has_person,
        "person_count": person_count,
        "image_base64": _frame_to_base64(frame),
    }


def inspect_single_video(video_id: str) -> dict:
    """Inspect a single video: extract frames, run detection, update DB.

    Returns structured result with frame thumbnails and detection scores.
    """
    # Fetch video info from DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id, channel_handle, title FROM youtube_videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()

    if row is None:
        raise ValueError(f"Video {video_id} not found")

    vid, channel_handle, title = row

    frame_results: list[dict] = []

    # Fetch all 4 YouTube auto-generated frames in parallel
    frames = _fetch_youtube_frames(video_id)
    for frame, label in frames:
        result = _analyze_frame(frame, label)
        frame_results.append(result)

    # Aggregate: take best scores across all frames
    best_overlay_score = 0.0
    best_otb_confidence = 0.0
    has_overlay = False
    has_otb = False
    has_person = False
    best_bbox = None

    for fr in frame_results:
        if fr["overlay_found"] and fr["overlay_score"] > best_overlay_score:
            best_overlay_score = fr["overlay_score"]
            has_overlay = True
            best_bbox = fr["overlay_bbox"]
        if fr["otb_found"] and fr["otb_confidence"] > best_otb_confidence:
            best_otb_confidence = fr["otb_confidence"]
            has_otb = True
        if fr.get("has_person"):
            has_person = True

    approved = has_overlay and has_otb and has_person

    # Update DB
    status = "approved" if approved else "rejected"
    bbox_json = None
    if best_bbox:
        bbox_json = overlay_bbox_to_json(tuple(best_bbox))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE youtube_videos
                SET screening_status = %s,
                    screening_confidence = %s,
                    overlay_bbox = %s,
                    has_otb_footage = %s,
                    layout_type = COALESCE(%s, layout_type),
                    updated_at = now()
                WHERE video_id = %s
                """,
                (
                    status,
                    best_otb_confidence,
                    bbox_json,
                    has_otb,
                    "overlay" if has_overlay else ("otb_only" if has_otb else None),
                    video_id,
                ),
            )
            conn.commit()

    return {
        "video_id": video_id,
        "title": title,
        "has_overlay": has_overlay,
        "has_otb": has_otb,
        "has_person": has_person,
        "overlay_score": round(best_overlay_score, 3),
        "otb_confidence": round(best_otb_confidence, 3),
        "approved": approved,
        "status": status,
        "frames": frame_results,
    }


# ── Batch job system ───────────────────────────────────────


def start_batch_job(video_ids: list[str]) -> str:
    """Start a background batch inspection job. Returns job_id."""
    job_id = uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "total": len(video_ids),
        "completed": 0,
        "approved": 0,
        "rejected": 0,
        "failed": 0,
        "results": [],
        "current_video": None,
    }
    thread = Thread(target=_run_batch, args=(job_id, video_ids), daemon=True)
    thread.start()
    return job_id


def _run_batch(job_id: str, video_ids: list[str]):
    """Process videos sequentially, updating job progress after each."""
    job = _jobs[job_id]
    for vid in video_ids:
        job["current_video"] = vid
        try:
            result = inspect_single_video(vid)
            job["results"].append(result)
            if result["approved"]:
                job["approved"] += 1
            else:
                job["rejected"] += 1
        except Exception as e:
            logger.error(f"Batch inspect failed for {vid}: {e}")
            job["results"].append({
                "video_id": vid,
                "error": str(e),
                "approved": False,
            })
            job["failed"] += 1

        job["completed"] += 1

    job["status"] = "done"
    job["current_video"] = None


def get_job_status(job_id: str) -> dict | None:
    """Return current job progress and results so far."""
    job = _jobs.get(job_id)
    if job is None:
        return None

    return {
        "job_id": job_id,
        "status": job["status"],
        "total": job["total"],
        "completed": job["completed"],
        "approved": job["approved"],
        "rejected": job["rejected"],
        "failed": job["failed"],
        "current_video": job["current_video"],
        # Only include results for completed videos (not frames — too large for polling)
        "results": [
            {
                "video_id": r["video_id"],
                "approved": r.get("approved", False),
                "has_overlay": r.get("has_overlay"),
                "has_otb": r.get("has_otb"),
                "overlay_score": r.get("overlay_score"),
                "otb_confidence": r.get("otb_confidence"),
                "status": r.get("status"),
                "error": r.get("error"),
            }
            for r in job["results"]
        ],
    }
