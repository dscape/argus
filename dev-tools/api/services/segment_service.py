"""Auto-segment a video into layout regions and auto-calibrate individual clips."""

import base64
import logging
import time

import cv2
import numpy as np

from api.services import crawl_service
from pipeline.overlay.auto_calibration import (
    _get_video_path,
    _scale_bbox,
    compute_camera_bbox,
    detect_board_orientation,
    detect_board_theme,
    detect_overlay_in_frame,
    propose_calibration_for_clip,
)
from pipeline.overlay.segmenter import segment_video_layouts

logger = logging.getLogger(__name__)


def _frame_to_base64(frame: np.ndarray, max_width: int = 640) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def auto_segment_video(
    video_id: str,
    sample_interval_sec: float = 30.0,
    replace_existing: bool = False,
) -> dict:
    """Run layout segmentation on a downloaded video and create video_clips entries.

    Returns a dict with ``segments``, ``gaps``, ``video_resolution``, etc.
    """
    from pipeline.db.connection import get_conn

    video_path = _get_video_path(video_id)
    if not video_path:
        raise ValueError(f"Video {video_id} is not downloaded")

    # Get video resolution for ref_resolution
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Optionally delete existing clips
    if replace_existing:
        existing = crawl_service.list_video_clips(video_id)
        for clip in existing:
            crawl_service.delete_video_clip(clip["id"])

    t0 = time.monotonic()
    segments, gaps = segment_video_layouts(video_path, sample_interval_sec)
    elapsed = time.monotonic() - t0

    # Count existing clips to check for overlap
    existing_clips = crawl_service.list_video_clips(video_id)
    if existing_clips and not replace_existing:
        # Don't create clips if some already exist (user should use replace_existing)
        return {
            "error": f"Video already has {len(existing_clips)} clip(s). "
            "Set replace_existing=true to replace them.",
            "existing_clips": len(existing_clips),
        }

    # Create video_clips entries
    created_segments = []
    for i, seg in enumerate(segments):
        clip_data = {
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "label": f"Segment {i + 1}",
            "overlay_bbox": list(seg.overlay_bbox) if seg.overlay_bbox else [0, 0, 100, 100],
            "camera_bbox": [0, 0, 100, 100],  # placeholder — calibrate step will fill
            "ref_resolution": [width, height],
            "board_flipped": False,
            "board_theme": "lichess_default",
        }
        created = crawl_service.create_video_clip(video_id, clip_data)
        created_segments.append({
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "overlay_bbox": list(seg.overlay_bbox) if seg.overlay_bbox else None,
            "score": round(seg.score, 3),
            "sample_count": seg.sample_count,
            "clip_id": created["id"],
        })

    return {
        "segments": created_segments,
        "gaps": [{"start_time": g[0], "end_time": g[1]} for g in gaps],
        "video_resolution": [width, height],
        "total_frames_sampled": sum(s.sample_count for s in segments),
        "processing_time_sec": round(elapsed, 1),
    }


def auto_calibrate_clip(video_id: str, clip_id: int) -> dict:
    """Auto-calibrate a single clip from actual video frames.

    Runs overlay detection, theme/orientation detection, and camera bbox
    computation within the clip's time range, then updates the clip in
    the database.
    """
    clip = crawl_service.get_video_clip(clip_id)
    if clip is None:
        raise ValueError(f"Clip {clip_id} not found")
    if clip["video_id"] != video_id:
        raise ValueError(f"Clip {clip_id} does not belong to video {video_id}")

    video_path = _get_video_path(video_id)
    if not video_path:
        raise ValueError(f"Video {video_id} is not downloaded")

    # Get video resolution
    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ref_resolution = (vid_w, vid_h)
    proposal = propose_calibration_for_clip(
        video_path,
        start_time=clip["start_time"],
        end_time=clip["end_time"],
        ref_resolution=ref_resolution,
    )

    if proposal is None:
        return {
            "clip_id": clip_id,
            "proposal": None,
            "applied": False,
            "preview_frame_b64": None,
            "camera_heatmap_b64": None,
        }

    # Update the clip in the DB
    crawl_service.update_video_clip(clip_id, {
        "overlay_bbox": list(proposal.overlay),
        "camera_bbox": list(proposal.camera),
        "ref_resolution": list(proposal.ref_resolution),
        "board_flipped": proposal.board_flipped,
        "board_theme": proposal.board_theme,
    })

    # Generate preview frame with bboxes drawn
    preview_b64 = None
    heatmap_b64 = None

    cap = cv2.VideoCapture(video_path)
    mid_time = (clip["start_time"] + (clip["end_time"] or 0)) / 2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
    ret, frame = cap.read()
    if ret:
        annotated = frame.copy()
        ox, oy, ow, oh = proposal.overlay
        cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
        cv2.putText(
            annotated, "Overlay", (ox + 6, oy + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
        cx, cy, cw, ch = proposal.camera
        cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 3)
        cv2.putText(
            annotated, "Camera", (cx + 6, cy + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
        )
        preview_b64 = _frame_to_base64(annotated)

        # Camera motion heatmap: compare two frames within clip
        clip_end = clip["end_time"] or mid_time * 2
        t1 = clip["start_time"] + (clip_end - clip["start_time"]) * 0.25
        t2 = clip["start_time"] + (clip_end - clip["start_time"]) * 0.75
        cap.set(cv2.CAP_PROP_POS_MSEC, t1 * 1000)
        ret1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, t2 * 1000)
        ret2, f2 = cap.read()
        if ret1 and ret2:
            gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if gray2.shape != gray1.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0])).astype(np.float32)
            diff = np.abs(gray1 - gray2)
            diff[oy: oy + oh, ox: ox + ow] = 0
            diff_norm = np.clip(diff / 30.0 * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
            heatmap_b64 = _frame_to_base64(heatmap)

    cap.release()

    return {
        "clip_id": clip_id,
        "proposal": {
            "overlay_bbox": list(proposal.overlay),
            "camera_bbox": list(proposal.camera),
            "board_theme": proposal.board_theme,
            "theme_confidence": round(proposal.theme_confidence, 3),
            "board_flipped": proposal.board_flipped,
            "orientation_confidence": round(proposal.orientation_confidence, 3),
            "ref_resolution": list(proposal.ref_resolution),
        },
        "applied": True,
        "preview_frame_b64": preview_b64,
        "camera_heatmap_b64": heatmap_b64,
    }
