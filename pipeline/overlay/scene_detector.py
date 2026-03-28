"""Detect scene/shot boundaries in video using TransNetV2.

TransNetV2 processes downscaled frames (48x27) in bulk and predicts
per-frame shot transition probabilities.  This is orders of magnitude
faster than sampling individual frames for overlay detection.
"""

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

import cv2

logger = logging.getLogger(__name__)

# Lazy-loaded TransNetV2 model (singleton).
_model = None


@dataclass
class SceneBoundary:
    """A contiguous scene between two shot transitions."""

    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float  # seconds


def _get_model():
    """Lazy-load the TransNetV2 model on first use."""
    global _model
    if _model is None:
        from transnetv2_pytorch import TransNetV2

        _model = TransNetV2(device="auto")
        _model.eval()
        logger.info("TransNetV2 model loaded")
    return _model


def _downscale_video(video_path: str) -> str | None:
    """Pre-downscale video to 48x27 for faster TransNetV2 processing on CPU.

    TransNetV2 internally downscales to 48x27 anyway; providing a pre-downscaled
    file eliminates the cost of decoding full-resolution frames on CPU.

    Returns path to a temporary file (caller must delete it), or None on failure.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()

        t0 = time.monotonic()
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", "scale=96:54",   # 400× less pixels than 1080p; TransNetV2 resizes internally
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                tmp.name,
            ],
            capture_output=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning(
                f"ffmpeg downscale failed (rc={result.returncode}): "
                f"{result.stderr.decode()[:300]}"
            )
            os.unlink(tmp.name)
            return None

        size_mb = os.path.getsize(tmp.name) / 1024 / 1024
        logger.info(
            f"Pre-downscaled to 48x27 in {time.monotonic() - t0:.1f}s "
            f"({size_mb:.1f} MB) → {tmp.name}"
        )
        return tmp.name
    except Exception as e:
        logger.warning(f"ffmpeg downscale error: {e}")
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return None


def detect_scenes(
    video_path: str,
    threshold: float = 0.5,
) -> list[SceneBoundary]:
    """Detect shot boundaries in a video using TransNetV2.

    Returns a list of scenes covering the full video duration.
    Falls back to a single scene (whole video) on failure.

    Args:
        video_path: Path to the local video file.
        threshold: Shot transition confidence threshold (0-1).

    Returns:
        List of SceneBoundary objects sorted by start_time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid video metadata: fps={fps}, frames={total_frames}")

    duration = total_frames / fps

    # Pre-downscale to 48x27 so TransNetV2 doesn't decode full-res frames on CPU.
    downscaled_path = _downscale_video(video_path)
    process_path = downscaled_path if downscaled_path else video_path

    try:
        model = _get_model()
        raw_scenes = model.detect_scenes(process_path, threshold=threshold)
    except Exception as e:
        logger.warning(f"TransNetV2 failed, falling back to single scene: {e}")
        raw_scenes = None
    finally:
        if downscaled_path:
            try:
                os.unlink(downscaled_path)
            except Exception:
                pass

    if not raw_scenes:
        return [SceneBoundary(
            start_frame=0,
            end_frame=total_frames - 1,
            start_time=0.0,
            end_time=duration,
        )]

    boundaries = []
    for scene in raw_scenes:
        sf = scene["start_frame"]
        ef = scene["end_frame"]
        boundaries.append(SceneBoundary(
            start_frame=sf,
            end_frame=ef,
            start_time=sf / fps,
            end_time=ef / fps,
        ))

    return boundaries
