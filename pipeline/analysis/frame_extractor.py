"""Video frame extraction helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """A single extracted video frame."""

    index: int
    timestamp: float
    image: np.ndarray


def extract_frames(video_path: str | Path, fps: float = 1.0) -> Iterator[FrameData]:
    """Extract frames from a video at the target FPS."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream_fps = float(stream.average_rate or 30.0)
    frame_skip = max(1, int(stream_fps / fps))

    logger.info(
        "Extracting frames: %s (%.1f fps, skip=%d)",
        video_path.name,
        stream_fps,
        frame_skip,
    )

    frame_count = 0
    extracted = 0
    for frame in container.decode(video=0):
        if frame_count % frame_skip == 0:
            image = frame.to_ndarray(format="rgb24")
            timestamp = float(frame.time) if frame.time is not None else frame_count / stream_fps
            yield FrameData(index=extracted, timestamp=timestamp, image=image)
            extracted += 1
        frame_count += 1

    container.close()
    logger.info("Extracted %d frames from %d total", extracted, frame_count)


def sample_frames(video_path: str | Path, count: int = 8) -> list[FrameData]:
    """Sample evenly spaced frames from a video."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    stream_fps = float(stream.average_rate or 30.0)

    if total_frames <= 0:
        for _ in container.decode(video=0):
            total_frames += 1
        container.close()
        container = av.open(str(video_path))

    if total_frames <= count:
        target_indices = set(range(total_frames))
    else:
        step = total_frames / count
        target_indices = {int(i * step) for i in range(count)}

    frames: list[FrameData] = []
    frame_idx = 0
    sample_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in target_indices:
            image = frame.to_ndarray(format="rgb24")
            timestamp = float(frame.time) if frame.time is not None else frame_idx / stream_fps
            frames.append(FrameData(index=sample_idx, timestamp=timestamp, image=image))
            sample_idx += 1
        frame_idx += 1

    container.close()
    logger.info("Sampled %d frames from %s", len(frames), video_path.name)
    return frames
