"""Camera placement and motion simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class CameraConfig:
    fov: float = 90.0
    height: float = 5.0
    distance: float = 8.0
    angle: float = 30.0
    pan_speed: float = 0.5
    jitter_std: float = 0.1


def compute_camera_trajectory(
    config: CameraConfig, num_frames: int, fps: float = 5.0, seed: int = 42,
) -> list[dict[str, tuple[float, float, float]]]:
    rng = random.Random(seed)
    trajectory: list[dict[str, tuple[float, float, float]]] = []
    for frame in range(num_frames):
        t = frame / fps
        azimuth = config.pan_speed * t
        azimuth_rad = math.radians(azimuth)
        x = config.distance * math.sin(azimuth_rad) + rng.gauss(0, config.jitter_std)
        y = -config.distance * math.cos(azimuth_rad) + rng.gauss(0, config.jitter_std)
        z = config.height + rng.gauss(0, config.jitter_std * 0.3)
        tilt_rad = math.radians(config.angle)
        rot_x = math.pi / 2 - tilt_rad
        trajectory.append({"position": (x, y, z), "rotation": (rot_x, 0.0, azimuth_rad + math.pi)})
    return trajectory
