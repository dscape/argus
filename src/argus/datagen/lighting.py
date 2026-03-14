"""Lighting variation for synthetic data rendering."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class LightingConfig:
    overhead_intensity_range: tuple[float, float] = (0.5, 1.5)
    window_intensity_range: tuple[float, float] = (0.0, 0.8)
    color_temp_range: tuple[int, int] = (4000, 6500)
    num_overhead_lights: int = 6
    flicker_enabled: bool = False
    flicker_amplitude: float = 0.05


def kelvin_to_rgb(kelvin: int) -> tuple[float, float, float]:
    temp = kelvin / 100.0
    r = 1.0 if temp <= 66 else max(0.0, min(1.0, 1.292936 * ((temp - 60) ** -0.1332047592)))
    if temp <= 66:
        g = max(0.0, min(1.0, (0.3900815 * (temp ** 0.5) - 0.6318414) / 2.5))
    else:
        g = max(0.0, min(1.0, 1.129891 * ((temp - 60) ** -0.0755148492)))
    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = max(0.0, min(1.0, (0.5432068 * ((temp - 10) ** 0.5) - 1.19625) / 3.0))
    return (r, g, b)


def randomize_lighting(config: LightingConfig, seed: int = 42) -> dict:
    rng = random.Random(seed)
    return {
        "overhead_intensity": rng.uniform(*config.overhead_intensity_range),
        "window_intensity": rng.uniform(*config.window_intensity_range),
        "color_temperature": rng.randint(*config.color_temp_range),
        "color_rgb": kelvin_to_rgb(rng.randint(*config.color_temp_range)),
        "window_angle": rng.uniform(0, 360),
        "num_lights": config.num_overhead_lights,
    }
