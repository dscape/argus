"""Player hand/body occlusion simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class HandMotion:
    player_side: str
    source_square: tuple[int, int]
    target_square: tuple[int, int]
    start_frame: int
    end_frame: int
    peak_frame: int


@dataclass
class OcclusionConfig:
    hand_size: float = 0.15
    arm_length: float = 0.4
    move_duration_range: tuple[float, float] = (0.5, 3.0)
    body_occlusion: bool = True
    body_width: float = 0.4


def compute_hand_trajectory(
    motion: HandMotion, board_size: float = 0.45, fps: float = 5.0,
) -> list[dict[str, tuple[float, float, float]]]:
    sq_size = board_size / 8
    src_x = (motion.source_square[0] - 3.5) * sq_size
    src_y = (motion.source_square[1] - 3.5) * sq_size
    tgt_x = (motion.target_square[0] - 3.5) * sq_size
    tgt_y = (motion.target_square[1] - 3.5) * sq_size
    rest_y = -board_size * 0.8 if motion.player_side == "white" else board_size * 0.8
    rest_x = 0.0
    trajectory: list[dict[str, tuple[float, float, float]]] = []
    total_frames = motion.end_frame - motion.start_frame
    for f in range(total_frames):
        t = f / max(total_frames - 1, 1)
        if t < 0.3:
            phase_t = t / 0.3
            x = rest_x + (src_x - rest_x) * phase_t
            y = rest_y + (src_y - rest_y) * phase_t
            z = 0.05 * math.sin(phase_t * math.pi)
        elif t < 0.7:
            phase_t = (t - 0.3) / 0.4
            x = src_x + (tgt_x - src_x) * phase_t
            y = src_y + (tgt_y - src_y) * phase_t
            z = 0.08 * math.sin(phase_t * math.pi)
        else:
            phase_t = (t - 0.7) / 0.3
            x = tgt_x + (rest_x - tgt_x) * phase_t
            y = tgt_y + (rest_y - tgt_y) * phase_t
            z = 0.03 * math.sin(phase_t * math.pi)
        trajectory.append({"position": (x, y, z), "scale": (0.15, 0.08, 0.03)})
    return trajectory


def generate_move_occlusions(
    move_frames: list[tuple[int, str, str]], fps: float = 5.0, seed: int = 42,
) -> list[HandMotion]:
    rng = random.Random(seed)
    motions: list[HandMotion] = []
    is_white_turn = True
    for frame_idx, src_sq, tgt_sq in move_frames:
        src_file, src_rank = ord(src_sq[0]) - ord('a'), int(src_sq[1]) - 1
        tgt_file, tgt_rank = ord(tgt_sq[0]) - ord('a'), int(tgt_sq[1]) - 1
        duration_frames = max(3, int(rng.uniform(0.5, 3.0) * fps))
        start = max(0, frame_idx - duration_frames // 2)
        end = frame_idx + duration_frames // 2
        motions.append(HandMotion(
            player_side="white" if is_white_turn else "black",
            source_square=(src_file, src_rank), target_square=(tgt_file, tgt_rank),
            start_frame=start, end_frame=end, peak_frame=frame_idx,
        ))
        is_white_turn = not is_white_turn
    return motions
