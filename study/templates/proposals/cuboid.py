from __future__ import annotations

from dataclasses import dataclass

from pipeline.physical.piece_projection import DEFAULT_PIECE_HEIGHT
from study.templates.proposals.common import ProposalFrame, SquareCropProposal
from study.templates.shared import load_base_head_data_module


@dataclass(frozen=True)
class CuboidProposalConfig:
    input_size: int = 224
    piece_height: float = DEFAULT_PIECE_HEIGHT
    flip_left_half: bool = True


def propose_cuboid(
    frame: ProposalFrame,
    *,
    config: CuboidProposalConfig | None = None,
) -> list[SquareCropProposal]:
    resolved_config = config or CuboidProposalConfig()
    base_head_data = load_base_head_data_module()
    proposals: list[SquareCropProposal] = []
    for square_index in range(64):
        crop_bgr = base_head_data.extract_study_piece_crop(
            frame.image_bgr,
            frame.corners,
            row=square_index // 8,
            col=square_index % 8,
            output_size=resolved_config.input_size,
            piece_height=resolved_config.piece_height,
            flip_left_half=resolved_config.flip_left_half,
        )
        proposals.append(
            SquareCropProposal(
                square=base_head_data.index_to_square_name(square_index),
                crop_bgr=crop_bgr,
            )
        )
    return proposals


__all__ = [
    "CuboidProposalConfig",
    "propose_cuboid",
]
