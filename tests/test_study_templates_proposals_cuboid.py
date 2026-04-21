from __future__ import annotations

import numpy as np
from study.templates.proposals.common import ProposalFrame
from study.templates.proposals.cuboid import propose_cuboid
from study.templates.shared import load_base_head_data_module


def test_propose_cuboid_returns_64_entries_with_expected_crop_dimensions() -> None:
    frame = ProposalFrame(
        image_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
        corners=((0.0, 0.0), (223.0, 0.0), (223.0, 223.0), (0.0, 223.0)),
        frame_id="synthetic",
    )

    proposals = propose_cuboid(frame)

    assert len(proposals) == 64
    assert proposals[0].square == "a8"
    assert proposals[-1].square == "h1"
    assert all(proposal.crop_bgr.shape == (224, 224, 3) for proposal in proposals)


def test_propose_cuboid_delegates_to_base_head_crop_extractor(monkeypatch) -> None:
    base_head_data = load_base_head_data_module()
    calls: list[tuple[int, int]] = []

    def fake_extract_study_piece_crop(
        image_bgr: np.ndarray,
        corners: tuple[tuple[float, float], ...],
        *,
        row: int,
        col: int,
        output_size: int,
        piece_height: float,
        flip_left_half: bool,
    ) -> np.ndarray:
        del image_bgr, corners, piece_height, flip_left_half
        calls.append((row, col))
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    monkeypatch.setattr(base_head_data, "extract_study_piece_crop", fake_extract_study_piece_crop)
    frame = ProposalFrame(
        image_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
        corners=((0.0, 0.0), (223.0, 0.0), (223.0, 223.0), (0.0, 223.0)),
        frame_id="synthetic",
    )

    proposals = propose_cuboid(frame)

    assert len(proposals) == 64
    assert calls[0] == (0, 0)
    assert calls[-1] == (7, 7)
    assert len(calls) == 64
