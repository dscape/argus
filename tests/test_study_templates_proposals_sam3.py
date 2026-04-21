from __future__ import annotations

from pathlib import Path

import numpy as np
from study.templates.proposals.common import ProposalFrame, SquareCropProposal
from study.templates.proposals.sam3_source import (
    Sam3MaskCandidate,
    Sam3ProposalConfig,
    propose_sam3,
    render_sam3_preview,
)


def test_propose_sam3_returns_one_square_tagged_entry_per_detected_piece(monkeypatch) -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    frame = ProposalFrame(
        image_bgr=image,
        corners=((0.0, 0.0), (127.0, 0.0), (127.0, 127.0), (0.0, 127.0)),
        frame_id="synthetic",
    )
    mask_a8 = np.zeros((128, 128), dtype=bool)
    mask_a8[4:14, 4:14] = True
    mask_h1 = np.zeros((128, 128), dtype=bool)
    mask_h1[114:124, 114:124] = True

    monkeypatch.setattr(
        "study.templates.proposals.sam3_source._collect_prompt_candidates",
        lambda _board_image_bgr, *, config: [
            Sam3MaskCandidate(mask_local=mask_a8, bbox_local=(4, 4, 14, 14), score=0.9),
            Sam3MaskCandidate(mask_local=mask_h1, bbox_local=(114, 114, 124, 124), score=0.8),
        ],
    )

    proposals = propose_sam3(frame, config=Sam3ProposalConfig())

    assert [proposal.square for proposal in proposals] == ["a8", "h1"]
    assert all(proposal.crop_bgr.size > 0 for proposal in proposals)


def test_render_sam3_preview_writes_contact_sheet(tmp_path: Path) -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    frame = ProposalFrame(
        image_bgr=image,
        corners=((0.0, 0.0), (127.0, 0.0), (127.0, 127.0), (0.0, 127.0)),
        frame_id="synthetic",
    )
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[4:14, 4:14] = 255
    output_path = tmp_path / "sam3_preview.png"

    render_sam3_preview(
        frame,
        [
            SquareCropProposal(
                square="a8",
                crop_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
                score=0.9,
                bbox=(4, 4, 14, 14),
                mask=mask,
            )
        ],
        output_path=output_path,
        config=Sam3ProposalConfig(crop_cell_size=64),
    )

    assert output_path.exists()
