#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from study.templates.builder.template_bank import build_replay_clip_selection
from study.templates.proposals.common import ProposalFrame
from study.templates.proposals.sam3_source import (
    Sam3ProposalConfig,
    propose_sam3,
    render_sam3_preview,
)
from study.templates.shared import load_base_head_data_module


def main() -> None:
    args = build_parser().parse_args()
    selection = build_replay_clip_selection(
        args.clip_path,
        tournament_id=args.tournament_id,
        max_frames=1,
        native_frame_stride=1,
    )
    row = selection.rows[0]
    base_head_data = load_base_head_data_module()
    native_loader = base_head_data.NativeFrameLoader()
    try:
        image_bgr, corners = base_head_data.load_row_frame_and_corners(
            row,
            clip_cache={},
            native_loader=native_loader,
        )
    finally:
        native_loader.close()

    frame = ProposalFrame(image_bgr=image_bgr, corners=corners, frame_id=str(row.row_id))
    config = Sam3ProposalConfig(
        confidence_threshold=args.confidence_threshold,
        board_crop_margin=args.board_crop_margin,
        crop_cell_size=args.crop_cell_size,
    )
    proposals = propose_sam3(frame, config=config)
    render_sam3_preview(frame, proposals, output_path=args.output_path, config=config)
    print(
        json.dumps(
            {
                "frame_id": frame.frame_id,
                "proposal_count": len(proposals),
                "squares": [proposal.square for proposal in proposals],
                "output_path": str(Path(args.output_path).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a SAM3 proposal preview for one frame.")
    parser.add_argument(
        "--clip-path",
        type=Path,
        default=(
            _PROJECT_ROOT / "data" / "argus" / "train_real" / "clip_overlay_2wWUKmCBr6A_clip5_0.pt"
        ),
    )
    parser.add_argument("--tournament-id", type=str, default="2wWUKmCBr6A")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_PROJECT_ROOT / "study" / "templates" / "proposals" / "sam3_preview.png",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--board-crop-margin", type=float, default=0.15)
    parser.add_argument("--crop-cell-size", type=int, default=144)
    return parser


if __name__ == "__main__":
    main()
