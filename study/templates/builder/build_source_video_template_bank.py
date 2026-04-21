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

from study.templates.builder.template_bank import (
    TemplateBankConfig,
    build_source_video_selection,
    build_template_bank,
)


def main() -> None:
    args = build_parser().parse_args()
    selection = build_source_video_selection(
        args.source_video_id,
        source_frame_indices=[int(value) for value in args.source_frame_indices],
        tournament_id=args.tournament_id,
    )
    output_path = args.output_path or (
        _PROJECT_ROOT / "study" / "templates" / "data" / f"{selection.tournament_id}.pt"
    )
    preview_path = args.preview_path or (
        _PROJECT_ROOT / "study" / "templates" / "data" / f"{selection.tournament_id}_preview.png"
    )
    payload = build_template_bank(
        selection.rows,
        tournament_id=selection.tournament_id,
        output_path=output_path,
        preview_path=preview_path,
        config=TemplateBankConfig(
            encoder_type=args.encoder_type,
            model_name=args.model_name,
            input_size=args.input_size,
            device=args.device,
            piece_height=args.piece_height,
            flip_left_half=not args.no_flip_left_half,
            jitter_variations=args.jitter_variations,
            jitter_pixels=args.jitter_pixels,
            batch_size=args.batch_size,
            preview_samples_per_piece_type=args.preview_samples_per_piece_type,
            max_base_crops_per_piece_type=args.max_base_crops_per_piece_type,
            temporal_consistency_weight=args.temporal_consistency_weight,
        ),
    )
    print(
        json.dumps(
            {
                "tournament_id": selection.tournament_id,
                "source_video_id": selection.source_video_id,
                "source_frame_indices": selection.source_frame_indices,
                "base_crop_counts_by_piece_type": payload["base_crop_counts_by_piece_type"],
                "embedding_counts_by_piece_type": payload["embedding_counts_by_piece_type"],
                "output_path": str(Path(output_path).resolve()),
                "preview_path": str(Path(preview_path).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a template bank from explicit source-video setup frames."
    )
    parser.add_argument("--source-video-id", type=str, required=True)
    parser.add_argument("--source-frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--tournament-id", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--preview-path", type=Path, default=None)
    parser.add_argument("--encoder-type", type=str, default="dinov3")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--piece-height", type=float, default=2.0)
    parser.add_argument("--jitter-variations", type=int, default=9)
    parser.add_argument("--jitter-pixels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preview-samples-per-piece-type", type=int, default=4)
    parser.add_argument("--max-base-crops-per-piece-type", type=int, default=4)
    parser.add_argument("--temporal-consistency-weight", type=float, default=8.0)
    parser.add_argument("--no-flip-left-half", action="store_true")
    return parser


if __name__ == "__main__":
    main()
