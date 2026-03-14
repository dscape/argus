#!/usr/bin/env python3
"""Inference entry point: video to PGN."""

import argparse
import logging
import sys

import torch

from argus.inference.pipeline import InferencePipeline
from argus.model.argus_model import ArgusModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus inference: video to PGN")
    parser.add_argument("--video", type=str, help="Input video file")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output_pgn")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--detect-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    model = ArgusModel(use_detector=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model from {args.checkpoint}")

    pipeline = InferencePipeline(
        model=model, device=device, detect_threshold=args.detect_threshold,
        move_confidence_threshold=args.confidence_threshold, fps=args.fps,
    )

    if args.video:
        tracks = pipeline.process_video(video_path=args.video, output_dir=args.output_dir)
        for track in tracks:
            logger.info(f"Board {track.board_id}: {len(track.moves)} moves, status={track.status}")
            print(f"\n--- Board {track.board_id} PGN ---")
            print(track.pgn)
    else:
        logger.error("Please provide --video path")
        sys.exit(1)


if __name__ == "__main__":
    main()
