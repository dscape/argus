#!/usr/bin/env python3
"""Synthetic data generation entry point."""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--type", choices=["2d", "3d"], default="2d")
    parser.add_argument("--num-clips", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frames-per-move", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "2d":
        from argus.datagen.synth2d import generate_dataset
        logger.info(f"Generating {args.num_clips} 2D synthetic clips...")

        pbar = tqdm(total=args.num_clips, desc="Generating clips", unit="clip")

        def on_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        clips = generate_dataset(
            num_clips=args.num_clips, clip_length=args.clip_length,
            image_size=args.image_size, frames_per_move=args.frames_per_move,
            output_dir=str(output_dir), seed=args.seed,
            on_progress=on_progress,
        )
        pbar.close()
        logger.info(f"Saved {len(clips)} clips to {output_dir}")
    elif args.type == "3d":
        from argus.datagen.synth3d import generate_dataset
        logger.info(f"Generating {args.num_clips} 3D Blender clips...")

        pbar = tqdm(total=args.num_clips, desc="Generating 3D clips", unit="clip")

        def on_progress(completed: int, total: int) -> None:
            pbar.n = completed
            pbar.refresh()

        clips = generate_dataset(
            num_clips=args.num_clips,
            clip_length=args.clip_length,
            image_size=args.image_size,
            frames_per_move=args.frames_per_move,
            output_dir=str(output_dir),
            seed=args.seed,
            on_progress=on_progress,
        )
        pbar.close()
        logger.info(f"Saved {len(clips)} 3D clips to {output_dir}")


if __name__ == "__main__":
    main()
