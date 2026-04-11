#!/usr/bin/env python3
"""Evaluation entry point for Argus."""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from argus.data.collate import argus_collate_fn
from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset
from argus.data.transforms import ValidationTransform
from argus.datagen.synth import generate_dataset
from argus.device import resolve_device
from argus.eval.evaluator import Evaluator
from argus.model.argus_model import ArgusModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Argus model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-clips", type=int, default=500)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frames-per-move", type=int, default=4)
    parser.add_argument("--occlusion-prob", type=float, default=0.2)
    parser.add_argument("--min-moves", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=80)
    parser.add_argument("--min-elo", type=int, default=1500)
    parser.add_argument("--game-source", choices=["random", "pgn_file"], default="random")
    parser.add_argument("--pgn-path", type=str, default=None)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable synthetic render augmentations when generating eval clips",
    )
    parser.add_argument("--detect-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = resolve_device(args.device)

    transform = ValidationTransform()

    if args.data_dir:
        logger.info(f"Loading evaluation clips from {args.data_dir}...")
        dataset = ArgusDataset(
            args.data_dir,
            clip_length=args.clip_length,
            max_clips=args.max_clips,
            transform=transform,
        )
    else:
        logger.info(f"Generating {args.num_clips} synthetic evaluation clips...")
        clips = generate_dataset(
            num_clips=args.num_clips,
            clip_length=args.clip_length,
            image_size=args.image_size,
            frames_per_move=args.frames_per_move,
            augment=args.augment,
            occlusion_prob=args.occlusion_prob,
            illegal_clip_prob=0.0,
            min_moves=args.min_moves,
            max_moves=args.max_moves,
            game_source=args.game_source,
            pgn_path=args.pgn_path,
            min_elo=args.min_elo,
            seed=12345,
        )
        dataset = ArgusInMemoryDataset(
            clips=clips,
            clip_length=args.clip_length,
            transform=transform,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=argus_collate_fn,
        num_workers=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ArgusModel.from_config(checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(
        model=model,
        device=device,
        detect_threshold=args.detect_threshold,
    )
    result = evaluator.evaluate(loader)

    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"  Move Accuracy:      {result.move_accuracy:.4f}")
    logger.info(f"  Move Accuracy Top5: {result.move_accuracy_top5:.4f}")
    logger.info(f"  Detection F1:       {result.move_detection_f1:.4f}")
    logger.info(f"  Avg PGN Edit Dist:  {result.avg_pgn_edit_distance:.4f}")
    logger.info(f"  Avg Prefix Accuracy:{result.avg_prefix_accuracy:.4f}")
    logger.info(f"  Games Evaluated:    {result.num_games}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
