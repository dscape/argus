#!/usr/bin/env python3
"""Evaluation entry point for Argus."""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from argus.data.collate import argus_collate_fn
from argus.data.dataset import ArgusInMemoryDataset
from argus.datagen.synth import generate_dataset
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    logger.info(f"Generating {args.num_clips} synthetic evaluation clips...")
    clips = generate_dataset(
        num_clips=args.num_clips,
        clip_length=16,
        image_size=224,
        seed=12345,
    )
    dataset = ArgusInMemoryDataset(clips=clips, clip_length=16)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=argus_collate_fn,
        num_workers=0,
    )

    model = ArgusModel(use_detector=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model=model, device=device)
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
