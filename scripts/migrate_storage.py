#!/usr/bin/env python3
"""Migrate video assets to the new unified folder structure.

Moves:
  data/screening/dataset/frames/{vid}/  → data/videos/{vid}/lores/
  data/overlay/dataset/frames/{vid}/    → data/videos/{vid}/{fullres|hires}/
  data/videos/{channel}/{vid}.mp4       → data/videos/{vid}/{vid}.mp4
  ground_truth.json                     → data/videos/ground_truth.json

Usage:
  python scripts/migrate_storage.py                  # full migration
  python scripts/migrate_storage.py --dry-run        # preview only
  python scripts/migrate_storage.py --skip-screening # skip lores frames
"""

import argparse
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
SCREENING_DIR = PROJECT_ROOT / "data" / "screening" / "dataset" / "frames"
OVERLAY_DIR = PROJECT_ROOT / "data" / "overlay" / "dataset" / "frames"
TORCH_CACHE = PROJECT_ROOT / "data" / "screening" / "dataset" / "torch"

FRAME_LABELS = {"25pct", "50pct", "75pct"}


def migrate_screening(dry_run: bool) -> int:
    """Move screening frames → data/videos/{vid}/lores/."""
    if not SCREENING_DIR.exists():
        print("  Screening dir not found, skipping")
        return 0

    dirs = sorted(SCREENING_DIR.iterdir())
    moved = 0
    for video_dir in dirs:
        if not video_dir.is_dir():
            continue
        vid = video_dir.name
        dest_dir = VIDEOS_DIR / vid / "lores"

        for jpg in video_dir.glob("*.jpg"):
            label = jpg.stem
            if label not in FRAME_LABELS:
                continue  # skip thumb.jpg, thumb_hires.jpg, etc.
            dest = dest_dir / jpg.name
            if dest.exists():
                continue
            if dry_run:
                print(f"  [dry-run] {jpg} → {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(jpg), str(dest))
            moved += 1

        # Remove empty source dir
        if not dry_run and video_dir.exists() and not any(video_dir.iterdir()):
            video_dir.rmdir()

    return moved


def migrate_overlay(dry_run: bool) -> tuple[int, int]:
    """Move overlay frames → fullres/ or hires/ based on dimensions."""
    if not OVERLAY_DIR.exists():
        print("  Overlay dir not found, skipping")
        return 0, 0

    fullres_count = 0
    hires_count = 0

    for video_dir in sorted(OVERLAY_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        vid = video_dir.name

        for jpg in video_dir.glob("*.jpg"):
            label = jpg.stem
            if label not in FRAME_LABELS:
                continue

            # Determine tier by image width
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            w = img.shape[1]
            tier = "fullres" if w >= 1920 else "hires"

            dest = VIDEOS_DIR / vid / tier / jpg.name
            if dest.exists():
                continue
            if dry_run:
                print(f"  [dry-run] {jpg} → {dest} ({w}px → {tier})")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(jpg), str(dest))

            if tier == "fullres":
                fullres_count += 1
            else:
                hires_count += 1

        # Remove empty source dir
        if not dry_run and video_dir.exists() and not any(video_dir.iterdir()):
            video_dir.rmdir()

    return fullres_count, hires_count


def migrate_videos(dry_run: bool) -> int:
    """Move data/videos/{channel}/{vid}.mp4 → data/videos/{vid}/{vid}.mp4."""
    moved = 0
    if not VIDEOS_DIR.exists():
        return 0

    for channel_dir in sorted(VIDEOS_DIR.iterdir()):
        if not channel_dir.is_dir():
            continue
        # Skip directories that look like video IDs (already migrated)
        # Channel handles are typically longer and contain letters
        # Video IDs are 11 chars with alphanumeric + dash/underscore
        for mp4 in channel_dir.glob("*.mp4"):
            vid = mp4.stem
            dest = VIDEOS_DIR / vid / f"{vid}.mp4"
            if dest.exists():
                continue
            if dry_run:
                print(f"  [dry-run] {mp4} → {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(mp4), str(dest))
            moved += 1

        # Remove empty channel dir
        if not dry_run and channel_dir.exists() and not any(channel_dir.iterdir()):
            channel_dir.rmdir()

    return moved


def migrate_ground_truth(dry_run: bool) -> bool:
    """Copy ground_truth.json to data/videos/."""
    src = OVERLAY_DIR / "ground_truth.json"
    dest = VIDEOS_DIR / "ground_truth.json"
    if not src.exists():
        print("  ground_truth.json not found, skipping")
        return False
    if dest.exists():
        print("  ground_truth.json already migrated")
        return False
    if dry_run:
        print(f"  [dry-run] {src} → {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
    return True


def invalidate_torch_cache(dry_run: bool) -> int:
    """Delete cached features (computed with 4 frames, now need 3)."""
    if not TORCH_CACHE.exists():
        return 0
    count = 0
    for pt_file in TORCH_CACHE.glob("*.pt"):
        if dry_run:
            print(f"  [dry-run] delete {pt_file}")
        else:
            pt_file.unlink()
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate video assets to new layout")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-screening", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true", help="Skip overlay frame migration")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video file migration")
    parser.add_argument("--skip-cache", action="store_true", help="Skip torch cache invalidation")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no files will be moved) ===\n")

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Screening frames
    if not args.skip_screening:
        print("Step 1: Migrating screening frames (lores)...")
        n = migrate_screening(args.dry_run)
        print(f"  {n} frames {'would be ' if args.dry_run else ''}moved\n")

    # Step 2: Overlay frames
    if not args.skip_overlay:
        print("Step 2: Migrating overlay frames (fullres/hires)...")
        fullres, hires = migrate_overlay(args.dry_run)
        verb = "would be moved" if args.dry_run else "moved"
        print(f"  {fullres} fullres + {hires} hires frames {verb}\n")

    # Step 3: Video files
    if not args.skip_videos:
        print("Step 3: Migrating video files...")
        n = migrate_videos(args.dry_run)
        print(f"  {n} videos {'would be ' if args.dry_run else ''}moved\n")

    # Step 4: Ground truth
    print("Step 4: Migrating ground_truth.json...")
    migrate_ground_truth(args.dry_run)
    print()

    # Step 5: Invalidate torch cache (old 4-frame features)
    if not args.skip_cache:
        print("Step 5: Invalidating torch cache (4-frame features)...")
        n = invalidate_torch_cache(args.dry_run)
        verb = "would be deleted" if args.dry_run else "deleted"
        print(f"  {n} cache files {verb}\n")
        if n > 0 and not args.dry_run:
            print(
                "  NOTE: Run `python -m pipeline.cli ai-extract` to"
                " regenerate features with 3 frames,"
            )
            print("  then `python -m pipeline.cli ai-train` to retrain the screening model.\n")

    print("Done!" if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    main()
