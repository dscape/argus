"""Train the screening classifier head on cached DINOv2 features.

Workflow:
1. extract_and_cache_features() — pre-compute embeddings (slow, one-time)
2. train() — train the MLP head on cached features (fast, iterative)
"""

import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pipeline.db.connection import get_conn
from pipeline.screen.ai_classifier import (
    CLASS_NAMES,
    EMBED_DIM,
    MODEL_CODE_VERSION,
    NUM_CLASSES,
    NUM_FRAMES,
    ScreeningClassifier,
    ScreeningFeatureExtractor,
)
from pipeline.screen.frame_fetcher import fetch_youtube_frames, is_vertical_video

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "ai_screening_cache")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "data", "ai_screening_checkpoints")
WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "weights", "ai_screening")


def _get_labelled_videos() -> list[tuple[str, str, int]]:
    """Fetch labelled videos from DB.

    Returns list of (video_id, channel_handle, class_index).
    Class mapping:
        0 = overlay  (approved, unless layout_type explicitly 'otb_only')
        1 = otb_only (approved + layout_type='otb_only')
        2 = reject   (rejected)

    Note: layout_type was never consistently set. Approved videos without
    layout_type are implicitly overlay (they have a 2D board overlay).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT video_id, channel_handle, screening_status, layout_type
                FROM youtube_videos
                WHERE screening_status IN ('approved', 'rejected')
                ORDER BY random()
            """)
            rows = cur.fetchall()

    videos = []
    for video_id, channel, status, layout_type in rows:
        if status == "approved" and layout_type == "otb_only":
            videos.append((video_id, channel, 1))
        elif status == "approved":
            # Approved without explicit otb_only = overlay (implicit default)
            videos.append((video_id, channel, 0))
        elif status == "rejected":
            videos.append((video_id, channel, 2))
    return videos


def extract_and_cache_features(device: str = "cpu") -> int:
    """Pre-compute DINOv2 embeddings + scanner scores for all labelled videos.

    Saves each video's features as a .pt file in CACHE_DIR.
    Skips videos already cached. Returns number of newly cached videos.
    Vertical videos are marked and excluded from training.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    videos = _get_labelled_videos()
    extractor = ScreeningFeatureExtractor(device=device)

    cached = 0
    skipped = 0
    failed = 0
    vertical = 0

    for i, (video_id, channel, label) in enumerate(videos):
        cache_path = os.path.join(CACHE_DIR, f"{video_id}.pt")
        if os.path.exists(cache_path):
            skipped += 1
            continue

        # Fetch frames first to check for vertical video
        frames = fetch_youtube_frames(video_id)
        if not frames:
            failed += 1
            logger.warning(f"[{i+1}/{len(videos)}] Failed to fetch frames for {video_id}")
            continue

        if is_vertical_video(frames):
            vertical += 1
            # Save marker so we skip on future runs without re-fetching
            torch.save({"vertical": True, "label": label, "channel": channel}, cache_path)
            continue

        features = extractor.extract_features(video_id)
        if features is None:
            failed += 1
            logger.warning(f"[{i+1}/{len(videos)}] Failed to extract features for {video_id}")
            continue

        features["label"] = label
        features["channel"] = channel
        torch.save(features, cache_path)
        cached += 1

        if (cached + skipped) % 50 == 0:
            print(f"  Progress: {cached + skipped + failed + vertical}/{len(videos)} "
                  f"(cached={cached}, skipped={skipped}, failed={failed}, vertical={vertical})")

    print(f"\nFeature extraction complete: {cached} new, {skipped} cached, "
          f"{failed} failed, {vertical} vertical (excluded)")
    return cached


def _load_cached_features() -> tuple[list[dict], list[int]]:
    """Load all cached features and their labels."""
    videos = _get_labelled_videos()
    video_labels = {vid: label for vid, _, label in videos}

    features = []
    labels = []

    for vid, _, expected_label in videos:
        cache_path = os.path.join(CACHE_DIR, f"{vid}.pt")
        if not os.path.exists(cache_path):
            continue
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        # Skip vertical videos
        if data.get("vertical"):
            continue
        features.append(data)
        labels.append(expected_label)

    return features, labels


def build_datasets(val_fraction: float = 0.15) -> tuple[TensorDataset, TensorDataset]:
    """Build train/val TensorDatasets from cached features.

    Splits by channel to prevent data leakage.
    """
    videos = _get_labelled_videos()
    video_map = {vid: (channel, label) for vid, channel, label in videos}

    # Group videos by channel
    channel_videos: dict[str, list[str]] = {}
    for vid, channel, _ in videos:
        channel_videos.setdefault(channel, []).append(vid)

    # Assign channels to val set until we reach target fraction
    channels = sorted(channel_videos.keys(), key=lambda c: len(channel_videos[c]))
    total = len(videos)
    val_target = int(total * val_fraction)

    val_channels = set()
    val_count = 0
    for ch in channels:
        if val_count >= val_target:
            break
        val_channels.add(ch)
        val_count += len(channel_videos[ch])

    # Load features and split
    train_emb, train_scan, train_otb, train_labels = [], [], [], []
    val_emb, val_scan, val_otb, val_labels = [], [], [], []

    for vid, channel, label in videos:
        cache_path = os.path.join(CACHE_DIR, f"{vid}.pt")
        if not os.path.exists(cache_path):
            continue

        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if data.get("vertical"):
            continue
        emb = data["embeddings"]           # (4, 768)
        scan = data["scanner_scores"]      # (4,)
        otb = data["otb_scores"]           # (4,)

        if channel in val_channels:
            val_emb.append(emb)
            val_scan.append(scan)
            val_otb.append(otb)
            val_labels.append(label)
        else:
            train_emb.append(emb)
            train_scan.append(scan)
            train_otb.append(otb)
            train_labels.append(label)

    def make_dataset(embs, scans, otbs, labels):
        if not embs:
            return TensorDataset(
                torch.zeros(0, NUM_FRAMES, EMBED_DIM),
                torch.zeros(0, NUM_FRAMES),
                torch.zeros(0, NUM_FRAMES),
                torch.zeros(0, dtype=torch.long),
            )
        return TensorDataset(
            torch.stack(embs),
            torch.stack(scans),
            torch.stack(otbs),
            torch.tensor(labels, dtype=torch.long),
        )

    train_ds = make_dataset(train_emb, train_scan, train_otb, train_labels)
    val_ds = make_dataset(val_emb, val_scan, val_otb, val_labels)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")
    print(f"  Val channels: {val_channels}")

    # Print class distribution
    for split_name, labels in [("Train", train_labels), ("Val", val_labels)]:
        counts = [0] * NUM_CLASSES
        for l in labels:
            counts[l] += 1
        dist = ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(NUM_CLASSES))
        print(f"  {split_name}: {dist}")

    return train_ds, val_ds


def train(
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> str:
    """Train the classifier head on cached features. Returns checkpoint path."""
    train_ds, val_ds = build_datasets()

    if len(train_ds) == 0:
        print("No training data. Run ai-extract first.")
        return ""

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None

    model = ScreeningClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for emb, scan, otb, labels in train_loader:
            emb, scan, otb, labels = emb.to(device), scan.to(device), otb.to(device), labels.to(device)
            logits = model(emb, scan, otb)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0

        # Validate
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for emb, scan, otb, labels in val_loader:
                    emb, scan, otb, labels = emb.to(device), scan.to(device), otb.to(device), labels.to(device)
                    logits = model(emb, scan, otb)
                    val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"loss={total_loss/total:.4f}  "
                f"train_acc={train_acc:.3f}  "
                f"val_acc={val_acc:.3f}"
            )

    # Save final model too
    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    torch.save(model.state_dict(), final_path)

    # Save metadata with version tracking
    import json
    from datetime import datetime, timezone

    metadata_path = os.path.join(CHECKPOINT_DIR, "metadata.json")
    revision = 1
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            old_meta = json.load(f)
        revision = old_meta.get("revision", 0) + 1

    metadata = {
        "code_version": MODEL_CODE_VERSION,
        "revision": revision,
        "version": f"{MODEL_CODE_VERSION}r{revision}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": epochs,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save to weights/ with versioned filename (committed to repo)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    version_str = metadata["version"]
    versioned_path = os.path.join(WEIGHTS_DIR, f"{version_str}.pt")
    torch.save(model.state_dict(), versioned_path)
    # Always keep a "best.pt" symlink/copy in weights/ for easy loading
    import shutil
    weights_best = os.path.join(WEIGHTS_DIR, "best.pt")
    shutil.copy2(best_path, weights_best)
    weights_meta = os.path.join(WEIGHTS_DIR, "metadata.json")
    shutil.copy2(metadata_path, weights_meta)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"  Model version: {metadata['version']}")
    print(f"  Weights: {versioned_path}")
    print(f"  Best checkpoint: {best_path}")

    return best_path
