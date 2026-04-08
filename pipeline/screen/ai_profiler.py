"""Profile AI screening inference to identify performance bottlenecks.

Replays the exact screening code path for a single video with timing
around each phase: model loading, frame fetching, DINOv2 inference,
overlay scanning, OTB detection, and classifier inference.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from pipeline.overlay.scanner import fast_overlay_check
from pipeline.paths import find_frame
from pipeline.screen.ai_classifier import (
    CLASS_NAMES,
    EMBED_DIM,
    NUM_FRAMES,
    ScreeningClassifier,
    ScreeningFeatureExtractor,
)
from pipeline.screen.ai_train import CACHE_DIR, CHECKPOINT_DIR
from pipeline.screen.dual_region_detector import detect_otb_region
from pipeline.screen.frame_fetcher import fetch_youtube_frames


@dataclass
class TimingRecord:
    """A single timed phase, optionally with per-frame sub-timings."""

    phase: str
    seconds: float
    sub_phases: list["TimingRecord"] = field(default_factory=list)


class PhaseTimer:
    """Collects named timing measurements."""

    def __init__(self) -> None:
        self.records: list[TimingRecord] = []
        self._sub_records: list[TimingRecord] | None = None

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        prev = self._sub_records
        self._sub_records = []
        yield
        elapsed = time.perf_counter() - t0
        subs = self._sub_records
        self._sub_records = prev
        record = TimingRecord(name, elapsed, subs)
        if self._sub_records is not None:
            self._sub_records.append(record)
        else:
            self.records.append(record)

    @contextmanager
    def sub_phase(self, name: str):
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        if self._sub_records is not None:
            self._sub_records.append(TimingRecord(name, elapsed))


def profile_video(
    video_id: str,
    checkpoint_path: str | None = None,
    threshold: float = 0.85,
    device: str = "cpu",
    force_uncached: bool = False,
) -> tuple[list[TimingRecord], dict]:
    """Profile AI screening for a single video.

    Returns (timing_records, prediction_dict).
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    timer = PhaseTimer()

    # Phase 1: Load classifier checkpoint
    with timer.phase("checkpoint_load"):
        model = ScreeningClassifier()
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        model.eval()

    # Phase 2: Load DINOv2 encoder
    with timer.phase("dinov2_load"):
        extractor = ScreeningFeatureExtractor(device=device)

    # Check feature cache
    cache_path = os.path.join(CACHE_DIR, f"{video_id}.pt")
    use_cache = os.path.exists(cache_path) and not force_uncached

    if use_cache:
        # Cached path: just load the .pt file
        with timer.phase("cache_load"):
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if "embeddings" not in data:
            return timer.records, {
                "video_id": video_id,
                "predicted_class": None,
                "confidence": 0.0,
                "auto_decided": False,
                "error": "Cached as vertical/excluded video (no features)",
            }
    else:
        # Uncached path: full feature extraction with per-phase timing

        # Phase 3: Fetch frames
        frames_from_cache = find_frame(video_id, "lores", "25pct") is not None
        frame_source = "disk" if frames_from_cache else "network"
        with timer.phase(f"frame_fetch ({frame_source})"):
            frames = fetch_youtube_frames(video_id)

        if not frames:
            return timer.records, {
                "video_id": video_id,
                "predicted_class": None,
                "confidence": 0.0,
                "auto_decided": False,
                "error": "Could not fetch thumbnails",
            }

        embeddings = torch.zeros(NUM_FRAMES, EMBED_DIM)
        scanner_scores = torch.zeros(NUM_FRAMES)
        otb_scores = torch.zeros(NUM_FRAMES)

        # Phase 4: Preprocess frames
        preprocessed = []
        with timer.phase("preprocess"):
            for i in range(min(NUM_FRAMES, len(frames))):
                frame_bgr, _ = frames[i]
                input_tensor = extractor._preprocess_frame(frame_bgr).unsqueeze(0).to(extractor.device)
                preprocessed.append((i, frame_bgr, input_tensor))

        # Phase 5: DINOv2 inference (per-frame sub-timings)
        with timer.phase("dinov2_inference"):
            for i, frame_bgr, input_tensor in preprocessed:
                with timer.sub_phase(f"frame_{i}"):
                    with torch.no_grad():
                        emb = extractor.encoder.forward_pooled(input_tensor)
                    embeddings[i] = emb.squeeze(0).cpu()

        # Phase 6: Overlay scanning (per-frame sub-timings)
        detections = []
        with timer.phase("overlay_scan"):
            for i, frame_bgr, _ in preprocessed:
                with timer.sub_phase(f"frame_{i}"):
                    detection = fast_overlay_check(frame_bgr)
                    scanner_scores[i] = detection.score if detection.found else 0.0
                    detections.append(detection)

        # Phase 7: OTB detection (per-frame sub-timings)
        with timer.phase("otb_detect"):
            for idx, (i, frame_bgr, _) in enumerate(preprocessed):
                detection = detections[idx]
                if detection.found and detection.bbox:
                    with timer.sub_phase(f"frame_{i}"):
                        otb_det = detect_otb_region(frame_bgr, detection.bbox)
                        otb_scores[i] = otb_det.confidence

        data = {
            "embeddings": embeddings,
            "scanner_scores": scanner_scores,
            "otb_scores": otb_scores,
        }

    # Phase 8: Classifier inference
    with timer.phase("classifier"):
        emb = data["embeddings"].unsqueeze(0)
        scan = data["scanner_scores"].unsqueeze(0)
        otb = data["otb_scores"].unsqueeze(0)

        with torch.no_grad():
            logits = model(emb, scan, otb)
            probs = F.softmax(logits, dim=-1)

        conf, pred = probs.max(dim=-1)
        confidence = conf.item()
        predicted_class = CLASS_NAMES[pred.item()]
        auto_decided = confidence >= threshold

    prediction = {
        "video_id": video_id,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "auto_decided": auto_decided,
    }

    return timer.records, prediction


def format_profile(records: list[TimingRecord], prediction: dict, video_id: str) -> None:
    """Print a formatted timing breakdown to stdout."""
    total = sum(r.seconds for r in records)

    print(f"\n=== AI Screening Profile: {video_id} ===\n")
    print(f"{'Phase':<30} {'Time (s)':>10}    {'% of Total':>10}")
    print("\u2500" * 56)

    for record in records:
        pct = (record.seconds / total * 100) if total > 0 else 0
        print(f"{record.phase:<30} {record.seconds:>10.3f}    {pct:>9.1f}%")
        for sub in record.sub_phases:
            print(f"  {sub.phase:<28} {sub.seconds:>10.3f}")

    print("\u2500" * 56)
    print(f"{'TOTAL':<30} {total:>10.3f}    {'100.0%':>10}")

    # Prediction summary
    cls = prediction.get("predicted_class")
    conf = prediction.get("confidence", 0)
    auto = prediction.get("auto_decided", False)
    err = prediction.get("error")
    print()
    if err:
        print(f"Result: ERROR - {err}")
    else:
        print(f"Prediction: {cls} (confidence={conf}, auto_decided={auto})")
    print()
