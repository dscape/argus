"""DINOv2-based screening classifier for automating video review.

Encodes 4 YouTube thumbnails per video through a frozen DINOv2 encoder,
concatenates the pooled embeddings with per-frame overlay/OTB scanner
scores, and classifies into overlay / otb_only / reject.
"""

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn

from pipeline.overlay.scanner import detect_overlay_in_frame
from pipeline.screen.dual_region_detector import detect_otb_region
from pipeline.screen.frame_fetcher import fetch_youtube_frames

logger = logging.getLogger(__name__)

# Bump this when model architecture or feature extraction changes. Format: v{N}
# v1: initial model with 120x90 frames (broken — frames 2-4 too small for scanner)
# v2: fixed to use 480x360 hq frames, added vertical video filtering
MODEL_CODE_VERSION = "v2"

# Class labels for the 3-way screening decision.
CLASS_NAMES = ["overlay", "otb_only", "reject"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_FRAMES = 4
EMBED_DIM = 768

# DINOv2 ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


class ScreeningClassifier(nn.Module):
    """Small MLP head on top of pre-extracted DINOv2 features.

    Input:  4 * 768 DINOv2 pooled embeddings + 4 overlay scores + 4 OTB scores = 3080
    Output: 3-class logits (overlay, otb_only, reject)
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_frames: int = NUM_FRAMES,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = num_frames * embed_dim + num_frames * 2  # embeddings + scanner + otb
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,     # (B, num_frames, embed_dim)
        scanner_scores: torch.Tensor,  # (B, num_frames)
        otb_scores: torch.Tensor,      # (B, num_frames)
    ) -> torch.Tensor:
        """Returns logits of shape (B, num_classes)."""
        B = embeddings.shape[0]
        flat_emb = embeddings.reshape(B, -1)  # (B, num_frames * embed_dim)
        features = torch.cat([flat_emb, scanner_scores, otb_scores], dim=-1)
        return self.classifier(features)


_cached_encoder = None
_cached_encoder_device: str | None = None


def _get_encoder(device: str = "cpu"):
    """Return a cached frozen DINOv2 encoder, loading it only once."""
    global _cached_encoder, _cached_encoder_device
    if _cached_encoder is not None and _cached_encoder_device == device:
        return _cached_encoder
    from argus.model.vision_encoder import VisionEncoder

    encoder = VisionEncoder(frozen=True).to(torch.device(device))
    encoder.eval()
    _cached_encoder = encoder
    _cached_encoder_device = device
    return encoder


class ScreeningFeatureExtractor:
    """Extract DINOv2 embeddings + scanner scores from YouTube thumbnails."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.encoder = _get_encoder(device)

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Resize and normalize a BGR frame for DINOv2 input."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor

    def extract_features(self, video_id: str) -> dict | None:
        """Extract features for a single video from its YouTube thumbnails.

        Returns dict with:
            embeddings: Tensor[NUM_FRAMES, EMBED_DIM]
            scanner_scores: Tensor[NUM_FRAMES]
            otb_scores: Tensor[NUM_FRAMES]
        Or None if thumbnails could not be fetched.
        """
        frames = fetch_youtube_frames(video_id)
        if not frames:
            return None
        return self.extract_features_from_frames(frames)

    def extract_features_from_frames(
        self,
        frames: list[tuple[np.ndarray, str]],
        precomputed_scores: list[tuple[float, float]] | None = None,
    ) -> dict:
        """Extract features from already-fetched frames.

        Args:
            frames: List of (frame_bgr, label) tuples.
            precomputed_scores: Optional list of (overlay_score, otb_score) per frame.
                When provided, skips the scanner/OTB calls entirely.
        """
        embeddings = torch.zeros(NUM_FRAMES, EMBED_DIM)
        scanner_scores = torch.zeros(NUM_FRAMES)
        otb_scores = torch.zeros(NUM_FRAMES)

        for i in range(NUM_FRAMES):
            if i < len(frames):
                frame_bgr, _ = frames[i]

                # DINOv2 embedding
                input_tensor = self._preprocess_frame(frame_bgr).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.encoder.forward_pooled(input_tensor)  # (1, 768)
                embeddings[i] = emb.squeeze(0).cpu()

                if precomputed_scores is not None and i < len(precomputed_scores):
                    scanner_scores[i] = precomputed_scores[i][0]
                    otb_scores[i] = precomputed_scores[i][1]
                else:
                    # Overlay scanner score
                    detection = detect_overlay_in_frame(frame_bgr)
                    scanner_scores[i] = detection.score if detection.found else 0.0

                    # OTB detection score
                    if detection.found and detection.bbox:
                        otb_det = detect_otb_region(frame_bgr, detection.bbox)
                        otb_scores[i] = otb_det.confidence

        return {
            "embeddings": embeddings,
            "scanner_scores": scanner_scores,
            "otb_scores": otb_scores,
        }
