"""Run AI screening predictions and apply auto-decisions to the database.

High-confidence predictions are written directly to the DB. Low-confidence
ones are left for manual review.
"""

import logging
import os

import torch
import torch.nn.functional as F

from pipeline.db.connection import get_conn
from pipeline.screen.ai_classifier import (
    CLASS_NAMES,
    ScreeningClassifier,
    ScreeningFeatureExtractor,
)
from pipeline.screen.ai_train import CACHE_DIR, CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def predict_batch(
    video_ids: list[str],
    checkpoint_path: str | None = None,
    threshold: float = 0.85,
    device: str = "cpu",
) -> list[dict]:
    """Run AI screening on a batch of videos.

    Returns list of dicts:
        video_id, predicted_class, confidence, auto_decided
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    model = ScreeningClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    extractor = ScreeningFeatureExtractor(device=device)

    results = []
    for i, video_id in enumerate(video_ids):
        # Try cache first
        cache_path = os.path.join(CACHE_DIR, f"{video_id}.pt")
        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            data = extractor.extract_features(video_id)
            if data is not None:
                os.makedirs(CACHE_DIR, exist_ok=True)
                torch.save(data, cache_path)

        if data is None:
            results.append({
                "video_id": video_id,
                "predicted_class": None,
                "confidence": 0.0,
                "auto_decided": False,
                "error": "Could not fetch thumbnails",
            })
            continue

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

        results.append({
            "video_id": video_id,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "auto_decided": auto_decided,
        })

        if (i + 1) % 25 == 0:
            print(f"  Predicted {i + 1}/{len(video_ids)}")

    return results


def apply_predictions(predictions: list[dict]) -> dict:
    """Write auto-decided predictions to the database.

    For auto-decided videos:
    - overlay → screening_status='approved', layout_type='overlay'
    - otb_only → screening_status='approved', layout_type='otb_only'
    - reject → screening_status='rejected'

    Also stores AI metadata in ai_screening_* columns.

    Returns summary counts.
    """
    auto_count = 0
    manual_count = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for pred in predictions:
                video_id = pred["video_id"]
                predicted_class = pred.get("predicted_class")
                confidence = pred.get("confidence", 0.0)
                auto_decided = pred.get("auto_decided", False)

                if predicted_class is None:
                    continue

                # Always write AI metadata
                cur.execute(
                    """
                    UPDATE youtube_videos
                    SET ai_screening_class = %s,
                        ai_screening_confidence = %s,
                        ai_screening_auto_decided = %s,
                        updated_at = now()
                    WHERE video_id = %s
                    """,
                    (predicted_class, confidence, auto_decided, video_id),
                )

                # For auto-decided, also update the screening status
                if auto_decided:
                    if predicted_class == "overlay":
                        status, layout = "approved", "overlay"
                    elif predicted_class == "otb_only":
                        status, layout = "approved", "otb_only"
                    else:
                        status, layout = "rejected", None

                    cur.execute(
                        """
                        UPDATE youtube_videos
                        SET screening_status = %s,
                            screening_confidence = %s,
                            layout_type = COALESCE(%s, layout_type),
                            updated_at = now()
                        WHERE video_id = %s
                          AND screening_status IS NULL
                        """,
                        (status, confidence, layout, video_id),
                    )
                    auto_count += 1
                else:
                    manual_count += 1

            conn.commit()

    return {"auto_decided": auto_count, "manual_review": manual_count}
