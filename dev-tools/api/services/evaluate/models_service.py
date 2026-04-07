"""Service layer for model inspection and evaluation tracking."""

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.overlay.scanner import detect_overlay_in_frame
from pipeline.screen.dual_region_detector import detect_otb_region
from pipeline.screen.frame_fetcher import fetch_youtube_frames, is_vertical_video
from pipeline.screen.title_filter import score_title

logger = logging.getLogger(__name__)


# ── AI Screening Inspection ─────────────────────────────────


def _get_checkpoint_dir() -> str:
    """Resolve checkpoint dir — prefers weights/ (committed), falls back to data/ (ephemeral)."""
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    weights_dir = os.path.join(_root, "weights", "screening")
    if os.path.exists(os.path.join(weights_dir, "best.pt")):
        return weights_dir
    return os.path.join(_root, "data", "screening", "checkpoints")


def _get_model_version() -> str | None:
    """Read model version from metadata.json saved during training."""
    import json
    metadata_path = os.path.join(_get_checkpoint_dir(), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f).get("version")
    return None


def _frame_to_base64(frame: np.ndarray, max_width: int = 640) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def inspect_ai_screening(video_id: str) -> dict | None:
    """Inspect AI screening for a single video — 3 frames + per-frame scores + prediction."""
    import torch
    import torch.nn.functional as F

    # Fetch video metadata
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, screening_status, layout_type FROM youtube_videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            title, human_status, human_layout = row

    # Compute title score on the fly
    _, title_score = score_title(title)

    # Fetch 25/50/75% frames
    frames = fetch_youtube_frames(video_id)
    if not frames:
        return {"video_id": video_id, "title": title, "error": "Could not fetch thumbnails"}

    # Detect vertical video
    vertical = is_vertical_video(frames)

    # Scan all frames in parallel (numpy/OpenCV release the GIL)
    def _scan_frame(frame_bgr: np.ndarray, label: str) -> dict:
        h, w = frame_bgr.shape[:2]
        det = detect_overlay_in_frame(frame_bgr)
        overlay_score = det.score if det.found else 0.0
        otb_score = 0.0
        if det.found and det.bbox:
            otb_det = detect_otb_region(frame_bgr, det.bbox)
            otb_score = otb_det.confidence
        return {
            "label": label,
            "width": w,
            "height": h,
            "image_base64": _frame_to_base64(frame_bgr),
            "overlay_score": round(overlay_score, 3),
            "otb_score": round(otb_score, 3),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_scan_frame, f, l) for f, l in frames]
        frame_results = [fut.result() for fut in futures]

    # Pre-computed scores to avoid re-running the scanner in the classifier
    precomputed_scores = [
        (fr["overlay_score"], fr["otb_score"]) for fr in frame_results
    ]

    # Auto-reject vertical videos without running the classifier
    prediction = None
    if vertical:
        prediction = {
            "class": "reject",
            "confidence": 1.0,
            "probabilities": {"overlay": 0.0, "otb_only": 0.0, "reject": 1.0},
        }

    checkpoint_path = os.path.join(_get_checkpoint_dir(), "best.pt")
    if prediction is None and os.path.exists(checkpoint_path):
        try:
            from pipeline.screen.ai_classifier import (
                CLASS_NAMES,
                ScreeningClassifier,
                ScreeningFeatureExtractor,
            )

            extractor = ScreeningFeatureExtractor(device="cpu")
            features = extractor.extract_features_from_frames(frames, precomputed_scores)

            if features is not None:
                model = ScreeningClassifier()
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                )
                model.eval()

                emb = features["embeddings"].unsqueeze(0)
                scan = features["scanner_scores"].unsqueeze(0)
                otb = features["otb_scores"].unsqueeze(0)

                with torch.no_grad():
                    logits = model(emb, scan, otb)
                    probs = F.softmax(logits, dim=-1).squeeze(0)

                conf, pred_idx = probs.max(dim=0)
                prediction = {
                    "class": CLASS_NAMES[pred_idx.item()],
                    "confidence": round(conf.item(), 4),
                    "probabilities": {
                        CLASS_NAMES[i]: round(probs[i].item(), 4)
                        for i in range(len(CLASS_NAMES))
                    },
                }
        except Exception as e:
            logger.warning(f"AI classifier error: {e}")

    return {
        "video_id": video_id,
        "title": title,
        "title_score": title_score,
        "vertical": vertical,
        "frames": frame_results,
        "prediction": prediction,
        "human_label": human_status,
        "human_layout_type": human_layout,
        "model_version": _get_model_version(),
    }


def inspect_ai_screening_batch(
    video_ids: list[str] | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Batch AI screening inspection. Returns results with thumbnail URLs instead of base64."""
    if video_ids is None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if status == "unscreened":
                    cur.execute(
                        "SELECT video_id FROM youtube_videos WHERE screening_status IS NULL ORDER BY published_at DESC LIMIT %s",
                        (limit,),
                    )
                elif status == "screened":
                    cur.execute(
                        "SELECT video_id FROM youtube_videos WHERE screening_status IN ('approved', 'rejected') ORDER BY random() LIMIT %s",
                        (limit,),
                    )
                else:
                    cur.execute(
                        "SELECT video_id FROM youtube_videos ORDER BY published_at DESC LIMIT %s",
                        (limit,),
                    )
                video_ids = [r[0] for r in cur.fetchall()]

    results = []
    for vid in video_ids:
        result = inspect_ai_screening(vid)
        if result:
            results.append(result)
    return results


def sample_labeled_video_ids(limit: int = 20, exclude: list[str] | None = None) -> list[str]:
    """Return random sample of video IDs from labeled (screened) videos."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if exclude:
                cur.execute(
                    "SELECT video_id FROM youtube_videos WHERE screening_status IN ('approved', 'rejected') AND video_id != ALL(%s) ORDER BY random() LIMIT %s",
                    (exclude, limit),
                )
            else:
                cur.execute(
                    "SELECT video_id FROM youtube_videos WHERE screening_status IN ('approved', 'rejected') ORDER BY random() LIMIT %s",
                    (limit,),
                )
            return [r[0] for r in cur.fetchall()]


def save_screening_eval(accuracy: float, sample_size: int, per_class: dict, model_version: str | None) -> dict:
    """Save a screening evaluation result to the model_evaluations table."""
    import json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_evaluations
                    (model_name, sample_size, accuracy, per_class, notes)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, evaluated_at
                """,
                (
                    "screening",
                    sample_size,
                    round(accuracy, 4),
                    json.dumps(per_class),
                    model_version,
                ),
            )
            eval_id, evaluated_at = cur.fetchone()
            conn.commit()

    return {
        "id": eval_id,
        "evaluated_at": evaluated_at.isoformat(),
        "accuracy": round(accuracy, 4),
        "model_version": model_version,
    }


# ── Screening Sessions ──────────────────────────────────────


def create_screening_session(
    results: list[dict],
    accuracy: float,
    sample_size: int,
    per_class: dict,
    model_version: str | None = None,
    pin_state: dict | None = None,
    evaluation_id: int | None = None,
) -> dict:
    """Create a shareable screening session. Strips base64 frame images before storing."""
    import json
    from uuid import uuid4

    session_id = uuid4().hex[:12]

    # Strip image_base64 from frames to keep storage lean
    light_results = []
    for r in results:
        lr = {k: v for k, v in r.items() if k != "frames"}
        if "frames" in r:
            lr["frames"] = [
                {k: v for k, v in f.items() if k != "image_base64"}
                for f in r["frames"]
            ]
        light_results.append(lr)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO screening_sessions
                    (id, sample_size, model_version, accuracy, per_class, results, pin_state, evaluation_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING created_at
                """,
                (
                    session_id,
                    sample_size,
                    model_version,
                    round(accuracy, 4) if accuracy is not None else None,
                    json.dumps(per_class) if per_class else None,
                    json.dumps(light_results),
                    json.dumps(pin_state or {}),
                    evaluation_id,
                ),
            )
            created_at = cur.fetchone()[0]
            conn.commit()

    return {"session_id": session_id, "created_at": created_at.isoformat()}


def get_screening_session(session_id: str) -> dict | None:
    """Fetch a screening session by ID."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, sample_size, model_version, accuracy,
                       per_class, results, pin_state, evaluation_id
                FROM screening_sessions WHERE id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

    return {
        "id": row[0],
        "created_at": row[1].isoformat() if row[1] else None,
        "sample_size": row[2],
        "model_version": row[3],
        "accuracy": row[4],
        "per_class": row[5],
        "results": row[6],
        "pin_state": row[7] or {},
        "evaluation_id": row[8],
    }


def update_session_pins(session_id: str, pin_state: dict) -> dict:
    """Merge pin_state updates into a screening session."""
    import json

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE screening_sessions
                SET pin_state = COALESCE(pin_state, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                RETURNING pin_state
                """,
                (json.dumps(pin_state), session_id),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": "Session not found"}
            conn.commit()

    return {"pin_state": row[0]}


def list_screening_sessions(limit: int = 20) -> list[dict]:
    """List recent screening sessions (lightweight — no results payload)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, sample_size, model_version, accuracy
                FROM screening_sessions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "created_at": r[1].isoformat() if r[1] else None,
            "sample_size": r[2],
            "model_version": r[3],
            "accuracy": r[4],
        }
        for r in rows
    ]


# ── Lightweight AI Screening for Screening Page ──────────────


def ai_screen_batch(video_ids: list[str], threshold: float = 0.90) -> list[dict]:
    """Run AI screening on a batch of videos.

    Two modes:
    1. Inline inference: if ML deps (torch/transformers) are available, runs the
       DINOv2 classifier directly.
    2. DB fallback: reads existing ai_screening_* predictions from the database
       (populated by `pipeline ai-screen` CLI command).

    In both modes, vertical detection and title scoring always run (only need
    opencv/numpy). Returns per-video: predicted_class, confidence, auto_decided,
    vertical, title_score, max_ovl_score, max_otb_score.
    """
    # Check if ML deps are available for inline inference
    has_ml_deps = True
    try:
        import torch
        import torch.nn.functional as F
        from pipeline.screen.ai_classifier import (
            CLASS_NAMES,
            ScreeningClassifier,
            ScreeningFeatureExtractor,
        )
        from pipeline.screen.ai_train import CACHE_DIR
    except ImportError:
        has_ml_deps = False

    # Fetch video metadata from DB (title + any existing AI predictions)
    video_data: dict[str, dict] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, title, ai_screening_class, ai_screening_confidence,
                       ai_screening_auto_decided
                FROM youtube_videos WHERE video_id = ANY(%s)
                """,
                (video_ids,),
            )
            for vid, title, ai_cls, ai_conf, ai_auto in cur.fetchall():
                video_data[vid] = {
                    "title": title,
                    "ai_class": ai_cls,
                    "ai_confidence": ai_conf,
                    "ai_auto": ai_auto,
                }

    # Load model if inline inference is possible
    model = None
    extractor = None
    model_version = _get_model_version()
    if has_ml_deps:
        checkpoint_path = os.path.join(_get_checkpoint_dir(), "best.pt")
        if os.path.exists(checkpoint_path):
            try:
                model = ScreeningClassifier()
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
                model.eval()
                extractor = ScreeningFeatureExtractor(device="cpu")
            except Exception as e:
                logger.warning(f"Failed to load AI model (falling back to DB): {type(e).__name__}: {e}")
                model = None

    results = []
    db_updates = []

    for video_id in video_ids:
        try:
            vdata = video_data.get(video_id, {})
            title = vdata.get("title", "")
            _, title_score = score_title(title) if title else (False, 0.0)

            # Base result dict — all fields always present for consistent contract
            base = {
                "video_id": video_id,
                "predicted_class": None,
                "confidence": 0.0,
                "auto_decided": False,
                "vertical": False,
                "title_score": round(title_score, 3),
                "max_ovl_score": 0.0,
                "max_otb_score": 0.0,
                "model_version": model_version,
                "error": None,
            }

            # Always check vertical (only needs opencv/numpy)
            frames = fetch_youtube_frames(video_id)
            if not frames:
                results.append({**base,
                    "predicted_class": "reject",
                    "confidence": 1.0,
                    "auto_decided": True,
                })
                db_updates.append({
                    "video_id": video_id,
                    "predicted_class": "reject",
                    "confidence": 1.0,
                    "auto_decided": True,
                })
                continue

            vertical = is_vertical_video(frames)

            if vertical:
                results.append({**base,
                    "predicted_class": "reject",
                    "confidence": 1.0,
                    "auto_decided": True,
                    "vertical": True,
                })
                db_updates.append({
                    "video_id": video_id,
                    "predicted_class": "reject",
                    "confidence": 1.0,
                    "auto_decided": True,
                })
                continue

            # Compute per-frame heuristic scores (only needs opencv/numpy)
            ovl_scores = []
            otb_scores = []
            for frame_bgr, _ in frames:
                det = detect_overlay_in_frame(frame_bgr)
                ovl_scores.append(det.score if det.found else 0.0)
                otb_score = 0.0
                if det.found and det.bbox:
                    otb_det = detect_otb_region(frame_bgr, det.bbox)
                    otb_score = otb_det.confidence
                otb_scores.append(otb_score)

            max_ovl = round(max(ovl_scores), 3) if ovl_scores else 0.0
            max_otb = round(max(otb_scores), 3) if otb_scores else 0.0

            # Try inline inference first, fall back to DB predictions
            predicted_class = None
            confidence = 0.0

            if model is not None and extractor is not None:
                # Inline inference
                cache_path = os.path.join(CACHE_DIR, f"{video_id}.pt")
                if os.path.exists(cache_path):
                    data = torch.load(cache_path, map_location="cpu", weights_only=True)
                else:
                    data = extractor.extract_features(video_id)
                    if data is not None:
                        os.makedirs(CACHE_DIR, exist_ok=True)
                        torch.save(data, cache_path)

                if data is not None:
                    emb = data["embeddings"].unsqueeze(0)
                    scan = data["scanner_scores"].unsqueeze(0)
                    otb = data["otb_scores"].unsqueeze(0)

                    with torch.no_grad():
                        logits = model(emb, scan, otb)
                        probs = F.softmax(logits, dim=-1).squeeze(0)

                    conf, pred_idx = probs.max(dim=0)
                    confidence = round(conf.item(), 4)
                    predicted_class = CLASS_NAMES[pred_idx.item()]

                    # Use cached feature scores for consistency
                    max_ovl = round(float(data["scanner_scores"].max()), 3)
                    max_otb = round(float(data["otb_scores"].max()), 3)

            elif vdata.get("ai_class"):
                # DB fallback — use existing predictions from `pipeline ai-screen`
                predicted_class = vdata["ai_class"]
                confidence = round(vdata["ai_confidence"] or 0.0, 4)

            if predicted_class is None:
                # No prediction available
                error_msg = "No AI prediction — run `pipeline ai-screen` first" if not has_ml_deps else "Feature extraction failed"
                results.append({**base,
                    "max_ovl_score": max_ovl,
                    "max_otb_score": max_otb,
                    "error": error_msg,
                })
                continue

            auto_decided = confidence >= threshold
            results.append({**base,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "auto_decided": auto_decided,
                "max_ovl_score": max_ovl,
                "max_otb_score": max_otb,
            })
            db_updates.append({
                "video_id": video_id,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "auto_decided": auto_decided,
            })
        except Exception as e:
            logger.warning(f"AI screen failed for {video_id}: {type(e).__name__}: {e}")
            results.append({
                "video_id": video_id,
                "predicted_class": None,
                "confidence": 0.0,
                "auto_decided": False,
                "vertical": False,
                "title_score": 0.0,
                "max_ovl_score": 0.0,
                "max_otb_score": 0.0,
                "model_version": model_version,
                "error": f"Processing failed: {type(e).__name__}: {e}",
            })
            continue

    # Batch write to DB
    if db_updates:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    for pred in db_updates:
                        vid = pred["video_id"]
                        cls = pred["predicted_class"]
                        conf = pred["confidence"]
                        auto = pred["auto_decided"]

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
                            (cls, conf, auto, vid),
                        )

                        # For auto-decided, also update screening status
                        if auto:
                            if cls == "overlay":
                                status, layout = "approved", "overlay"
                            elif cls == "otb_only":
                                status, layout = "approved", "otb_only"
                            else:
                                status, layout = "rejected", None

                            cur.execute(
                                """
                                UPDATE youtube_videos
                                SET screening_status = %s,
                                    screening_confidence = %s,
                                    layout_type = COALESCE(%s, layout_type),
                                    screened_by = 'ai',
                                    updated_at = now()
                                WHERE video_id = %s
                                  AND screening_status IS NULL
                                """,
                                (status, conf, layout, vid),
                            )
                    conn.commit()
        except Exception as e:
            logger.error(f"DB update for AI screening failed: {type(e).__name__}: {e}")
            for r in results:
                if "error" not in r:
                    r["db_write_failed"] = True

    return results


# ── Auto-Calibration Inspection ──────────────────────────────


def inspect_auto_calibration(video_id: str) -> dict | None:
    """Inspect auto-calibration for a video — overlay detection at multiple timestamps."""
    from pipeline.overlay.auto_calibration import (
        _get_video_path,
        compute_camera_bbox,
        detect_board_orientation,
        detect_board_theme,
    )
    from pipeline.overlay.calibration import get_calibration

    video_path = _get_video_path(video_id)
    source = "local_video" if video_path else "thumbnails"

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"video_id": video_id, "error": "Cannot open video"}

        all_frames = []
        timestamps = [60, 120, 300]
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)

        for ts in timestamps:
            if ts > duration:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret:
                all_frames.append((frame, ts))
        cap.release()
    else:
        yt_frames = fetch_youtube_frames(video_id)
        all_frames = [(f, i) for i, (f, _) in enumerate(yt_frames)]

    if not all_frames:
        return {"video_id": video_id, "error": "No frames available"}

    frame_results = []
    best_detection = None
    best_frame = None

    for frame, ts in all_frames:
        det = detect_overlay_in_frame(frame)
        fh, fw = frame.shape[:2]

        # Draw bboxes on frame
        annotated = frame.copy()
        if det.found and det.bbox:
            ox, oy, ow, oh = det.bbox
            cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
            if det.seed_bbox:
                sx, sy, sw, sh = det.seed_bbox
                cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

        frame_results.append({
            "timestamp_sec": ts,
            "image_base64": _frame_to_base64(annotated),
            "overlay_detection": {
                "found": det.found,
                "bbox": list(det.bbox) if det.bbox else None,
                "seed_bbox": list(det.seed_bbox) if det.seed_bbox else None,
                "score": round(det.score, 3),
            },
        })

        if det.found:
            area = det.bbox[2] * det.bbox[3] if det.bbox else 0
            best_area = best_detection.bbox[2] * best_detection.bbox[3] if best_detection and best_detection.bbox else 0
            if area > best_area:
                best_detection = det
                best_frame = frame

    # Compute proposal from best detection
    proposal = None
    proposal_frame_b64 = None
    camera_heatmap_b64 = None
    if best_detection and best_frame is not None:
        fh, fw = best_frame.shape[:2]
        overlay_crop = best_frame[
            best_detection.bbox[1] : best_detection.bbox[1] + best_detection.bbox[3],
            best_detection.bbox[0] : best_detection.bbox[0] + best_detection.bbox[2],
        ]

        theme, theme_conf = detect_board_theme(overlay_crop)
        flipped, orient_conf = detect_board_orientation(overlay_crop, theme)
        camera = compute_camera_bbox(
            [f for f, _ in all_frames], best_detection.bbox
        )

        proposal = {
            "overlay": list(best_detection.bbox),
            "camera": list(camera),
            "theme": theme,
            "theme_confidence": round(theme_conf, 3),
            "board_flipped": flipped,
            "orientation_confidence": round(orient_conf, 3),
        }

        # Draw proposal overlay + camera bboxes on best frame for visual verification
        proposal_annotated = best_frame.copy()
        ox, oy, ow, oh = best_detection.bbox
        cv2.rectangle(proposal_annotated, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
        cv2.putText(proposal_annotated, "Overlay", (ox + 6, oy + 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cx, cy, cw, ch = camera
        cv2.rectangle(proposal_annotated, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 3)
        cv2.putText(proposal_annotated, "Camera", (cx + 6, cy + 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        proposal_frame_b64 = _frame_to_base64(proposal_annotated)

        # Generate camera motion heatmap
        if len(all_frames) >= 2:
            target_h, target_w = all_frames[0][0].shape[:2]
            f1 = cv2.cvtColor(all_frames[0][0], cv2.COLOR_BGR2GRAY).astype(np.float32)
            f2_raw = cv2.cvtColor(all_frames[-1][0], cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Resize f2 to match f1 if resolutions differ (e.g. mixed-res thumbnails)
            if f2_raw.shape != f1.shape:
                f2_raw = cv2.resize(f2_raw, (target_w, target_h)).astype(np.float32)
            diff = np.abs(f1 - f2_raw)
            # Zero out overlay
            ox, oy, ow, oh = best_detection.bbox
            diff[oy : oy + oh, ox : ox + ow] = 0
            # Normalize and colorize
            diff_norm = np.clip(diff / 30.0 * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
            camera_heatmap_b64 = _frame_to_base64(heatmap)

    # Mark which frame was used for the proposal
    best_bbox_list = list(best_detection.bbox) if best_detection and best_detection.bbox else None
    for fr in frame_results:
        fr["used_for_proposal"] = (
            fr["overlay_detection"]["bbox"] == best_bbox_list
            if best_bbox_list else False
        )

    # Get saved calibration for comparison
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT channel_handle FROM youtube_videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()

    saved_cal = None
    if row:
        cal = get_calibration(row[0])
        if cal:
            saved_cal = {
                "overlay": list(cal.overlay),
                "camera": list(cal.camera),
                "ref_resolution": list(cal.ref_resolution),
                "board_flipped": cal.board_flipped,
                "board_theme": cal.board_theme,
            }

    return {
        "video_id": video_id,
        "source": source,
        "frames": frame_results,
        "proposal": proposal,
        "proposal_frame_base64": proposal_frame_b64,
        "saved_calibration": saved_cal,
        "camera_motion_heatmap_base64": camera_heatmap_b64,
    }


# ── Hard Cut Detection Inspection ────────────────────────────


def inspect_hard_cuts(video_id: str, sample_fps: float = 2.0) -> dict | None:
    """Inspect hard cut detection on a video with calibration."""
    from pipeline.overlay.auto_calibration import _get_video_path
    from pipeline.overlay.calibration import get_calibration
    from pipeline.overlay.overlay_move_detector import (
        count_fen_differences,
        detect_moves,
    )
    from pipeline.overlay.grid_detector import detect_grid
    from pipeline.overlay.piece_classifier import read_fen_with_grid

    # Get channel for calibration
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT channel_handle FROM youtube_videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    channel = row[0]
    cal = get_calibration(channel)
    if cal is None:
        return {"video_id": video_id, "error": f"No calibration for {channel}"}

    video_path = _get_video_path(video_id)
    if not video_path:
        return {"video_id": video_id, "error": "Video not downloaded"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video_id": video_id, "error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scaled_cal = cal.scale_to_resolution(width, height)

    frame_skip = max(1, int(fps / sample_fps))
    fens = []
    frame_indices = []

    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        ox, oy, ow, oh = scaled_cal.overlay
        overlay_crop = frame[oy : oy + oh, ox : ox + ow]
        grid = detect_grid(overlay_crop)
        fen = read_fen_with_grid(overlay_crop, grid) if grid else None

        fens.append(fen)
        frame_indices.append(current_frame)
        current_frame += frame_skip

    cap.release()

    if len(fens) < 2:
        return {"video_id": video_id, "error": "Too few frames readable"}

    segments = detect_moves(fens, frame_indices, fps=fps, start_time=0.0)

    # Build response
    seg_results = []
    for seg in segments:
        moves = []
        for m in seg.moves:
            squares_changed = 0
            if m.fen_before and m.fen_after:
                squares_changed = count_fen_differences(m.fen_before, m.fen_after)
            moves.append({
                "move_san": m.move_san,
                "move_uci": m.move_uci,
                "frame_idx": m.frame_idx,
                "timestamp_sec": round(m.timestamp_seconds, 2),
                "confidence": round(m.confidence, 3),
                "fen_before": m.fen_before,
                "fen_after": m.fen_after,
                "squares_changed": squares_changed,
            })

        seg_results.append({
            "game_index": len(seg_results),
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "start_time": round(seg.start_time, 2),
            "end_time": round(seg.end_time, 2),
            "num_moves": seg.num_moves,
            "pgn_moves": seg.pgn_moves,
            "moves": moves,
        })

    readable = sum(1 for f in fens if f is not None)
    all_confidences = [m["confidence"] for s in seg_results for m in s["moves"]]
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return {
        "video_id": video_id,
        "total_frames_sampled": len(fens),
        "readable_fens": readable,
        "segments": seg_results,
        "total_segments": len(seg_results),
        "total_moves": sum(s["num_moves"] for s in seg_results),
        "avg_confidence": round(avg_conf, 3),
    }


# ── Model Evaluation History ─────────────────────────────────


def run_evaluation(model_name: str, sample_size: int = 500, notes: str | None = None) -> dict:
    """Run a standardized evaluation and store results."""
    if model_name == "screening":
        return _eval_ai_screening(sample_size, notes=notes)
    else:
        return {"error": f"Unknown model: {model_name}"}


def _eval_ai_screening(sample_size: int, notes: str | None = None) -> dict:
    """Evaluate AI screening classifier on a channel-stratified sample."""
    import torch
    import torch.nn.functional as F

    from pipeline.screen.ai_classifier import (
        CLASS_NAMES,
        NUM_CLASSES,
        ScreeningClassifier,
    )
    from pipeline.screen.ai_train import CACHE_DIR, CHECKPOINT_DIR, _get_labelled_videos

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if not os.path.exists(checkpoint_path):
        return {"error": "No checkpoint found. Run ai-train first."}

    # Get labelled videos and sample
    videos = _get_labelled_videos()
    if len(videos) < sample_size:
        sample_size = len(videos)

    # Channel-stratified sampling
    import random
    random.seed(42)

    by_channel: dict[str, list] = {}
    for vid, ch, label in videos:
        by_channel.setdefault(ch, []).append((vid, label))

    # Sample proportionally from each channel
    sampled = []
    channels = list(by_channel.keys())
    random.shuffle(channels)
    per_channel = max(1, sample_size // len(channels))
    for ch in channels:
        ch_vids = by_channel[ch]
        random.shuffle(ch_vids)
        sampled.extend(ch_vids[:per_channel])
        if len(sampled) >= sample_size:
            break

    sampled = sampled[:sample_size]

    # Load model
    model = ScreeningClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    # Predict
    all_preds = []
    all_labels = []

    for vid, label in sampled:
        cache_path = os.path.join(CACHE_DIR, f"{vid}.pt")
        if not os.path.exists(cache_path):
            continue

        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        emb = data["embeddings"].unsqueeze(0)
        scan = data["scanner_scores"].unsqueeze(0)
        otb = data["otb_scores"].unsqueeze(0)

        with torch.no_grad():
            logits = model(emb, scan, otb)
            probs = F.softmax(logits, dim=-1)
            _, pred = probs.max(dim=-1)

        all_preds.append(pred.item())
        all_labels.append(label)

    # Compute metrics
    per_class = {}
    total_correct = 0
    for cls_idx in range(NUM_CLASSES):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l == cls_idx)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l != cls_idx)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != cls_idx and l == cls_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_correct += tp

        per_class[CLASS_NAMES[cls_idx]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    accuracy = total_correct / len(all_labels) if all_labels else 0.0
    avg_p = sum(c["precision"] for c in per_class.values()) / NUM_CLASSES
    avg_r = sum(c["recall"] for c in per_class.values()) / NUM_CLASSES
    avg_f1 = sum(c["f1"] for c in per_class.values()) / NUM_CLASSES

    # Store in DB
    import json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_evaluations
                    (model_name, sample_size, accuracy, precision_avg, recall_avg, f1_avg, per_class, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, evaluated_at
                """,
                (
                    "screening",
                    len(all_labels),
                    round(accuracy, 4),
                    round(avg_p, 4),
                    round(avg_r, 4),
                    round(avg_f1, 4),
                    json.dumps(per_class),
                    notes,
                ),
            )
            eval_id, evaluated_at = cur.fetchone()
            conn.commit()

    return {
        "id": eval_id,
        "model_name": "screening",
        "evaluated_at": evaluated_at.isoformat(),
        "sample_size": len(all_labels),
        "accuracy": round(accuracy, 4),
        "precision_avg": round(avg_p, 4),
        "recall_avg": round(avg_r, 4),
        "f1_avg": round(avg_f1, 4),
        "per_class": per_class,
    }


def get_evaluation_history(model_name: str | None = None) -> list[dict]:
    """Fetch evaluation history from DB."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if model_name:
                cur.execute(
                    """
                    SELECT id, model_name, evaluated_at, sample_size, accuracy,
                           precision_avg, recall_avg, f1_avg, per_class, threshold, auto_rate, notes
                    FROM model_evaluations
                    WHERE model_name = %s
                    ORDER BY evaluated_at DESC
                    LIMIT 50
                    """,
                    (model_name,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, model_name, evaluated_at, sample_size, accuracy,
                           precision_avg, recall_avg, f1_avg, per_class, threshold, auto_rate, notes
                    FROM model_evaluations
                    ORDER BY evaluated_at DESC
                    LIMIT 50
                    """,
                )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "model_name": r[1],
            "evaluated_at": r[2].isoformat() if r[2] else None,
            "sample_size": r[3],
            "accuracy": r[4],
            "precision_avg": r[5],
            "recall_avg": r[6],
            "f1_avg": r[7],
            "per_class": r[8],
            "threshold": r[9],
            "auto_rate": r[10],
            "notes": r[11],
        }
        for r in rows
    ]
