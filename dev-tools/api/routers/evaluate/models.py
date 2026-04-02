"""Model inspection and evaluation endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.services.evaluate import (
    calibration_eval_service,
    models_service,
    overlay_test_service,
    segmentation_eval_service,
)

router = APIRouter()


# ── Request models ──────────────────────────────────────────


class InspectRequest(BaseModel):
    video_id: str


class BatchInspectRequest(BaseModel):
    video_ids: list[str] | None = None
    status: str | None = None
    limit: int = 20


class HardCutInspectRequest(BaseModel):
    video_id: str
    sample_fps: float = 2.0


class EvaluateRequest(BaseModel):
    model_name: str
    sample_size: int = 500
    notes: str | None = None


class SaveEvalRequest(BaseModel):
    accuracy: float
    sample_size: int
    per_class: dict
    model_version: str | None = None


class CreateSessionRequest(BaseModel):
    results: list[dict]
    accuracy: float | None = None
    sample_size: int = 0
    per_class: dict | None = None
    model_version: str | None = None
    pin_state: dict | None = None
    evaluation_id: int | None = None


class UpdatePinsRequest(BaseModel):
    pin_state: dict


# ── AI Screening ────────────────────────────────────────────


@router.get("/ai-screening/sample")
async def sample_labeled_videos(limit: int = 20, exclude: str | None = None):
    """Return random sample of labeled video IDs for progressive inspection."""
    exclude_list = exclude.split(",") if exclude else None
    video_ids = await run_in_threadpool(models_service.sample_labeled_video_ids, limit, exclude_list)
    return {"video_ids": video_ids}


@router.post("/ai-screening/save-eval")
async def save_screening_eval(body: SaveEvalRequest):
    """Save a screening evaluation result."""
    result = await run_in_threadpool(
        models_service.save_screening_eval,
        body.accuracy,
        body.sample_size,
        body.per_class,
        body.model_version,
    )
    return result


@router.post("/ai-screening/inspect")
async def inspect_ai_screening(body: InspectRequest):
    """Inspect AI screening for a single video — all 4 frames + scores + prediction."""
    result = await run_in_threadpool(models_service.inspect_ai_screening, body.video_id)
    if result is None:
        raise HTTPException(404, f"Video {body.video_id} not found")
    return result


@router.post("/ai-screening/batch")
async def batch_ai_screening(body: BatchInspectRequest):
    """Batch AI screening inspection with inline thumbnails."""
    result = await run_in_threadpool(
        models_service.inspect_ai_screening_batch,
        body.video_ids,
        body.status,
        body.limit,
    )
    return {"results": result, "total": len(result)}


# ── Screening Sessions ─────────────────────────────────────


@router.post("/ai-screening/sessions")
async def create_session(body: CreateSessionRequest):
    """Create a shareable screening session."""
    result = await run_in_threadpool(
        models_service.create_screening_session,
        body.results,
        body.accuracy,
        body.sample_size,
        body.per_class,
        body.model_version,
        body.pin_state,
        body.evaluation_id,
    )
    return result


@router.get("/ai-screening/sessions")
async def list_sessions(limit: int = 20):
    """List recent screening sessions."""
    sessions = await run_in_threadpool(models_service.list_screening_sessions, limit)
    return {"sessions": sessions}


@router.get("/ai-screening/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a screening session by ID."""
    session = await run_in_threadpool(models_service.get_screening_session, session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.patch("/ai-screening/sessions/{session_id}/pins")
async def update_pins(session_id: str, body: UpdatePinsRequest):
    """Update pin state for a screening session."""
    result = await run_in_threadpool(
        models_service.update_session_pins, session_id, body.pin_state
    )
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


# ── Auto-Calibration ────────────────────────────────────────


@router.post("/auto-calibration/inspect")
async def inspect_auto_calibration(body: InspectRequest):
    """Inspect auto-calibration — overlay detection at multiple timestamps."""
    result = await run_in_threadpool(
        models_service.inspect_auto_calibration, body.video_id
    )
    if result is None:
        raise HTTPException(404, f"Video {body.video_id} not found")
    return result


# ── Hard Cut Detection ──────────────────────────────────────


@router.post("/hard-cuts/inspect")
async def inspect_hard_cuts(body: HardCutInspectRequest):
    """Inspect hard cut detection — segments, moves, confidence scores."""
    result = await run_in_threadpool(
        models_service.inspect_hard_cuts, body.video_id, body.sample_fps
    )
    if result is None:
        raise HTTPException(404, f"Video {body.video_id} not found")
    return result


# ── Evaluation History ──────────────────────────────────────


@router.post("/evaluate")
async def run_evaluation(body: EvaluateRequest):
    """Run a standardized evaluation and store results."""
    result = await run_in_threadpool(
        models_service.run_evaluation, body.model_name, body.sample_size, body.notes
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/evaluations")
async def get_evaluations(model_name: str | None = None):
    """Get evaluation history."""
    history = models_service.get_evaluation_history(model_name)
    return {"evaluations": history}


# ── Overlay Testing (Piece Classifier) ─────────────────────


class OverlayInspectRequest(BaseModel):
    filename: str


class SaveOverlayEvalRequest(BaseModel):
    accuracy: float
    sample_size: int
    piece_accuracy: float | None = None
    images_per_minute: int | None = None
    notes: str | None = None


class CreateOverlaySessionRequest(BaseModel):
    results: list[dict]
    accuracy: float | None = None
    sample_size: int = 0
    piece_accuracy: float | None = None
    pin_state: dict | None = None
    evaluation_id: int | None = None


class UpdateOverlayPinsRequest(BaseModel):
    pin_state: dict


@router.get("/overlay-test/sample")
async def sample_overlay_boards(limit: int = 20, exclude: str | None = None):
    """Return random sample of board image filenames from chess-positions test set."""
    exclude_list = exclude.split(",") if exclude else None
    filenames = await run_in_threadpool(
        overlay_test_service.sample_board_filenames, limit, exclude_list
    )
    return {"filenames": filenames}


@router.post("/overlay-test/inspect")
async def inspect_overlay_board(body: OverlayInspectRequest):
    """Inspect piece classifier accuracy on a single board image."""
    try:
        result = await run_in_threadpool(
            overlay_test_service.inspect_board, body.filename
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@router.post("/overlay-test/save-eval")
async def save_overlay_eval(body: SaveOverlayEvalRequest):
    """Save an overlay evaluation result."""
    result = await run_in_threadpool(
        overlay_test_service.save_overlay_eval,
        body.accuracy,
        body.sample_size,
        body.piece_accuracy,
        body.images_per_minute,
        body.notes,
    )
    return result


@router.post("/overlay-test/sessions")
async def create_overlay_session(body: CreateOverlaySessionRequest):
    """Create a shareable overlay test session."""
    result = await run_in_threadpool(
        overlay_test_service.create_overlay_test_session,
        body.results,
        body.accuracy,
        body.sample_size,
        body.piece_accuracy,
        body.pin_state,
        body.evaluation_id,
    )
    return result


@router.get("/overlay-test/sessions")
async def list_overlay_sessions(limit: int = 20):
    """List recent overlay test sessions."""
    sessions = await run_in_threadpool(
        overlay_test_service.list_overlay_test_sessions, limit
    )
    return {"sessions": sessions}


@router.get("/overlay-test/sessions/{session_id}")
async def get_overlay_session(session_id: str):
    """Get an overlay test session by ID."""
    session = await run_in_threadpool(
        overlay_test_service.get_overlay_test_session, session_id
    )
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.patch("/overlay-test/sessions/{session_id}/pins")
async def update_overlay_pins(session_id: str, body: UpdateOverlayPinsRequest):
    """Update pin state for an overlay test session."""
    result = await run_in_threadpool(
        overlay_test_service.update_overlay_session_pins, session_id, body.pin_state
    )
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.get("/overlay-test/board-image/{filename}")
async def get_board_image(filename: str):
    """Serve a test board image by filename.

    Real samples are prefixed with ``real__`` and served from the real overlay
    test directory; all other filenames are served from the synthetic test dir.
    """
    if ".." in filename:
        raise HTTPException(400, "Invalid filename")
    _REAL_PREFIX = overlay_test_service._REAL_PREFIX
    if filename.startswith(_REAL_PREFIX):
        actual = filename[len(_REAL_PREFIX):]
        if ".." in actual or "/" in actual:
            raise HTTPException(400, "Invalid filename")
        path = overlay_test_service.REAL_OVERLAY_TEST_DIR / actual
    else:
        if "/" in filename:
            raise HTTPException(400, "Invalid filename")
        path = overlay_test_service.CHESS_POSITIONS_TEST_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Board image not found: {filename}")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/overlay-test/validate-real")
async def validate_overlay_detection(limit: int = 100):
    """Validate fast_overlay_check accuracy on real video frames.

    Samples frames from downloaded overlay videos across diverse channels
    and tests whether the overlay detector correctly identifies them.
    """
    result = await run_in_threadpool(
        overlay_test_service.validate_overlay_detection, limit
    )
    return result


@router.post("/overlay-test/extract-real-samples")
async def extract_real_overlay_samples(limit: int = 200):
    """Extract real board crops from video_clips and save to the real test directory.

    This is a one-time (or periodic) operation.  Each clip's mid-point frame is
    cropped to the overlay region, the grid is detected, pieces are classified
    for a pseudo-label FEN, and the crop is saved to
    ``data/overlay/val_real/``.
    """
    result = await run_in_threadpool(
        overlay_test_service.extract_real_overlay_samples, limit
    )
    return result


@router.get("/overlay-test/extract-preview")
async def preview_overlay_extractions(
    limit: int = 200,
    video_ids: str | None = None,
):
    """Preview overlay crops from video_clips with auto-labeled FENs.

    Returns crops as base64 images with predicted FENs for user review
    before saving to disk.  Pass ``video_ids`` as a comma-separated list
    to restrict extraction to specific videos (much faster).
    """
    vid_list = [v.strip() for v in video_ids.split(",") if v.strip()] if video_ids else None
    results = await run_in_threadpool(
        overlay_test_service.preview_real_overlay_extractions, limit, vid_list
    )
    return {"results": results}


class SaveExtractionsRequest(BaseModel):
    confirmations: list[dict]


@router.post("/overlay-test/extract-save")
async def save_confirmed_overlay_extractions(body: SaveExtractionsRequest):
    """Save user-confirmed overlay extractions to test_real/ directory."""
    result = await run_in_threadpool(
        overlay_test_service.save_confirmed_extractions, body.confirmations
    )
    return result


# ── Segmentation Evaluation ────────────────────────────────


class SegmentationInspectRequest(BaseModel):
    video_id: str


class SaveSegmentationEvalRequest(BaseModel):
    segment_consistency: float
    gap_consistency: float
    piece_readability: float
    false_negative_rate: float
    coverage_ratio: float
    sample_size: int
    notes: str | None = None


class CreateSegmentationEvalSessionRequest(BaseModel):
    results: list[dict]
    segment_consistency: float | None = None
    gap_consistency: float | None = None
    piece_readability: float | None = None
    false_negative_rate: float | None = None
    coverage_ratio: float | None = None
    sample_size: int = 0
    pin_state: dict | None = None
    evaluation_id: int | None = None


@router.get("/segmentation-eval/sample")
async def sample_segmentation_videos(limit: int = 10, exclude: str | None = None):
    """Return random sample of downloaded video IDs for segmentation evaluation."""
    exclude_list = exclude.split(",") if exclude else None
    video_ids = await run_in_threadpool(
        segmentation_eval_service.sample_downloaded_videos, limit, exclude_list
    )
    return {"video_ids": video_ids}


@router.post("/segmentation-eval/inspect")
async def inspect_segmentation(body: SegmentationInspectRequest):
    """Run segmentation evaluation on a single video."""
    result = await run_in_threadpool(
        segmentation_eval_service.inspect_segmentation, body.video_id
    )
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@router.post("/segmentation-eval/save-eval")
async def save_segmentation_eval(body: SaveSegmentationEvalRequest):
    """Save a segmentation evaluation result."""
    result = await run_in_threadpool(
        segmentation_eval_service.save_segmentation_eval,
        body.segment_consistency,
        body.gap_consistency,
        body.piece_readability,
        body.false_negative_rate,
        body.coverage_ratio,
        body.sample_size,
        body.notes,
    )
    return result


@router.post("/segmentation-eval/sessions")
async def create_segmentation_eval_session(body: CreateSegmentationEvalSessionRequest):
    """Create a shareable segmentation eval session."""
    result = await run_in_threadpool(
        segmentation_eval_service.create_segmentation_eval_session,
        body.results,
        body.segment_consistency or 0.0,
        body.gap_consistency or 0.0,
        body.piece_readability or 0.0,
        body.false_negative_rate or 0.0,
        body.coverage_ratio or 0.0,
        body.sample_size,
        body.pin_state,
        body.evaluation_id,
    )
    return result


@router.get("/segmentation-eval/sessions")
async def list_segmentation_eval_sessions(limit: int = 20):
    """List recent segmentation eval sessions."""
    sessions = await run_in_threadpool(
        segmentation_eval_service.list_segmentation_eval_sessions, limit
    )
    return {"sessions": sessions}


@router.get("/segmentation-eval/sessions/{session_id}")
async def get_segmentation_eval_session(session_id: str):
    """Get a segmentation eval session by ID."""
    session = await run_in_threadpool(
        segmentation_eval_service.get_segmentation_eval_session, session_id
    )
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.patch("/segmentation-eval/sessions/{session_id}/pins")
async def update_segmentation_eval_pins(session_id: str, body: UpdatePinsRequest):
    """Update pin state for a segmentation eval session."""
    result = await run_in_threadpool(
        segmentation_eval_service.update_segmentation_eval_pins,
        session_id,
        body.pin_state,
    )
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


# ── Calibration Evaluation ─────────────────────────────────


class CalibrationEvalInspectRequest(BaseModel):
    clip_id: int


class SaveCalibrationEvalRequest(BaseModel):
    overlay_iou: float
    grid_success_rate: float
    fen_validity_rate: float
    theme_accuracy: float
    orientation_accuracy: float
    camera_iou: float
    sample_size: int
    notes: str | None = None


class CreateCalibrationEvalSessionRequest(BaseModel):
    results: list[dict]
    overlay_iou_avg: float | None = None
    theme_accuracy: float | None = None
    orientation_accuracy: float | None = None
    grid_success_rate: float | None = None
    fen_validity_rate: float | None = None
    sample_size: int = 0
    pin_state: dict | None = None
    evaluation_id: int | None = None


@router.get("/calibration-eval/sample")
async def sample_calibration_clips(limit: int = 10, exclude: str | None = None):
    """Return sample of calibrated clips for evaluation."""
    exclude_list = [int(x) for x in exclude.split(",")] if exclude else None
    clips = await run_in_threadpool(
        calibration_eval_service.sample_calibration_clips, limit, exclude_list
    )
    return {"clips": clips}


@router.post("/calibration-eval/inspect")
async def inspect_calibration(body: CalibrationEvalInspectRequest):
    """Run calibration evaluation on a single clip."""
    try:
        result = await run_in_threadpool(
            calibration_eval_service.inspect_calibration, body.clip_id
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    return result


@router.post("/calibration-eval/save-eval")
async def save_calibration_eval(body: SaveCalibrationEvalRequest):
    """Save a calibration evaluation result."""
    result = await run_in_threadpool(
        calibration_eval_service.save_calibration_eval,
        body.overlay_iou,
        body.grid_success_rate,
        body.fen_validity_rate,
        body.theme_accuracy,
        body.orientation_accuracy,
        body.camera_iou,
        body.sample_size,
        body.notes,
    )
    return result


@router.post("/calibration-eval/sessions")
async def create_calibration_eval_session(body: CreateCalibrationEvalSessionRequest):
    """Create a shareable calibration eval session."""
    result = await run_in_threadpool(
        calibration_eval_service.create_calibration_eval_session,
        body.results,
        body.overlay_iou_avg or 0.0,
        body.theme_accuracy or 0.0,
        body.orientation_accuracy or 0.0,
        body.grid_success_rate or 0.0,
        body.fen_validity_rate or 0.0,
        body.sample_size,
        body.pin_state,
        body.evaluation_id,
    )
    return result


@router.get("/calibration-eval/sessions")
async def list_calibration_eval_sessions(limit: int = 20):
    """List recent calibration eval sessions."""
    sessions = await run_in_threadpool(
        calibration_eval_service.list_calibration_eval_sessions, limit
    )
    return {"sessions": sessions}


@router.get("/calibration-eval/sessions/{session_id}")
async def get_calibration_eval_session(session_id: str):
    """Get a calibration eval session by ID."""
    session = await run_in_threadpool(
        calibration_eval_service.get_calibration_eval_session, session_id
    )
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.patch("/calibration-eval/sessions/{session_id}/pins")
async def update_calibration_eval_pins(session_id: str, body: UpdatePinsRequest):
    """Update pin state for a calibration eval session."""
    result = await run_in_threadpool(
        calibration_eval_service.update_calibration_eval_pins,
        session_id,
        body.pin_state,
    )
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result
