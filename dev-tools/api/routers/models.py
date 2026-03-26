"""Model inspection and evaluation endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services import models_service, overlay_test_service

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
