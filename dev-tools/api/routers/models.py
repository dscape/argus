"""Model inspection and evaluation endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services import models_service

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
