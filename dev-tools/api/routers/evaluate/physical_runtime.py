"""Endpoints for physical runtime evaluation and visualization."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.services.evaluate import physical_runtime_service

router = APIRouter()


class RenderRuntimeRequest(BaseModel):
    clip_path: str | None = None
    frame_start: int = 0
    frame_count: int = 8
    panel_size: int = 240
    device: str = "cpu"
    model_path: str | None = None


class InspectRuntimeFrameRequest(BaseModel):
    annotation_id: str
    panel_size: int = 240
    device: str = "cpu"
    model_path: str | None = None


class InspectRuntimeFramesRequest(BaseModel):
    annotation_ids: list[str]
    panel_size: int = 240
    device: str = "cpu"
    model_path: str | None = None


class SavePhysicalRuntimeEvalRequest(BaseModel):
    square_accuracy: float
    non_empty_accuracy: float | None = None
    exact_match_rate: float | None = None
    sample_size: int
    elapsed_ms_avg: float | None = None
    images_per_minute: int | None = None
    stateless_square_accuracy: float | None = None
    stateless_non_empty_accuracy: float | None = None
    stateless_exact_match_rate: float | None = None
    notes: str | None = None
    model_path: str | None = None


class CreatePhysicalRuntimeSessionRequest(BaseModel):
    results: list[dict]
    square_accuracy: float | None = None
    non_empty_accuracy: float | None = None
    exact_match_rate: float | None = None
    sample_size: int = 0
    pin_state: dict | None = None
    evaluation_id: int | None = None


class UpdatePinsRequest(BaseModel):
    pin_state: dict


@router.post("/render")
async def render_runtime_visualization(body: RenderRuntimeRequest):
    try:
        return await run_in_threadpool(
            physical_runtime_service.render_runtime_visualization,
            clip_path=body.clip_path,
            frame_start=body.frame_start,
            frame_count=body.frame_count,
            panel_size=body.panel_size,
            device=body.device,
            model_path=body.model_path,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/sample")
async def sample_runtime_frames(limit: int = 20, exclude: str | None = None):
    exclude_list = exclude.split(",") if exclude else None
    try:
        frames = await run_in_threadpool(
            physical_runtime_service.sample_runtime_frames,
            limit,
            exclude_list,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"frames": frames}


@router.get("/models")
async def list_runtime_models():
    models = await run_in_threadpool(physical_runtime_service.list_runtime_models)
    return {"models": models}


@router.post("/inspect")
async def inspect_runtime_frame(body: InspectRuntimeFrameRequest):
    try:
        return await run_in_threadpool(
            physical_runtime_service.inspect_runtime_frame,
            annotation_id=body.annotation_id,
            panel_size=body.panel_size,
            device=body.device,
            model_path=body.model_path,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.post("/inspect-batch")
async def inspect_runtime_frames(body: InspectRuntimeFramesRequest):
    try:
        results = await run_in_threadpool(
            physical_runtime_service.inspect_runtime_frames,
            annotation_ids=body.annotation_ids,
            panel_size=body.panel_size,
            device=body.device,
            model_path=body.model_path,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    return {"results": results}


@router.post("/save-eval")
async def save_physical_runtime_eval(body: SavePhysicalRuntimeEvalRequest):
    return await run_in_threadpool(
        physical_runtime_service.save_physical_runtime_eval,
        body.square_accuracy,
        body.non_empty_accuracy,
        body.exact_match_rate,
        body.sample_size,
        body.elapsed_ms_avg,
        body.images_per_minute,
        body.stateless_square_accuracy,
        body.stateless_non_empty_accuracy,
        body.stateless_exact_match_rate,
        body.notes,
        body.model_path,
    )


@router.post("/sessions")
async def create_physical_runtime_session(body: CreatePhysicalRuntimeSessionRequest):
    return await run_in_threadpool(
        physical_runtime_service.create_physical_runtime_session,
        body.results,
        body.square_accuracy,
        body.non_empty_accuracy,
        body.exact_match_rate,
        body.sample_size,
        body.pin_state,
        body.evaluation_id,
    )


@router.get("/sessions")
async def list_physical_runtime_sessions(limit: int = 20):
    sessions = await run_in_threadpool(
        physical_runtime_service.list_physical_runtime_sessions,
        limit,
    )
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_physical_runtime_session(session_id: str):
    session = await run_in_threadpool(
        physical_runtime_service.get_physical_runtime_session,
        session_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@router.patch("/sessions/{session_id}/pins")
async def update_physical_runtime_pins(session_id: str, body: UpdatePinsRequest):
    result = await run_in_threadpool(
        physical_runtime_service.update_physical_runtime_pins,
        session_id,
        body.pin_state,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/session-image/{session_id}/{filename}")
async def get_session_image(session_id: str, filename: str):
    path = physical_runtime_service.get_session_image_path(session_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")
