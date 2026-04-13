"""Endpoints for physical runtime visualization on held-out eval clips."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services.evaluate import physical_runtime_service

router = APIRouter()


class RenderRuntimeRequest(BaseModel):
    clip_path: str | None = None
    frame_start: int = 0
    frame_count: int = 8
    panel_size: int = 240
    device: str = "cpu"


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
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
