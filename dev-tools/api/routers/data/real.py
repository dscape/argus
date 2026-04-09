"""Real-footage training data inventory and processing endpoints."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.services.data import real_service

router = APIRouter()


@router.get("/overview")
async def get_overview(
    clips_dir: str = Query("data/argus/train_real"),
    limit: int = Query(100, ge=1, le=5000),
):
    try:
        return await run_in_threadpool(
            real_service.get_overview,
            clips_dir,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


class ProcessRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)
    clips_dir: str = "data/argus/train_real"
    min_moves: int = Field(default=5, ge=1)


@router.post("/process")
async def start_processing(body: ProcessRequest):
    try:
        return await run_in_threadpool(
            real_service.start_processing,
            limit=body.limit,
            clips_dir=body.clips_dir,
            min_moves=body.min_moves,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/process/status")
async def get_processing_status():
    return real_service.get_processing_status()


@router.post("/process/stop")
async def stop_processing():
    return await run_in_threadpool(real_service.stop_processing)
