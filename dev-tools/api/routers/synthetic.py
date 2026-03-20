"""Synthetic data generation monitoring endpoints."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services import synthetic_service

router = APIRouter()


@router.get("/scan")
async def scan_directory(
    directory: str = Query(..., description="Path to the output directory"),
    expected_clips: int | None = Query(None, description="Expected total clips for progress"),
):
    """Scan a directory for .pt clip files. Lightweight — no torch loading."""
    return await run_in_threadpool(
        synthetic_service.scan_directory, directory, expected_clips
    )


@router.get("/stats")
async def get_stats(
    directory: str = Query(..., description="Path to the output directory"),
):
    """Compute aggregated stats by loading all .pt clips. Heavy operation."""
    try:
        return await run_in_threadpool(synthetic_service.get_clip_stats, directory)
    except Exception as e:
        raise HTTPException(500, str(e))


class InspectClipRequest(BaseModel):
    filepath: str


@router.post("/inspect")
async def inspect_clip(req: InspectClipRequest):
    """Load a .pt clip from disk into a clip_service session for inspection."""
    try:
        session_id = await run_in_threadpool(
            synthetic_service.load_clip_from_path, req.filepath
        )
        return {"session_id": session_id}
    except ValueError as e:
        raise HTTPException(404, str(e))
