"""Synthetic data monitoring and generation endpoints."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services.data import generation_service, synthetic_service

router = APIRouter()


@router.get("/scan")
async def scan_directory(
    directory: str = Query(..., description="Path to the output directory"),
    expected_clips: int | None = Query(None, description="Expected total clips for progress"),
):
    """Scan a directory for .pt clips without loading them into memory."""
    return await run_in_threadpool(
        synthetic_service.scan_directory,
        directory,
        expected_clips,
    )


@router.get("/stats")
async def get_stats(directory: str = Query(..., description="Path to the output directory")):
    """Compute aggregate statistics for a synthetic clip directory."""
    try:
        return await run_in_threadpool(synthetic_service.get_clip_stats, directory)
    except Exception as error:
        raise HTTPException(500, str(error)) from error


class InspectClipRequest(BaseModel):
    filepath: str


@router.post("/inspect")
async def inspect_clip(request: InspectClipRequest):
    """Load a clip from disk into a clip inspection session."""
    try:
        session_id = await run_in_threadpool(
            synthetic_service.load_clip_from_path,
            request.filepath,
        )
        return {"session_id": session_id}
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


class GenerateRequest(BaseModel):
    num_clips: int = 100
    output_dir: str = "data/train"
    image_size: int = 224
    clip_length: int = 16
    frames_per_move: int = 4
    seed: int = 42
    quality: str = "training"


@router.post("/generate")
async def start_generation(request: GenerateRequest):
    """Start a synthetic data generation job."""
    try:
        return await run_in_threadpool(
            generation_service.start_generation,
            request.num_clips,
            request.output_dir,
            request.image_size,
            request.clip_length,
            request.frames_per_move,
            request.seed,
            request.quality,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except Exception as error:
        raise HTTPException(500, str(error)) from error


@router.get("/generate/status")
async def get_generation_status():
    """Return the current synthetic generation status."""
    return generation_service.get_status()


@router.post("/generate/stop")
async def stop_generation():
    """Request cancellation of the current generation job."""
    return await run_in_threadpool(generation_service.stop_generation)
