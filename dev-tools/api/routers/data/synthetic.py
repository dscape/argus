"""Synthetic data generation monitoring and control endpoints."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services.data import generation_service, synthetic_service

router = APIRouter()


# ── Monitoring ─────────────────────────────────────────────


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


# ── Generation Control ─────────────────────────────────────


class GenerateRequest(BaseModel):
    num_clips: int = 100
    output_dir: str = "data/argus/train"
    image_size: int = 224
    clip_length: int = 16
    frames_per_move: int = 4
    seed: int = 42
    quality: str = "training"
    broadcast_bias: float = 0.0


@router.post("/generate")
async def start_generation(body: GenerateRequest):
    """Start synthetic data generation in the background."""
    try:
        return await run_in_threadpool(
            generation_service.start_generation,
            body.num_clips,
            body.output_dir,
            body.image_size,
            body.clip_length,
            body.frames_per_move,
            body.seed,
            body.quality,
            body.broadcast_bias,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/generate/status")
async def get_generation_status():
    """Get current generation job status."""
    return generation_service.get_status()


@router.post("/generate/stop")
async def stop_generation():
    """Stop the current generation job."""
    return await run_in_threadpool(generation_service.stop_generation)
