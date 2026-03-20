"""Clip inspector endpoints (session-based)."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api.services import clip_service

router = APIRouter()


@router.post("/load")
async def load_clip(clip_file: UploadFile = File(...)):
    """Upload a .pt clip and create an inspection session."""
    file_bytes = await clip_file.read()
    session_id = await run_in_threadpool(
        clip_service.create_session,
        file_bytes,
        clip_file.filename or "clip.pt",
    )
    return {"session_id": session_id}


@router.get("/{session_id}/info")
async def get_clip_info(session_id: str):
    """Get full inspection data for a loaded clip."""
    try:
        return await run_in_threadpool(clip_service.inspect, session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/{session_id}/frame/{frame_index}")
async def get_clip_frame(session_id: str, frame_index: int):
    """Get a single frame as a PNG image."""
    try:
        png_bytes = await run_in_threadpool(
            clip_service.get_frame_png, session_id, frame_index
        )
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/{session_id}")
def delete_clip_session(session_id: str):
    """Clean up a clip session."""
    clip_service.delete_session(session_id)
    return {"deleted": True}
