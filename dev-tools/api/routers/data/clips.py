"""Clip inspector endpoints."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api.services.data import clip_service

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
    """Return inspection data for a loaded clip."""
    try:
        return await run_in_threadpool(clip_service.inspect, session_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@router.get("/{session_id}/frame/{frame_index}")
async def get_clip_frame(session_id: str, frame_index: int):
    """Return one clip frame as a PNG image."""
    try:
        png_bytes = await run_in_threadpool(
            clip_service.get_frame_png,
            session_id,
            frame_index,
        )
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@router.delete("/{session_id}")
def delete_clip_session(session_id: str):
    """Delete a clip inspection session."""
    clip_service.delete_session(session_id)
    return {"deleted": True}
