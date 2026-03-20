"""Video annotator endpoints (session-based)."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel

from api.services import video_service

router = APIRouter()


class OpenVideoRequest(BaseModel):
    video_path: str
    channel_handle: str | None = None


class DetectMovesRequest(BaseModel):
    sample_fps: float = 2.0


@router.post("/open")
async def open_video(body: OpenVideoRequest):
    """Open a video file and create an annotation session."""
    try:
        return await run_in_threadpool(
            video_service.open_video,
            body.video_path,
            body.channel_handle,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{session_id}/frame")
async def get_frame(session_id: str, index: int = Query(...)):
    """Get a single video frame as JPEG."""
    try:
        jpeg_bytes = await run_in_threadpool(
            video_service.get_frame_jpeg, session_id, index
        )
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/{session_id}/overlay-read")
async def read_overlay(session_id: str, index: int = Query(...)):
    """Read overlay FEN at a specific frame."""
    try:
        return await run_in_threadpool(
            video_service.read_overlay_at_frame, session_id, index
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{session_id}/detect-moves")
async def detect_moves(session_id: str, body: DetectMovesRequest):
    """Run full move detection on the video."""
    try:
        return await run_in_threadpool(
            video_service.detect_moves, session_id, body.sample_fps
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """Close a video session."""
    video_service.delete_session(session_id)
    return {"deleted": True}
