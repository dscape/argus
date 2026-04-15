"""Clip inspector endpoints (session-based)."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from api.services.data import clip_service

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_PROJECT_ROOT / candidate).resolve()

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


class LoadFromPathRequest(BaseModel):
    filepath: str


@router.post("/load-from-path")
async def load_clip_from_path(body: LoadFromPathRequest):
    """Load a .pt clip from a server-side file path."""
    resolved = _resolve(body.filepath)
    if not resolved.exists():
        raise HTTPException(404, f"File not found: {resolved}")
    with resolved.open("rb") as f:
        file_bytes = f.read()
    session_id = await run_in_threadpool(
        clip_service.create_session,
        file_bytes,
        resolved.name,
        source_filepath=str(resolved),
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
    """Get a single stored clip frame as a PNG image."""
    try:
        png_bytes = await run_in_threadpool(
            clip_service.get_frame_png, session_id, frame_index
        )
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/{session_id}/camera-frame/{frame_index}")
async def get_clip_camera_frame(
    session_id: str, frame_index: int, padding: int = Query(0, ge=0, le=500)
):
    """Get the camera crop from the source video with optional padding."""
    try:
        png_bytes = await run_in_threadpool(
            clip_service.get_camera_frame_png, session_id, frame_index, padding_px=padding
        )
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/{session_id}/overlay-frame/{frame_index}")
async def get_clip_overlay_frame(session_id: str, frame_index: int):
    """Get the corresponding overlay crop from the source video as a PNG image."""
    try:
        png_bytes = await run_in_threadpool(
            clip_service.get_overlay_frame_png, session_id, frame_index
        )
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.api_route("/{session_id}/source-video", methods=["GET", "HEAD"])
async def get_clip_source_video(session_id: str):
    """Get the source video file for a real clip review session."""
    try:
        video_path = await run_in_threadpool(clip_service.get_source_video_path, session_id)
        return FileResponse(video_path)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/annotation")
async def get_clip_annotation(filename: str = Query(...)):
    """Load reviewer notes for a clip filename."""
    return await run_in_threadpool(clip_service.get_annotation, filename)


class SaveAnnotationRequest(BaseModel):
    filename: str
    content: str


@router.put("/annotation")
async def save_clip_annotation(body: SaveAnnotationRequest):
    """Save reviewer notes for a clip filename."""
    return await run_in_threadpool(
        clip_service.save_annotation,
        body.filename,
        body.content,
    )


@router.delete("/{session_id}")
def delete_clip_session(session_id: str):
    """Clean up a clip session."""
    clip_service.delete_session(session_id)
    return {"deleted": True}
