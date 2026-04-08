"""Overlay bbox training-annotation endpoints.

These endpoints maintain YOLO detector ground truth. Runtime overlay
localization uses committed detector weights, not these saved bboxes.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from api.services.annotate import overlay_bbox_service

router = APIRouter()


@router.get("/frames")
def list_frames():
    """List all frames with annotation status."""
    return {"frames": overlay_bbox_service.list_frames()}


@router.get("/frame-image/{video_id}/{label}")
def get_frame_image(video_id: str, label: str):
    """Serve a frame image."""
    path = overlay_bbox_service.get_frame_path(video_id, label)
    if path is None:
        raise HTTPException(404, f"Frame not found: {video_id}/{label}")
    return FileResponse(path, media_type="image/jpeg")


class RefineInput(BaseModel):
    frame_key: str
    rough_bbox: list[int]


class AutoDetectInput(BaseModel):
    frame_key: str


@router.post("/auto-detect")
async def auto_detect_bbox(body: AutoDetectInput):
    """Suggest a bbox for an unannotated frame using the committed detector."""
    video_id, label = body.frame_key.split("/", 1)
    path = overlay_bbox_service.get_frame_path(video_id, label)
    if path is None:
        raise HTTPException(404, f"Frame not found: {body.frame_key}")
    result = await run_in_threadpool(overlay_bbox_service.auto_detect_bbox, path)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@router.post("/refine")
async def refine_bbox(body: RefineInput):
    """Smart-correct a rough training bbox to align with the overlay grid."""
    if len(body.rough_bbox) != 4:
        raise HTTPException(400, "rough_bbox must have 4 elements [x, y, w, h]")
    video_id, label = body.frame_key.split("/", 1)
    path = overlay_bbox_service.get_frame_path(video_id, label)
    if path is None:
        raise HTTPException(404, f"Frame not found: {body.frame_key}")
    result = await run_in_threadpool(
        overlay_bbox_service.refine_bbox, path, body.rough_bbox
    )
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


class AnnotateInput(BaseModel):
    frame_key: str
    has_overlay: bool
    bbox: list[int] | None = None
    notes: str = ""


@router.post("/annotate")
def save_annotation(body: AnnotateInput):
    """Save a YOLO-training overlay bbox annotation."""
    if body.has_overlay and body.bbox is None:
        raise HTTPException(400, "bbox required when has_overlay is true")
    if body.bbox is not None and len(body.bbox) != 4:
        raise HTTPException(400, "bbox must have 4 elements [x, y, w, h]")
    result = overlay_bbox_service.save_annotation(
        body.frame_key, body.has_overlay, body.bbox, body.notes
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/annotate/{frame_key:path}")
def delete_annotation(frame_key: str):
    """Remove an annotation."""
    if not overlay_bbox_service.delete_annotation(frame_key):
        raise HTTPException(404, f"No annotation for {frame_key}")
    return {"deleted": True}
