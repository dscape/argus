"""Endpoints for building manually labeled non-held-out physical-board training data."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.services.annotate import physical_train_service

router = APIRouter()


class RectifyRequest(BaseModel):
    session_id: str
    frame_index: int
    corners: list[list[float]] = Field(min_length=4, max_length=4)
    output_size: int = 512
    padding_px: int = 0


class SaveAnnotationRequest(BaseModel):
    session_id: str
    clip_path: str
    frame_index: int
    corners: list[list[float]] = Field(min_length=4, max_length=4)
    labels: list[int | None] = Field(min_length=64, max_length=64)
    output_size: int = 512
    padding_px: int = 0


@router.get("/clips")
async def list_clip_files(clips_dir: str = "data/argus/train_real", limit: int = 200):
    return await run_in_threadpool(
        physical_train_service.list_clip_files,
        clips_dir,
        limit=limit,
    )


@router.get("/summary")
async def get_annotation_summary():
    return await run_in_threadpool(physical_train_service.get_annotation_summary)


@router.get("/annotation")
async def get_frame_annotation(
    clip_path: str,
    frame_index: int,
    session_id: str | None = None,
    padding_px: int = 0,
):
    return {
        "annotation": await run_in_threadpool(
            physical_train_service.get_frame_annotation,
            clip_path,
            frame_index,
            session_id=session_id,
            padding_px=padding_px,
        )
    }


@router.get("/corrections")
async def get_move_corrections(session_id: str, clip_path: str):
    try:
        return await run_in_threadpool(
            physical_train_service.get_move_corrections,
            session_id,
            clip_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/annotation")
async def delete_annotation(clip_path: str, frame_index: int):
    summary = await run_in_threadpool(
        physical_train_service.delete_annotation,
        clip_path,
        frame_index,
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"summary": summary}


class DetectCornersRequest(BaseModel):
    session_id: str
    frame_index: int
    padding_px: int = 0


@router.post("/detect-corners")
async def detect_corners(body: DetectCornersRequest):
    from api.services.annotate import physical_eval_service

    try:
        result = await run_in_threadpool(
            physical_eval_service.detect_corners,
            body.session_id,
            body.frame_index,
            padding_px=body.padding_px,
        )
        return {"detection": result}
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/rectify")
async def rectify_frame(body: RectifyRequest):
    return await run_in_threadpool(
        physical_train_service.rectify_frame,
        body.session_id,
        body.frame_index,
        body.corners,
        output_size=body.output_size,
        padding_px=body.padding_px,
    )


@router.post("/save")
async def save_annotation(body: SaveAnnotationRequest):
    return await run_in_threadpool(
        physical_train_service.save_annotation,
        body.session_id,
        body.clip_path,
        body.frame_index,
        body.corners,
        body.labels,
        output_size=body.output_size,
        padding_px=body.padding_px,
    )
