"""Overlay tester endpoints."""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.services import overlay_service

router = APIRouter()


@router.post("/test-image")
async def test_image(
    image: UploadFile = File(...),
    overlay_bbox: str | None = Form(None),
    flipped: bool = Form(False),
    theme: str = Form("lichess_default"),
):
    """Test overlay detection and reading on an uploaded image."""
    image_bytes = await image.read()

    bbox = None
    if overlay_bbox:
        parts = [int(x.strip()) for x in overlay_bbox.split(",")]
        bbox = tuple(parts)

    result = await run_in_threadpool(
        overlay_service.test_image,
        image_bytes,
        overlay_bbox=bbox,
        flipped=flipped,
        theme=theme,
    )
    return result
