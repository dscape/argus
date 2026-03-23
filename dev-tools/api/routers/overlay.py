"""Overlay tester endpoints."""

import urllib.request

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

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


class TestUrlRequest(BaseModel):
    url: str
    overlay_bbox: str | None = None
    flipped: bool = False
    theme: str = "lichess_default"


@router.post("/test-url")
async def test_url(body: TestUrlRequest):
    """Test overlay detection on an image fetched from a URL."""
    try:
        req = urllib.request.Request(body.url, headers={"User-Agent": "argus-dev-tools/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            image_bytes = resp.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch image: {e}")

    bbox = None
    if body.overlay_bbox:
        parts = [int(x.strip()) for x in body.overlay_bbox.split(",")]
        bbox = tuple(parts)

    result = await run_in_threadpool(
        overlay_service.test_image,
        image_bytes,
        overlay_bbox=bbox,
        flipped=body.flipped,
        theme=body.theme,
    )
    return result
