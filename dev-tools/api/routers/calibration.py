"""Calibration CRUD endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services import calibration_service

router = APIRouter()


class CalibrationInput(BaseModel):
    overlay: list[int]
    camera: list[int]
    ref_resolution: list[int] = [1920, 1080]
    board_flipped: bool = False
    board_theme: str = "lichess_default"


@router.get("/")
def list_calibrations():
    """List all saved calibrations."""
    return {"calibrations": calibration_service.list_all()}


@router.get("/{channel_handle}")
def get_calibration(channel_handle: str):
    """Get calibration for a channel."""
    result = calibration_service.get_one(channel_handle)
    if result is None:
        raise HTTPException(404, f"No calibration for {channel_handle}")
    return result


@router.put("/{channel_handle}")
def save_calibration(channel_handle: str, body: CalibrationInput):
    """Save or update calibration for a channel."""
    return calibration_service.save(channel_handle, body.model_dump())


@router.delete("/{channel_handle}")
def delete_calibration(channel_handle: str):
    """Delete calibration for a channel."""
    if not calibration_service.delete(channel_handle):
        raise HTTPException(404, f"No calibration for {channel_handle}")
    return {"deleted": True}
