"""Shared board-reading helpers for local analysis and dev tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.overlay.grid_detector import detect_grid
from pipeline.overlay.piece_classifier import read_fen_with_grid
from pipeline.overlay.scanner import detect_overlay_runtime

logger = logging.getLogger(__name__)


@dataclass
class CropReadResult:
    """FEN read from a calibrated board crop."""

    fen: str | None
    method: str | None


@dataclass
class FrameReadResult:
    """FEN read from a full video frame."""

    fen: str | None
    method: str | None


class OverlayFrameReader:
    """Read a board state from a full frame using the overlay stack."""

    def __init__(self, config: VideoAnalysisConfig) -> None:
        self.config = config
        self._locked_bbox: tuple[int, int, int, int] | None = None
        self._locked_grid = None

    def read(self, frame_rgb: np.ndarray) -> FrameReadResult:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self._locked_bbox is not None and self._locked_grid is not None:
            fen = self._read_locked(frame_bgr)
            if _fen_looks_plausible(fen):
                return FrameReadResult(fen=fen, method="overlay_locked")
            self._locked_bbox = None
            self._locked_grid = None

        detection = detect_overlay_runtime(frame_bgr)
        if not detection.found or detection.bbox is None:
            return FrameReadResult(fen=None, method=None)

        read_result = _read_overlay_bbox(detection.bbox, frame_bgr, self.config)
        if read_result.fen is None or read_result.grid is None:
            return FrameReadResult(fen=None, method=None)

        self._locked_bbox = detection.bbox
        self._locked_grid = read_result.grid
        logger.info(
            "Overlay locked: bbox=%s sq=%d h0=%d v0=%d",
            detection.bbox,
            read_result.grid.sq_size,
            read_result.grid.h_lines[0],
            read_result.grid.v_lines[0],
        )
        return FrameReadResult(fen=read_result.fen, method="overlay_runtime")

    def _read_locked(self, frame_bgr: np.ndarray) -> str | None:
        assert self._locked_bbox is not None
        assert self._locked_grid is not None

        overlay_crop = _crop_bbox(frame_bgr, self._locked_bbox)
        if overlay_crop is None:
            return None
        return read_fen_with_grid(overlay_crop, self._locked_grid, device=self.config.device)


class HybridFrameReader(OverlayFrameReader):
    """Overlay-first reader with MLX fallback for missed frames."""

    def read(self, frame_rgb: np.ndarray) -> FrameReadResult:
        result = super().read(frame_rgb)
        if result.fen is not None:
            return result

        from pipeline.mlx.board_segmenter import segment_board
        from pipeline.mlx.piece_detector import detect_pieces

        segment = segment_board(frame_rgb, self.config)
        if segment is None:
            return FrameReadResult(fen=None, method=None)

        state = detect_pieces(
            segment.cropped_board,
            self.config,
            segmenter_method=segment.method,
        )
        if state is None or not _fen_looks_plausible(state.fen):
            return FrameReadResult(fen=None, method=None)

        return FrameReadResult(fen=state.fen, method=f"hybrid_{state.method}")


def build_frame_reader(config: VideoAnalysisConfig) -> OverlayFrameReader:
    """Build the configured full-frame reader."""
    if config.reader_backend == "overlay":
        return OverlayFrameReader(config)
    if config.reader_backend == "hybrid":
        return HybridFrameReader(config)
    raise ValueError(f"Unsupported reader backend: {config.reader_backend}")


def read_overlay_crop(
    overlay_crop: np.ndarray,
    config: VideoAnalysisConfig,
) -> CropReadResult:
    """Read a board state from a calibrated board crop."""
    grid = find_board_in_crop(overlay_crop)
    if grid is not None:
        fen = read_fen_with_grid(overlay_crop, grid, device=config.device)
        if _fen_looks_plausible(fen):
            return CropReadResult(fen=fen, method="overlay")

    if config.reader_backend == "overlay":
        return CropReadResult(fen=None, method=None)
    if config.reader_backend != "hybrid":
        raise ValueError(f"Unsupported reader backend: {config.reader_backend}")

    from pipeline.mlx.vlm_analyzer import read_board_position

    overlay_rgb = cv2.cvtColor(overlay_crop, cv2.COLOR_BGR2RGB)
    fen = read_board_position(overlay_rgb, config)
    if _fen_looks_plausible(fen):
        return CropReadResult(fen=fen, method="vlm_direct")
    return CropReadResult(fen=None, method=None)


@dataclass
class _OverlayCropRead:
    fen: str | None
    grid: object | None


def _read_overlay_bbox(
    bbox: tuple[int, int, int, int],
    frame_bgr: np.ndarray,
    config: VideoAnalysisConfig,
) -> _OverlayCropRead:
    overlay_crop = _crop_bbox(frame_bgr, bbox)
    if overlay_crop is None:
        return _OverlayCropRead(fen=None, grid=None)

    grid = find_board_in_crop(overlay_crop)
    if grid is None:
        return _OverlayCropRead(fen=None, grid=None)

    fen = read_fen_with_grid(overlay_crop, grid, device=config.device)
    if not _fen_looks_plausible(fen):
        return _OverlayCropRead(fen=None, grid=None)

    return _OverlayCropRead(fen=fen, grid=grid)


def _crop_bbox(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    x, y, w, h = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    if w <= 0 or h <= 0:
        return None
    return frame_bgr[y : y + h, x : x + w]


def find_board_in_crop(overlay_crop: np.ndarray):
    """Detect the board grid inside an already-cropped overlay region."""
    return detect_grid(overlay_crop)


def _fen_looks_plausible(fen: str | None) -> bool:
    if not fen:
        return False
    return fen.count("/") == 7 and "K" in fen and "k" in fen
