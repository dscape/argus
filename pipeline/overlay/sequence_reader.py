"""Fast incremental overlay board reading for sampled video sequences."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.overlay.grid_detector import GridResult
from pipeline.overlay.overlay_move_detector import MAX_MOVE_CHANGED_SQUARES, count_fen_differences
from pipeline.overlay.piece_classifier import (
    BoardRead,
    class_grid_to_fen,
    classify_square_crops,
    read_board_with_grid,
)

MIN_MOVE_CHANGED_SQUARES = 2
_DESCRIPTOR_SIZE = 12
_BORDER_FRACTION = 0.15


@dataclass(frozen=True)
class GateDecision:
    """Decision returned by :class:`SquareChangeGate`."""

    should_read: bool
    changed_indices: tuple[int, ...] = ()
    reason: str = "stable"


@dataclass(frozen=True)
class SequenceReadResult:
    """Incremental board read result."""

    fen: str
    method: str
    did_expensive_read: bool


def extract_square_descriptors(
    frame: np.ndarray,
    grid: GridResult,
    *,
    descriptor_size: int = _DESCRIPTOR_SIZE,
    border_fraction: float = _BORDER_FRACTION,
) -> np.ndarray:
    """Extract cheap per-square appearance descriptors for change gating."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    descriptors = np.empty((64, descriptor_size * descriptor_size), dtype=np.float32)
    index = 0
    height, width = gray.shape[:2]

    for row in range(8):
        y1 = max(0, min(grid.h_lines[row], height - 1))
        y2 = max(y1 + 1, min(grid.h_lines[row + 1], height))
        trim_y = max(1, int((y2 - y1) * border_fraction))
        inner_y1 = min(y1 + trim_y, y2 - 1)
        inner_y2 = max(inner_y1 + 1, y2 - trim_y)

        for col in range(8):
            x1 = max(0, min(grid.v_lines[col], width - 1))
            x2 = max(x1 + 1, min(grid.v_lines[col + 1], width))
            trim_x = max(1, int((x2 - x1) * border_fraction))
            inner_x1 = min(x1 + trim_x, x2 - 1)
            inner_x2 = max(inner_x1 + 1, x2 - trim_x)

            square = gray[inner_y1:inner_y2, inner_x1:inner_x2]
            small = cv2.resize(square, (descriptor_size, descriptor_size), interpolation=cv2.INTER_AREA)
            descriptor = small.astype(np.float32)
            descriptor -= float(descriptor.mean())
            std = float(descriptor.std())
            if std > 1e-3:
                descriptor /= std
            descriptors[index] = descriptor.reshape(-1)
            index += 1

    return descriptors


def descriptor_delta_scores(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return one mean absolute delta score per square."""
    return np.mean(np.abs(current - reference), axis=1)


class SquareChangeGate:
    """Cheap per-square change gate with debounce, hysteresis, and resync."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.30,
        low_threshold: float = 0.22,
        min_changed_squares: int = MIN_MOVE_CHANGED_SQUARES,
        max_changed_squares: int = MAX_MOVE_CHANGED_SQUARES,
        debounce_frames: int = 1,
        resync_interval: int = 12,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_changed_squares = min_changed_squares
        self.max_changed_squares = max_changed_squares
        self.debounce_frames = debounce_frames
        self.resync_interval = resync_interval

        self._reference: np.ndarray | None = None
        self._candidate_mask: np.ndarray | None = None
        self._candidate_frames = 0
        self._suppressed_mask: np.ndarray | None = None
        self._samples_since_read = 0

    def mark_read(self, descriptors: np.ndarray) -> None:
        self._reference = descriptors.copy()
        self._candidate_mask = None
        self._candidate_frames = 0
        self._suppressed_mask = None
        self._samples_since_read = 0

    def record_skip(self) -> None:
        self._samples_since_read += 1

    def suppress(self, changed_indices: tuple[int, ...]) -> None:
        mask = np.zeros(64, dtype=bool)
        if changed_indices:
            mask[list(changed_indices)] = True
            self._suppressed_mask = mask
            self._candidate_mask = mask
        else:
            self._suppressed_mask = None
            self._candidate_mask = None
        self._candidate_frames = 0
        self._samples_since_read = 0

    def evaluate(self, descriptors: np.ndarray) -> GateDecision:
        if self._reference is None:
            return GateDecision(should_read=True, reason="bootstrap")

        scores = descriptor_delta_scores(self._reference, descriptors)
        changed = self._apply_hysteresis(scores)
        changed_indices = tuple(int(idx) for idx in np.flatnonzero(changed).tolist())
        changed_count = len(changed_indices)

        if changed_count == 0:
            self._candidate_mask = None
            self._candidate_frames = 0
            self._suppressed_mask = None
            if self._samples_since_read >= self.resync_interval:
                return GateDecision(should_read=True, reason="resync")
            return GateDecision(should_read=False, reason="stable")

        if self._samples_since_read >= self.resync_interval:
            self._candidate_mask = changed
            self._candidate_frames = 1
            return GateDecision(True, changed_indices, reason="resync")

        if self._suppressed_mask is not None and np.array_equal(changed, self._suppressed_mask):
            return GateDecision(False, changed_indices, reason="suppressed")

        if self.min_changed_squares <= changed_count <= self.max_changed_squares:
            if self._candidate_mask is not None and np.array_equal(changed, self._candidate_mask):
                self._candidate_frames += 1
            else:
                self._candidate_mask = changed
                self._candidate_frames = 1

            if self._candidate_frames >= self.debounce_frames:
                return GateDecision(True, changed_indices, reason="candidate")
            return GateDecision(False, changed_indices, reason="debouncing")

        self._candidate_mask = changed
        self._candidate_frames = 1
        return GateDecision(False, changed_indices, reason="unstable")

    def _apply_hysteresis(self, scores: np.ndarray) -> np.ndarray:
        changed = scores >= self.high_threshold
        if self._candidate_mask is None:
            return changed
        kept = np.logical_and(self._candidate_mask, scores >= self.low_threshold)
        return np.logical_or(changed, kept)


class LockedOverlaySequenceReader:
    """Read a locked overlay crop efficiently across a sampled frame sequence."""

    def __init__(
        self,
        grid: GridResult,
        *,
        device: str = "cpu",
        gate: SquareChangeGate | None = None,
    ) -> None:
        self.grid = grid
        self.device = device
        self.gate = gate or SquareChangeGate()
        self.board_state: BoardRead | None = None
        self.num_full_reads = 0
        self.num_partial_reads = 0
        self.num_cached_reads = 0
        self.num_suppressed_reads = 0

    def seed(self, overlay_crop: np.ndarray, board_state: BoardRead | None = None) -> SequenceReadResult:
        descriptors = extract_square_descriptors(overlay_crop, self.grid)
        if board_state is None:
            board_state = read_board_with_grid(overlay_crop, self.grid, device=self.device)
            self.num_full_reads += 1
        self.board_state = board_state
        self.gate.mark_read(descriptors)
        return SequenceReadResult(
            fen=board_state.fen,
            method="overlay_full",
            did_expensive_read=True,
        )

    def read(self, overlay_crop: np.ndarray) -> SequenceReadResult:
        descriptors = extract_square_descriptors(overlay_crop, self.grid)

        if self.board_state is None:
            return self.seed(overlay_crop)

        decision = self.gate.evaluate(descriptors)
        if not decision.should_read:
            self.gate.record_skip()
            self.num_cached_reads += 1
            return SequenceReadResult(
                fen=self.board_state.fen,
                method="overlay_cached",
                did_expensive_read=False,
            )

        next_state, method = self._read_transition(overlay_crop, decision)
        if next_state.fen == self.board_state.fen:
            if decision.changed_indices:
                self.gate.suppress(decision.changed_indices)
                self.num_suppressed_reads += 1
            else:
                self.gate.mark_read(descriptors)
                self.num_full_reads += 1
            return SequenceReadResult(
                fen=self.board_state.fen,
                method="overlay_cached",
                did_expensive_read=True,
            )

        self.board_state = next_state
        self.gate.mark_read(descriptors)
        if method == "overlay_partial":
            self.num_partial_reads += 1
        else:
            self.num_full_reads += 1
        return SequenceReadResult(
            fen=next_state.fen,
            method=method,
            did_expensive_read=True,
        )

    def _read_transition(
        self,
        overlay_crop: np.ndarray,
        decision: GateDecision,
    ) -> tuple[BoardRead, str]:
        assert self.board_state is not None

        if decision.reason == "candidate" and decision.changed_indices:
            partial_state = self._read_partial_state(overlay_crop, decision.changed_indices)
            diff_count = count_fen_differences(self.board_state.fen, partial_state.fen)
            if diff_count == 0 or MIN_MOVE_CHANGED_SQUARES <= diff_count <= MAX_MOVE_CHANGED_SQUARES:
                return partial_state, "overlay_partial"

        full_state = read_board_with_grid(overlay_crop, self.grid, device=self.device)
        if decision.reason == "resync":
            return full_state, "overlay_resync"
        return full_state, "overlay_full"

    def _read_partial_state(
        self,
        overlay_crop: np.ndarray,
        changed_indices: tuple[int, ...],
    ) -> BoardRead:
        assert self.board_state is not None

        crops = [
            _extract_square_crop(overlay_crop, self.grid, row=index // 8, col=index % 8)
            for index in changed_indices
        ]
        preds = classify_square_crops(crops, device=self.device)

        class_grid = [row.copy() for row in self.board_state.class_grid]
        for index, pred in zip(changed_indices, preds):
            row_idx, col_idx = divmod(index, 8)
            class_grid[row_idx][col_idx] = pred

        fen, _ = class_grid_to_fen(
            class_grid,
            flipped=self.board_state.flipped,
            detect_orientation=False,
        )
        return BoardRead(
            fen=fen,
            class_grid=class_grid,
            flipped=self.board_state.flipped,
        )


def _extract_square_crop(frame: np.ndarray, grid: GridResult, *, row: int, col: int) -> np.ndarray:
    y1 = grid.h_lines[row]
    y2 = grid.h_lines[row + 1]
    x1 = grid.v_lines[col]
    x2 = grid.v_lines[col + 1]
    return frame[y1:y2, x1:x2]
