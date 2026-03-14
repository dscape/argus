"""Detect chess moves in video by frame differencing."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MoveDetector:
    """Detect when chess pieces are moved using frame differencing.

    Compares consecutive warped board images and finds peaks in the
    difference signal that correspond to piece movements.
    """

    def __init__(
        self,
        diff_threshold: float = 30.0,
        min_changed_pixels: float = 0.005,
        min_move_gap_seconds: float = 2.0,
        cooldown_frames: int = 5,
    ):
        """
        Args:
            diff_threshold: Pixel intensity difference threshold (0-255).
            min_changed_pixels: Minimum fraction of board pixels that must change.
            min_move_gap_seconds: Minimum time between detected moves.
            cooldown_frames: Frames to skip after detecting a move.
        """
        self.diff_threshold = diff_threshold
        self.min_changed_pixels = min_changed_pixels
        self.min_move_gap_seconds = min_move_gap_seconds
        self.cooldown_frames = cooldown_frames

    def detect_moves(
        self,
        board_frames: list[np.ndarray],
        fps: float,
        start_time: float = 0.0,
    ) -> list[dict]:
        """Detect move timestamps from a sequence of warped board images.

        Args:
            board_frames: List of warped board images (square, same size).
            fps: Frames per second of the source video.
            start_time: Offset in seconds (for chapter-based extraction).

        Returns:
            List of dicts with 'frame_idx', 'timestamp_seconds', 'score'
            for each detected move.
        """
        if len(board_frames) < 2:
            return []

        # Compute per-frame difference scores
        diff_scores = []
        for i in range(1, len(board_frames)):
            score = self._frame_diff_score(board_frames[i - 1], board_frames[i])
            diff_scores.append(score)

        # Find peaks (move events)
        moves = self._find_peaks(diff_scores, fps, start_time)

        return moves

    def detect_camera_cuts(
        self,
        frames: list[np.ndarray],
        cut_threshold: float = 0.3,
    ) -> list[int]:
        """Detect camera cuts (large full-frame differences).

        Useful for multi-board streams where the broadcast switches boards.
        """
        cuts = []
        for i in range(1, len(frames)):
            gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            changed = np.mean(diff > self.diff_threshold)
            if changed > cut_threshold:
                cuts.append(i)
        return cuts

    def _frame_diff_score(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Compute a motion score between two board images.

        Returns fraction of board area that changed significantly.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        diff = cv2.absdiff(gray1, gray2)
        # Apply threshold
        _, binary = cv2.threshold(
            diff, int(self.diff_threshold), 255, cv2.THRESH_BINARY
        )

        # Fraction of pixels that changed
        changed_fraction = np.count_nonzero(binary) / binary.size

        return changed_fraction

    def _find_peaks(
        self,
        scores: list[float],
        fps: float,
        start_time: float,
    ) -> list[dict]:
        """Find peaks in difference scores that correspond to moves.

        Uses a simple threshold + cooldown approach.
        """
        moves = []
        min_gap_frames = int(self.min_move_gap_seconds * fps)
        last_move_frame = -min_gap_frames

        for i, score in enumerate(scores):
            frame_idx = i + 1  # Diff is between frame i and i+1

            if score >= self.min_changed_pixels and (frame_idx - last_move_frame) >= min_gap_frames:
                timestamp = start_time + frame_idx / fps
                moves.append({
                    "frame_idx": frame_idx,
                    "timestamp_seconds": timestamp,
                    "score": score,
                })
                last_move_frame = frame_idx

        return moves

    def get_changed_squares(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        grid_size: int = 8,
    ) -> list[tuple[int, int]]:
        """Determine which board squares changed between two frames.

        Args:
            frame1, frame2: Warped board images (square).
            grid_size: Board grid size (8 for standard chess).

        Returns:
            List of (row, col) tuples for changed squares.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        h, w = gray1.shape
        sq_h = h // grid_size
        sq_w = w // grid_size

        changed = []
        for row in range(grid_size):
            for col in range(grid_size):
                y1, y2 = row * sq_h, (row + 1) * sq_h
                x1, x2 = col * sq_w, (col + 1) * sq_w

                sq1 = gray1[y1:y2, x1:x2]
                sq2 = gray2[y1:y2, x1:x2]

                diff = cv2.absdiff(sq1, sq2)
                if np.mean(diff) > self.diff_threshold:
                    changed.append((row, col))

        return changed
