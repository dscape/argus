"""Detect chess board region in video frames using OpenCV."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BoardDetector:
    """Detect and extract the chess board region from a video frame.

    Uses a combination of edge detection, line detection, and
    contour analysis to find the board.
    """

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def detect(self, frame: np.ndarray) -> dict | None:
        """Detect the chess board in a frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Dict with 'corners' (4x2 array), 'bbox' (x, y, w, h), and
            'transform' (3x3 perspective matrix), or None if not found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try multiple approaches
        result = self._detect_via_contours(gray, frame)
        if result is not None:
            return result

        result = self._detect_via_lines(gray, frame)
        if result is not None:
            return result

        return None

    def warp_board(
        self,
        frame: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """Warp the detected board region to a square image.

        Args:
            frame: Original BGR frame.
            transform: 3x3 perspective transform matrix.

        Returns:
            Square image (target_size x target_size x 3).
        """
        return cv2.warpPerspective(
            frame, transform, (self.target_size, self.target_size)
        )

    def _detect_via_contours(
        self, gray: np.ndarray, frame: np.ndarray
    ) -> dict | None:
        """Detect board by finding the largest square-like contour."""
        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        h, w = gray.shape
        frame_area = h * w
        best = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            # Board should be a significant portion of the frame
            if area < frame_area * 0.05:
                continue

            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Should be roughly a quadrilateral
            if len(approx) == 4 and area > best_area:
                # Check aspect ratio (board should be roughly square)
                rect = cv2.minAreaRect(contour)
                w_rect, h_rect = rect[1]
                if w_rect == 0 or h_rect == 0:
                    continue
                aspect = min(w_rect, h_rect) / max(w_rect, h_rect)
                if aspect > 0.6:  # Roughly square
                    best = approx
                    best_area = area

        if best is None:
            return None

        corners = best.reshape(4, 2).astype(np.float32)
        corners = self._order_corners(corners)

        dst = np.array([
            [0, 0],
            [self.target_size - 1, 0],
            [self.target_size - 1, self.target_size - 1],
            [0, self.target_size - 1],
        ], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(corners, dst)
        x, y, bw, bh = cv2.boundingRect(corners.astype(np.int32))

        return {
            "corners": corners,
            "bbox": (x, y, bw, bh),
            "transform": transform,
        }

    def _detect_via_lines(
        self, gray: np.ndarray, frame: np.ndarray
    ) -> dict | None:
        """Detect board by finding a grid of lines (Hough transform)."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100,
            minLineLength=100, maxLineGap=10
        )

        if lines is None or len(lines) < 8:
            return None

        # Separate horizontal and vertical lines
        horizontal = []
        vertical = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1))
            if angle < np.pi / 6:
                horizontal.append(line[0])
            elif angle > np.pi / 3:
                vertical.append(line[0])

        if len(horizontal) < 4 or len(vertical) < 4:
            return None

        # Find bounding box from line endpoints
        all_points = np.array(horizontal + vertical)
        x_min = all_points[:, [0, 2]].min()
        x_max = all_points[:, [0, 2]].max()
        y_min = all_points[:, [1, 3]].min()
        y_max = all_points[:, [1, 3]].max()

        corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ], dtype=np.float32)

        dst = np.array([
            [0, 0],
            [self.target_size - 1, 0],
            [self.target_size - 1, self.target_size - 1],
            [0, self.target_size - 1],
        ], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(corners, dst)

        return {
            "corners": corners,
            "bbox": (int(x_min), int(y_min),
                     int(x_max - x_min), int(y_max - y_min)),
            "transform": transform,
        }

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y coordinate
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        # Top two points
        top = sorted_by_y[:2]
        bottom = sorted_by_y[2:]
        # Sort top by x
        top = top[np.argsort(top[:, 0])]
        # Sort bottom by x
        bottom = bottom[np.argsort(bottom[:, 0])]

        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
