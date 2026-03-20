"""Service layer wrapping pipeline.overlay modules for the overlay tester."""

import base64

import chess
import cv2
import numpy as np


# Singleton readers per theme to avoid rebuilding templates each request
_readers: dict = {}


def _get_reader(theme: str):
    from pipeline.overlay.overlay_reader import OverlayReader

    if theme not in _readers:
        _readers[theme] = OverlayReader(board_theme=theme)
    return _readers[theme]


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def encode_image_b64(img: np.ndarray, fmt: str = ".png") -> str:
    _, buffer = cv2.imencode(fmt, img)
    return base64.b64encode(buffer).decode("utf-8")


def test_image(
    image_bytes: bytes,
    overlay_bbox: tuple[int, int, int, int] | None = None,
    flipped: bool = False,
    theme: str = "lichess_default",
) -> dict:
    """Run overlay detection + reading and return results as a dict."""
    from pipeline.overlay.scanner import detect_overlay_in_frame

    frame = decode_image(image_bytes)
    h, w = frame.shape[:2]

    detection_score = None
    bbox = overlay_bbox

    if bbox is None:
        detection = detect_overlay_in_frame(frame)
        if not detection.found:
            # Return annotated image even on failure
            return {
                "detected": False,
                "bbox": None,
                "detection_score": None,
                "fen": None,
                "piece_count": None,
                "board_ascii": None,
                "annotated_image_b64": encode_image_b64(frame),
                "image_width": w,
                "image_height": h,
            }
        bbox = detection.bbox
        detection_score = detection.score

    x, y, bw, bh = bbox

    # Read board
    overlay_crop = frame[y : y + bh, x : x + bw]
    reader = _get_reader(theme)
    board = reader.read_board(overlay_crop, flipped=flipped)

    # Annotate
    annotated = frame.copy()
    color = (0, 255, 0) if board is not None else (0, 0, 255)
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

    label = "OVERLAY"
    if detection_score is not None:
        label += f" ({detection_score:.2f})"
    cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw grid
    sq_w = bw // 8
    sq_h = bh // 8
    for i in range(1, 8):
        cv2.line(annotated, (x + i * sq_w, y), (x + i * sq_w, y + bh), (0, 255, 255), 1)
        cv2.line(annotated, (x, y + i * sq_h), (x + bw, y + i * sq_h), (0, 255, 255), 1)

    # Piece labels
    if board is not None:
        for row in range(8):
            for col in range(8):
                chess_file = col if not flipped else 7 - col
                chess_rank = (7 - row) if not flipped else row
                sq = chess.square(chess_file, chess_rank)
                piece = board.piece_at(sq)
                if piece is not None:
                    cx = x + col * sq_w + sq_w // 2 - 5
                    cy = y + row * sq_h + sq_h // 2 + 5
                    cv2.putText(
                        annotated, piece.symbol(), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                    )

    # FEN label
    fen = None
    board_ascii = None
    piece_count = None
    if board is not None:
        fen = board.board_fen()
        board_ascii = str(board)
        piece_count = sum(1 for sq in chess.SQUARES if board.piece_at(sq) is not None)
        cv2.putText(
            annotated, f"FEN: {fen}", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return {
        "detected": True,
        "bbox": list(bbox),
        "detection_score": detection_score,
        "fen": fen,
        "piece_count": piece_count,
        "board_ascii": board_ascii,
        "annotated_image_b64": encode_image_b64(annotated),
        "image_width": w,
        "image_height": h,
    }
