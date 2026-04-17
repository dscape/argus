from __future__ import annotations

from pipeline.physical.piece_detection_assignment import (
    PieceDetection,
    assign_detections_to_squares,
)


def _detection(
    *, piece_label: int, row: int, col: int, up_squares: float, score: float = 0.9
) -> PieceDetection:
    """Synthesize a detection whose LOWER HALF sits exactly on the ``(row, col)`` square.

    The full bbox starts ``up_squares`` squares above that square's top edge, so
    the bottom half of the box is the square tile itself, and the top half is
    the piece's upward extent.
    """
    square_size = 64.0  # 512 / 8
    total_height = (1.0 + up_squares) * square_size
    xmin = col * square_size
    xmax = xmin + square_size
    ymax = (row + 1) * square_size
    ymin = ymax - total_height
    return PieceDetection(
        piece_label=piece_label,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        score=score,
    )


def test_assign_returns_all_empty_when_no_detections() -> None:
    result = assign_detections_to_squares([])
    assert result == tuple([0] * 64)


def test_assign_places_detection_on_its_square_via_lower_half_iou() -> None:
    # piece_label=5 corresponds to white king (SQUARE_CLASS_NAMES index 6).
    detection = _detection(piece_label=5, row=2, col=3, up_squares=2.0)
    result = assign_detections_to_squares([detection])

    expected = [0] * 64
    expected[2 * 8 + 3] = 6
    assert result == tuple(expected)


def test_assign_resolves_collisions_by_score() -> None:
    low = _detection(piece_label=0, row=0, col=0, up_squares=1.0, score=0.2)  # P (class 1)
    high = _detection(piece_label=6, row=0, col=0, up_squares=1.0, score=0.9)  # p (class 7)

    result = assign_detections_to_squares([low, high])

    assert result[0] == 7  # black pawn wins


def test_assign_ignores_detection_below_iou_threshold() -> None:
    # A tiny detection that doesn't overlap the square much.
    tiny = PieceDetection(piece_label=0, xmin=0.0, ymin=0.0, xmax=4.0, ymax=4.0, score=0.9)
    result = assign_detections_to_squares([tiny], iou_threshold=0.3)
    assert result == tuple([0] * 64)


def test_assign_handles_zero_area_detections() -> None:
    zero = PieceDetection(piece_label=0, xmin=10.0, ymin=10.0, xmax=10.0, ymax=10.0, score=0.9)
    result = assign_detections_to_squares([zero])
    assert result == tuple([0] * 64)


def test_assign_places_multiple_detections_on_different_squares() -> None:
    detections = [
        _detection(piece_label=0, row=0, col=0, up_squares=2.0),  # a8 -> class 1 (P)
        _detection(piece_label=11, row=7, col=7, up_squares=0.5),  # h1 -> class 12 (k)
        _detection(piece_label=5, row=4, col=4, up_squares=1.5),  # e4 -> class 6 (K)
    ]
    result = assign_detections_to_squares(detections)
    assert result[0] == 1
    assert result[4 * 8 + 4] == 6
    assert result[7 * 8 + 7] == 12
    # Everything else should remain empty.
    occupied_indices = {0, 4 * 8 + 4, 7 * 8 + 7}
    for i, value in enumerate(result):
        if i not in occupied_indices:
            assert value == 0
