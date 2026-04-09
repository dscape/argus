"""Tests for incremental overlay sequence reading and change gating."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import chess
import cv2
import numpy as np
import pipeline.overlay.sequence_reader as sequence_reader
import pytest
from pipeline.overlay.grid_detector import GridResult
from pipeline.overlay.overlay_move_detector import detect_moves
from pipeline.overlay.piece_classifier import BoardRead, class_grid_to_fen

SQ_SIZE = 16
GRID = GridResult(
    v_lines=list(range(0, SQ_SIZE * 8 + 1, SQ_SIZE)),
    h_lines=list(range(0, SQ_SIZE * 8 + 1, SQ_SIZE)),
    sq_size=SQ_SIZE,
)
MAX_FULL_BOARD_READ_SECONDS = 1.8
MIN_LOCKED_READER_STEADY_STATE_FPS = 1.0
NUM_FULL_BOARD_READ_WARMUPS = 2
NUM_FULL_BOARD_READ_SAMPLES = 3

PIECE_SYMBOL_TO_CLASS = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}


@pytest.fixture(autouse=True)
def _fake_piece_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_board_with_grid(
        frame: np.ndarray,
        _grid: GridResult,
        device: str = "cpu",
        detect_orientation: bool = True,
    ) -> BoardRead:
        del device, detect_orientation
        class_grid = _decode_class_grid(frame)
        fen, _ = class_grid_to_fen(class_grid, detect_orientation=False)
        return BoardRead(fen=fen, class_grid=class_grid, flipped=False)

    def fake_classify_square_crops(
        crops: list[np.ndarray],
        device: str = "cpu",
    ) -> list[int]:
        del device
        return [_decode_square_crop(crop) for crop in crops]

    monkeypatch.setattr(sequence_reader, "read_board_with_grid", fake_read_board_with_grid)
    monkeypatch.setattr(sequence_reader, "classify_square_crops", fake_classify_square_crops)


def test_change_gate_debounces_and_uses_hysteresis() -> None:
    gate = sequence_reader.SquareChangeGate(
        high_threshold=0.30,
        low_threshold=0.18,
        debounce_frames=2,
        resync_interval=99,
    )
    reference = np.zeros((64, 4), dtype=np.float32)
    candidate = reference.copy()
    candidate[12] = 0.40
    candidate[28] = 0.40
    hysteresis_frame = reference.copy()
    hysteresis_frame[12] = 0.20
    hysteresis_frame[28] = 0.20

    gate.mark_read(reference)

    first = gate.evaluate(candidate)
    assert not first.should_read
    assert first.reason == "debouncing"
    gate.record_skip()

    second = gate.evaluate(hysteresis_frame)
    assert second.should_read
    assert second.reason == "candidate"
    assert second.changed_indices == (12, 28)


def test_change_gate_suppresses_repeated_false_positive_masks() -> None:
    gate = sequence_reader.SquareChangeGate(high_threshold=0.30, low_threshold=0.20)
    reference = np.zeros((64, 4), dtype=np.float32)
    candidate = reference.copy()
    candidate[4] = 0.50
    candidate[5] = 0.50

    gate.mark_read(reference)

    first = gate.evaluate(candidate)
    assert first.should_read
    gate.suppress(first.changed_indices)

    second = gate.evaluate(candidate)
    assert not second.should_read
    assert second.reason == "suppressed"

    cleared = gate.evaluate(reference)
    assert not cleared.should_read
    assert cleared.reason == "stable"


def test_change_gate_periodically_resyncs_static_boards() -> None:
    gate = sequence_reader.SquareChangeGate(resync_interval=2)
    reference = np.zeros((64, 4), dtype=np.float32)
    gate.mark_read(reference)

    assert gate.evaluate(reference).reason == "stable"
    gate.record_skip()
    assert gate.evaluate(reference).reason == "stable"
    gate.record_skip()

    decision = gate.evaluate(reference)
    assert decision.should_read
    assert decision.reason == "resync"


def test_sequence_reader_skips_static_frames_and_detects_a_move() -> None:
    board = chess.Board()
    fen_before = board.board_fen()
    board.push(chess.Move.from_uci("e2e4"))
    fen_after = board.board_fen()

    frames = [
        _make_frame(fen_before),
        _make_frame(fen_before),
        _make_frame(fen_before),
        _make_frame(fen_after),
        _make_frame(fen_after),
        _make_frame(fen_after),
    ]

    reader = sequence_reader.LockedOverlaySequenceReader(GRID)
    fens = [reader.read(frame).fen for frame in frames]

    assert reader.num_full_reads == 1
    assert reader.num_partial_reads == 1
    assert reader.num_cached_reads == 4
    assert fens[:3] == [fen_before] * 3
    assert fens[3:] == [fen_after] * 3

    segments = detect_moves(fens, list(range(len(fens))), fps=2.0, stability_window=1)
    assert len(segments) == 1
    assert [move.move_uci for move in segments[0].moves] == ["e2e4"]


@pytest.mark.parametrize(
    ("board_fen", "move_uci"),
    [
        ("4k3/8/8/8/8/8/8/4K2R", "e1g1"),
        ("4k3/8/8/3p4/4P3/8/8/4K3", "e4d5"),
    ],
    ids=["castling", "capture"],
)
def test_sequence_reader_handles_special_legal_moves(board_fen: str, move_uci: str) -> None:
    board = chess.Board(f"{board_fen} w KQkq - 0 1")
    fen_before = board.board_fen()
    board.push(chess.Move.from_uci(move_uci))
    fen_after = board.board_fen()

    reader = sequence_reader.LockedOverlaySequenceReader(GRID)
    fens = [
        reader.read(_make_frame(fen_before)).fen,
        reader.read(_make_frame(fen_before)).fen,
        reader.read(_make_frame(fen_after)).fen,
        reader.read(_make_frame(fen_after)).fen,
    ]

    assert reader.num_full_reads == 1
    assert reader.num_partial_reads == 1

    segments = detect_moves(fens, list(range(len(fens))), fps=2.0, stability_window=1)
    assert len(segments) == 1
    assert [move.move_uci for move in segments[0].moves] == [move_uci]


def test_sequence_reader_avoids_false_moves_from_highlight_animation() -> None:
    fen = chess.STARTING_BOARD_FEN
    frames = [
        _make_frame(fen),
        _make_frame(fen, highlights=((6, 4), (4, 4)), highlight_value=255),
        _make_frame(fen, highlights=((6, 4), (4, 4)), highlight_value=220),
        _make_frame(fen, highlights=((6, 4), (4, 4)), highlight_value=255),
        _make_frame(fen),
    ]

    reader = sequence_reader.LockedOverlaySequenceReader(GRID)
    fens = [reader.read(frame).fen for frame in frames]

    assert fens == [fen] * len(frames)
    assert reader.num_full_reads == 1
    assert reader.num_partial_reads == 0
    assert reader.num_cached_reads == 4

    segments = detect_moves(fens, list(range(len(fens))), fps=2.0, stability_window=1)
    assert segments == []


@pytest.mark.skipif(
    not Path("tests/fixtures/boards/ground_truth.json").exists(),
    reason="No board fixtures yet",
)
def test_locked_overlay_reader_cached_path_exceeds_one_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipeline.overlay.piece_classifier import classify_square_crops, read_board_with_grid

    monkeypatch.setattr(sequence_reader, "read_board_with_grid", read_board_with_grid)
    monkeypatch.setattr(sequence_reader, "classify_square_crops", classify_square_crops)

    ground_truth = _load_board_ground_truth()
    key = next(iter(ground_truth))
    entry = ground_truth[key]
    frame = cv2.imread(str(Path("tests/fixtures/boards") / entry["image"]), cv2.IMREAD_COLOR)
    assert frame is not None

    grid_info = entry["grid"]
    grid = GridResult(
        v_lines=grid_info["v_lines"],
        h_lines=grid_info["h_lines"],
        sq_size=grid_info["v_lines"][1] - grid_info["v_lines"][0],
    )

    reader = sequence_reader.LockedOverlaySequenceReader(grid)
    seeded = reader.read(frame)
    assert seeded.fen == entry["fen"]

    num_frames = 16
    start = time.perf_counter()
    for _ in range(num_frames):
        result = reader.read(frame)
        assert result.fen == entry["fen"]
    elapsed = time.perf_counter() - start
    fps = num_frames / elapsed

    assert fps >= MIN_LOCKED_READER_STEADY_STATE_FPS, (
        f"Locked overlay steady-state throughput was {fps:.2f} fps; "
        f"expected >= {MIN_LOCKED_READER_STEADY_STATE_FPS:.2f} fps"
    )


@pytest.mark.skipif(
    not Path("tests/fixtures/boards/ground_truth.json").exists(),
    reason="No board fixtures yet",
)
def test_full_board_read_microbenchmark() -> None:
    from pipeline.overlay.piece_classifier import read_board_with_grid

    ground_truth = _load_board_ground_truth()
    key = next(iter(ground_truth))
    entry = ground_truth[key]
    frame = cv2.imread(str(Path("tests/fixtures/boards") / entry["image"]), cv2.IMREAD_COLOR)
    assert frame is not None

    grid_info = entry["grid"]
    grid = GridResult(
        v_lines=grid_info["v_lines"],
        h_lines=grid_info["h_lines"],
        sq_size=grid_info["v_lines"][1] - grid_info["v_lines"][0],
    )

    # Torch CPU inference has a noticeable warmup phase for kernel selection and caches.
    # Measure the steady-state median instead of a single noisy post-load sample.
    for _ in range(NUM_FULL_BOARD_READ_WARMUPS):
        warm = read_board_with_grid(frame, grid, detect_orientation=False)
        assert warm.fen == entry["fen"]

    samples: list[float] = []
    for _ in range(NUM_FULL_BOARD_READ_SAMPLES):
        start = time.perf_counter()
        result = read_board_with_grid(frame, grid, detect_orientation=False)
        samples.append(time.perf_counter() - start)
        assert result.fen == entry["fen"]

    elapsed = statistics.median(samples)
    assert elapsed < MAX_FULL_BOARD_READ_SECONDS, (
        f"Median full 64-square board read took {elapsed:.3f}s "
        f"across samples {[round(sample, 3) for sample in samples]}; "
        f"expected < {MAX_FULL_BOARD_READ_SECONDS:.3f}s"
    )


def _make_frame(
    board_fen: str,
    *,
    highlights: tuple[tuple[int, int], ...] = (),
    highlight_value: int = 255,
) -> np.ndarray:
    class_grid = _class_grid_from_board_fen(board_fen)
    frame = np.zeros((SQ_SIZE * 8, SQ_SIZE * 8, 3), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            y1 = row * SQ_SIZE
            y2 = y1 + SQ_SIZE
            x1 = col * SQ_SIZE
            x2 = x1 + SQ_SIZE
            color = 180 if (row + col) % 2 == 0 else 100
            frame[y1:y2, x1:x2] = color

            class_id = class_grid[row][col]
            if class_id:
                center = SQ_SIZE // 2
                pattern = _piece_pattern(class_id)
                patch = np.where(pattern[..., None] == 1, class_id, color).astype(np.uint8)
                frame[
                    y1 + center - 3 : y1 + center + 3,
                    x1 + center - 3 : x1 + center + 3,
                ] = patch
                frame[y1 + 3, x1 + 3] = class_id

    for row, col in highlights:
        y1 = row * SQ_SIZE
        y2 = y1 + SQ_SIZE
        x1 = col * SQ_SIZE
        x2 = x1 + SQ_SIZE
        frame[y1 : y1 + 2, x1:x2] = highlight_value
        frame[y2 - 2 : y2, x1:x2] = highlight_value
        frame[y1:y2, x1 : x1 + 2] = highlight_value
        frame[y1:y2, x2 - 2 : x2] = highlight_value

    return frame


def _class_grid_from_board_fen(board_fen: str) -> list[list[int]]:
    board = chess.Board(f"{board_fen} w - - 0 1")
    class_grid = [[0 for _ in range(8)] for _ in range(8)]

    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece is not None:
                class_grid[row][col] = PIECE_SYMBOL_TO_CLASS[piece.symbol()]

    return class_grid


def _decode_class_grid(frame: np.ndarray) -> list[list[int]]:
    class_grid = [[0 for _ in range(8)] for _ in range(8)]
    for row in range(8):
        for col in range(8):
            value = int(frame[row * SQ_SIZE + 3, col * SQ_SIZE + 3, 0])
            if 1 <= value <= 12:
                class_grid[row][col] = value
    return class_grid


def _decode_square_crop(crop: np.ndarray) -> int:
    value = int(crop[3, 3, 0])
    if 1 <= value <= 12:
        return value
    return 0


def _piece_pattern(class_id: int) -> np.ndarray:
    rng = np.random.default_rng(class_id)
    pattern = rng.integers(0, 2, size=(6, 6), dtype=np.uint8)
    pattern[0, 0] = 0
    return pattern


def _load_board_ground_truth() -> dict[str, dict]:
    import json

    path = Path("tests/fixtures/boards/ground_truth.json")
    with open(path) as f:
        return json.load(f)
