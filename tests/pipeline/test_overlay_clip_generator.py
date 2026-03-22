"""Tests for pipeline.overlay.overlay_clip_generator."""

import os
import tempfile

import chess
import numpy as np
import pytest
import torch

from pipeline.overlay.overlay_clip_generator import OverlayClipGenerator
from pipeline.overlay.overlay_move_detector import GameSegment, OverlayDetectedMove
from pipeline.overlay.calibration import LayoutCalibration


# Try to import argus vocabulary
try:
    from argus.chess.constraint_mask import get_legal_mask
    from argus.chess.move_vocabulary import get_vocabulary, NO_MOVE_IDX

    VOCAB = get_vocabulary()
    HAS_ARGUS = True
except ImportError:
    VOCAB = None
    NO_MOVE_IDX = None
    HAS_ARGUS = False


def _make_camera_crops(num_frames: int, size: int = 100) -> list[np.ndarray]:
    """Create synthetic camera crop images."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(num_frames)]


def _make_game_segment(
    moves_uci: list[str],
    frames_per_move: int = 5,
    start_frame: int = 0,
) -> tuple[GameSegment, list[int]]:
    """Create a GameSegment from UCI moves with evenly spaced frame indices.

    Returns (segment, all_frame_indices).
    """
    board = chess.Board()
    detected_moves = []
    frame_idx = start_frame

    # Initial position frames
    all_frame_indices = list(range(frame_idx, frame_idx + frames_per_move))
    frame_idx += frames_per_move

    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        fen_before = board.board_fen()
        board.push(move)
        fen_after = board.board_fen()

        detected_moves.append(OverlayDetectedMove(
            move_index=len(detected_moves),
            move_uci=uci,
            move_san=san,
            frame_idx=frame_idx,
            timestamp_seconds=frame_idx / 30.0,
            fen_before=fen_before,
            fen_after=fen_after,
        ))

        new_frames = list(range(frame_idx, frame_idx + frames_per_move))
        all_frame_indices.extend(new_frames)
        frame_idx += frames_per_move

    segment = GameSegment(
        moves=detected_moves,
        start_frame=start_frame,
        end_frame=all_frame_indices[-1],
    )
    return segment, all_frame_indices


# A short Italian Game for testing
ITALIAN_GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8e7"]


class TestBuildTrainingClip:
    """Test _build_training_clip tensor construction."""

    @pytest.fixture
    def generator(self, tmp_path):
        return OverlayClipGenerator(output_dir=str(tmp_path))

    def test_basic_output_structure(self, generator):
        """Output should contain all expected tensor keys."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert "frames" in clip

        if HAS_ARGUS:
            assert "move_targets" in clip
            assert "detect_targets" in clip
            assert "legal_masks" in clip
            assert "move_mask" in clip
        else:
            assert "pgn_moves" in clip
            assert "num_moves" in clip

    def test_frame_tensor_shape(self, generator):
        """Frames tensor should be (T, 3, 224, 224) uint8."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        frames = clip["frames"]
        assert frames.ndim == 4
        assert frames.shape[1] == 3  # C
        assert frames.shape[2] == 224  # H
        assert frames.shape[3] == 224  # W
        assert frames.dtype == torch.uint8

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_move_target_indices_valid(self, generator):
        """Move targets should be valid vocabulary indices."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        targets = clip["move_targets"]
        assert targets.dtype == torch.long
        assert (targets >= 0).all()
        assert (targets < VOCAB.size).all()

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_detect_targets_binary(self, generator):
        """Detect targets should be 0.0 or 1.0."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        detect = clip["detect_targets"]
        unique_vals = set(detect.unique().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_detect_targets_count_matches_moves(self, generator):
        """Number of detection events should match number of moves."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        detect_count = int(clip["detect_targets"].sum().item())
        assert detect_count == len(ITALIAN_GAME)

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_legal_masks_shape(self, generator):
        """Legal masks should be (T, vocab_size) bool."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        masks = clip["legal_masks"]
        assert masks.ndim == 2
        assert masks.shape[1] == VOCAB.size
        assert masks.dtype == torch.bool

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_game_replay_from_clip(self, generator):
        """Replay moves from clip and verify they match the input game."""
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None

        # Extract move indices where detect_targets == 1.0
        detect = clip["detect_targets"]
        targets = clip["move_targets"]

        move_frames = (detect == 1.0).nonzero(as_tuple=True)[0]
        extracted_moves = []
        for frame_t in move_frames:
            idx = targets[frame_t].item()
            if idx != NO_MOVE_IDX:
                uci = VOCAB.index_to_uci(idx)
                extracted_moves.append(uci)

        assert extracted_moves == ITALIAN_GAME

    def test_short_segment_skipped(self, generator):
        """Segments with < 5 frames should return None."""
        segment, frame_indices = _make_game_segment(["e2e4"], frames_per_move=1)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is None

    def test_fallback_format_without_argus(self, generator, monkeypatch):
        """Without argus, clip should have basic format."""
        import pipeline.overlay.overlay_clip_generator as mod

        monkeypatch.setattr(mod, "VOCAB", None)

        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert "frames" in clip
        assert "pgn_moves" in clip
        assert "num_moves" in clip
        assert clip["num_moves"] == len(ITALIAN_GAME)
