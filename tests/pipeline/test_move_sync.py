"""Tests for move synchronization (delay compensation) in overlay_clip_generator."""

import chess
import numpy as np
import pytest

from pipeline.overlay.overlay_clip_generator import OverlayClipGenerator
from pipeline.overlay.overlay_move_detector import GameSegment, OverlayDetectedMove


def _make_camera_crops(num_frames: int, size: int = 100) -> list[np.ndarray]:
    """Create synthetic camera crop images."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(num_frames)]


def _make_segment_with_one_move(move_frame: int, total_frames: int) -> GameSegment:
    """Create a segment with a single e2e4 move at the given frame index."""
    board = chess.Board()
    san = board.san(chess.Move.from_uci("e2e4"))
    return GameSegment(
        moves=[
            OverlayDetectedMove(
                move_index=0,
                move_uci="e2e4",
                move_san=san,
                frame_idx=move_frame,
                timestamp_seconds=move_frame / 30.0,
                fen_before=board.board_fen(),
                fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
            )
        ],
        start_frame=0,
        end_frame=total_frames - 1,
    )


# Try to import argus vocabulary
try:
    from argus.chess.move_vocabulary import get_vocabulary
    VOCAB = get_vocabulary()
    HAS_ARGUS = True
except ImportError:
    VOCAB = None
    HAS_ARGUS = False


class TestMoveDelaySynchronization:
    """Test that move_delay_seconds shifts move timestamps backward."""

    @pytest.fixture
    def generator(self, tmp_path):
        return OverlayClipGenerator(output_dir=str(tmp_path))

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_zero_delay_no_shift(self, generator):
        """With delay=0, move stays at its original frame."""
        total = 30
        move_frame = 15
        segment = _make_segment_with_one_move(move_frame, total)
        frame_indices = list(range(total))
        crops = _make_camera_crops(total)

        clip = generator._build_training_clip(
            camera_crops=crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
            frame_skip=1,
            move_delay_seconds=0.0,
        )

        assert clip is not None
        detect = clip["detect_targets"]
        move_at = (detect == 1.0).nonzero(as_tuple=True)[0].tolist()
        assert move_frame in move_at

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_delay_shifts_backward(self, generator):
        """With delay=2s at 30fps, move should shift backward by 60 frames."""
        total = 100
        move_frame = 80
        segment = _make_segment_with_one_move(move_frame, total)
        frame_indices = list(range(total))
        crops = _make_camera_crops(total)

        clip = generator._build_training_clip(
            camera_crops=crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
            frame_skip=1,
            move_delay_seconds=2.0,
        )

        assert clip is not None
        detect = clip["detect_targets"]
        move_at = (detect == 1.0).nonzero(as_tuple=True)[0].tolist()
        # Should be at frame 80 - 60 = 20
        assert 20 in move_at
        assert move_frame not in move_at

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_delay_clamps_to_start(self, generator):
        """Delay should not produce negative frame indices — clamp to start."""
        total = 30
        move_frame = 5
        segment = _make_segment_with_one_move(move_frame, total)
        frame_indices = list(range(total))
        crops = _make_camera_crops(total)

        clip = generator._build_training_clip(
            camera_crops=crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
            frame_skip=1,
            move_delay_seconds=5.0,  # Would shift 150 frames — way past start
        )

        assert clip is not None
        detect = clip["detect_targets"]
        move_at = (detect == 1.0).nonzero(as_tuple=True)[0].tolist()
        # Should clamp to frame 0 (start_frame)
        assert 0 in move_at

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_frame_skip_affects_delay(self, generator):
        """frame_skip should be accounted for in delay calculation."""
        total = 50
        move_frame = 40
        segment = _make_segment_with_one_move(move_frame, total)
        frame_indices = list(range(total))
        crops = _make_camera_crops(total)

        # 2s delay at 30fps with frame_skip=15 -> shift by int(2*30/15)=4 frames
        clip = generator._build_training_clip(
            camera_crops=crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
            frame_skip=15,
            move_delay_seconds=2.0,
        )

        assert clip is not None
        detect = clip["detect_targets"]
        move_at = (detect == 1.0).nonzero(as_tuple=True)[0].tolist()
        # Should be at frame 40 - 4 = 36
        assert 36 in move_at
