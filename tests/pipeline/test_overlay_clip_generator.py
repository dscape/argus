"""Tests for pipeline.overlay.overlay_clip_generator."""

import chess
import numpy as np
import pytest
import torch
from pipeline.overlay.calibration import LayoutCalibration
from pipeline.overlay.overlay_clip_generator import (
    ClipGenerationDiagnostics,
    OverlayClipGenerator,
    generate_from_video,
)
from pipeline.overlay.overlay_move_detector import GameSegment, OverlayDetectedMove

# Try to import argus vocabulary
try:
    from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary

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
    return _make_game_segment_from_board(
        chess.Board(),
        moves_uci,
        frames_per_move=frames_per_move,
        start_frame=start_frame,
    )


def _make_game_segment_from_board(
    board: chess.Board,
    moves_uci: list[str],
    frames_per_move: int = 5,
    start_frame: int = 0,
) -> tuple[GameSegment, list[int]]:
    detected_moves = []
    frame_idx = start_frame

    all_frame_indices = list(range(frame_idx, frame_idx + frames_per_move))
    frame_idx += frames_per_move

    working_board = board.copy()
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        san = working_board.san(move)
        fen_before = working_board.board_fen()
        working_board.push(move)
        fen_after = working_board.board_fen()

        detected_moves.append(
            OverlayDetectedMove(
                move_index=len(detected_moves),
                move_uci=uci,
                move_san=san,
                frame_idx=frame_idx,
                timestamp_seconds=frame_idx / 30.0,
                fen_before=fen_before,
                fen_after=fen_after,
            )
        )

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

    def test_clip_includes_pgn_and_timing_metadata(self, generator):
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert clip["pgn_moves"] == segment.pgn_moves
        assert clip["move_ucis"] == ITALIAN_GAME
        assert clip["move_sans"] == [move.move_san for move in segment.moves]
        assert clip["num_moves"] == len(ITALIAN_GAME)
        assert clip["training_target_timing"] == "overlay_confirm_post_move"
        assert clip["estimated_otb_delay_seconds"] == 0.0
        assert torch.equal(clip["frame_indices"], torch.tensor(frame_indices, dtype=torch.long))
        assert torch.allclose(
            clip["frame_timestamps_seconds"],
            torch.tensor([idx / 30.0 for idx in frame_indices], dtype=torch.float32),
        )
        assert torch.equal(
            clip["move_frame_indices"],
            torch.tensor([move.frame_idx for move in segment.moves], dtype=torch.long),
        )
        assert torch.allclose(
            clip["move_timestamps_seconds"],
            torch.tensor([move.frame_idx / 30.0 for move in segment.moves], dtype=torch.float32),
        )

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
    def test_move_mask_marks_only_move_frames(self, generator):
        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert clip["move_mask"].dtype == torch.bool
        assert torch.equal(clip["move_mask"], clip["detect_targets"].bool())
        assert int(clip["move_mask"].sum().item()) == len(ITALIAN_GAME)

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

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_midgame_segment_uses_first_move_fen_before(self, generator):
        board = chess.Board()
        for uci in ["e2e4", "e7e5"]:
            board.push(chess.Move.from_uci(uci))

        segment, frame_indices = _make_game_segment_from_board(
            board,
            ["g1f3", "b8c6"],
            frames_per_move=3,
        )
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert clip["initial_board_fen"] == board.board_fen()

        detect = clip["detect_targets"]
        targets = clip["move_targets"]
        move_frames = (detect == 1.0).nonzero(as_tuple=True)[0]
        extracted_moves = []
        for frame_t in move_frames:
            idx = targets[frame_t].item()
            if idx != NO_MOVE_IDX:
                extracted_moves.append(VOCAB.index_to_uci(idx))

        assert extracted_moves == ["g1f3", "b8c6"]

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_midgame_segment_infers_black_to_move_from_first_move(self, generator):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))

        segment, frame_indices = _make_game_segment_from_board(
            board,
            ["e7e5", "g1f3"],
            frames_per_move=3,
        )
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert clip["initial_board_fen"] == board.board_fen()
        assert clip["initial_side_to_move"] == "b"

        detect = clip["detect_targets"]
        targets = clip["move_targets"]
        move_frames = (detect == 1.0).nonzero(as_tuple=True)[0]
        extracted_moves = []
        for frame_t in move_frames:
            idx = targets[frame_t].item()
            if idx != NO_MOVE_IDX:
                extracted_moves.append(VOCAB.index_to_uci(idx))

        assert extracted_moves == ["e7e5", "g1f3"]

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_build_training_clip_keeps_one_pre_move_frame_when_available(self, generator):
        segment, frame_indices = _make_game_segment(["e2e4", "e7e5"], frames_per_move=3)
        segment.start_frame = segment.moves[0].frame_idx
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert clip["frame_indices"][0].item() == segment.moves[0].frame_idx - 1
        move_frames = (clip["detect_targets"] == 1.0).nonzero(as_tuple=True)[0]
        assert move_frames.tolist()[0] == 1

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_build_training_clip_keeps_pre_move_frame_when_delay_metadata_exists(self, generator):
        segment, frame_indices = _make_game_segment(["e2e4", "e7e5"], frames_per_move=3)
        segment.start_frame = frame_indices[2]
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=3.0,
            frame_skip=1,
            move_delay_seconds=1.0,
        )

        assert clip is not None
        assert clip["frame_indices"][0].item() == frame_indices[2]
        move_frames = (clip["detect_targets"] == 1.0).nonzero(as_tuple=True)[0]
        assert move_frames.tolist()[0] == 1
        assert clip["move_frame_indices"][0].item() == frame_indices[3]
        assert clip["estimated_otb_frame_indices"][0].item() == frame_indices[2]

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

    def test_custom_min_moves_allows_short_legal_segment(self, tmp_path):
        generator = OverlayClipGenerator(output_dir=str(tmp_path), min_moves_per_segment=1)
        segment, frame_indices = _make_game_segment(["e2e4"], frames_per_move=5)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        assert clip["num_moves"] == 1

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_illegal_segment_skipped(self, generator):
        segment, frame_indices = _make_game_segment(["e1g1"], frames_per_move=5)
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is None

    def test_clip_stores_estimated_otb_timing_separately_from_targets(self, generator):
        segment, frame_indices = _make_game_segment(["e2e4", "e7e5"], frames_per_move=4)
        frame_indices = [frame_idx * 2 for frame_idx in frame_indices]
        for move in segment.moves:
            move.frame_idx *= 2
            move.timestamp_seconds = move.frame_idx / 8.0
        segment.end_frame = frame_indices[-1]
        camera_crops = _make_camera_crops(len(frame_indices))

        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=8.0,
            frame_skip=2,
            move_delay_seconds=1.0,
        )

        assert clip is not None
        assert torch.equal(
            clip["move_frame_indices"],
            torch.tensor([8, 16], dtype=torch.long),
        )
        assert torch.allclose(
            clip["move_timestamps_seconds"],
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        assert torch.equal(
            clip["estimated_otb_frame_indices"],
            torch.tensor([0, 8], dtype=torch.long),
        )
        assert torch.allclose(
            clip["estimated_otb_timestamps_seconds"],
            torch.tensor([0.0, 1.0], dtype=torch.float32),
        )

    @pytest.mark.skipif(not HAS_ARGUS, reason="argus package not installed")
    def test_saved_clip_prefix_loads_with_argus_dataset(self, generator, tmp_path):
        from argus.data.dataset import ArgusDataset

        segment, frame_indices = _make_game_segment(ITALIAN_GAME, frames_per_move=3)
        camera_crops = _make_camera_crops(len(frame_indices))
        clip = generator._build_training_clip(
            camera_crops=camera_crops,
            frame_indices=frame_indices,
            segment=segment,
            fps=30.0,
        )

        assert clip is not None
        clip_path = tmp_path / "clip_overlay_demo_0.pt"
        torch.save(clip, clip_path)

        dataset = ArgusDataset(tmp_path, clip_length=clip["frames"].shape[0])
        assert len(dataset) == 1

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


class TestGenerateFromVideo:
    def test_dry_run_collects_diagnostics_without_writing(self, monkeypatch, tmp_path):
        import pipeline.overlay.overlay_clip_generator as mod

        monkeypatch.setattr(mod, "_get_db_clips", lambda video_id: [])
        monkeypatch.setattr(
            mod,
            "get_calibration",
            lambda channel_handle: LayoutCalibration(
                overlay=(0, 0, 100, 100),
                camera=(0, 0, 100, 100),
                ref_resolution=(1920, 1080),
                board_flipped=False,
                board_theme="lichess_default",
            ),
        )

        calls = []

        def fake_generate_clips(
            self,
            video_path,
            calibration,
            video_id="",
            start_time=None,
            end_time=None,
            channel_handle=None,
            save_clips=True,
            diagnostics=None,
        ):
            calls.append((video_path, video_id, channel_handle, save_clips))
            assert diagnostics is not None
            diagnostics.sampled_frame_count = 12
            diagnostics.readable_fen_count = 12
            diagnostics.saved_clip_move_counts.append(6)
            return [{"filepath": "clip_overlay_demo123_0.pt", "num_moves": 6, "num_frames": 42}]

        monkeypatch.setattr(OverlayClipGenerator, "generate_clips", fake_generate_clips)

        diagnostics: list[ClipGenerationDiagnostics] = []
        results = generate_from_video(
            "/tmp/demo123.mp4",
            channel_handle="@demo",
            output_dir=str(tmp_path),
            min_moves_per_segment=5,
            save_clips=False,
            diagnostics=diagnostics,
        )

        assert results == [
            {
                "filepath": "clip_overlay_demo123_0.pt",
                "num_moves": 6,
                "num_frames": 42,
            }
        ]
        assert calls == [("/tmp/demo123.mp4", "demo123", "@demo", False)]
        assert len(diagnostics) == 1
        assert diagnostics[0].clip_label == "demo123"
        assert diagnostics[0].sampled_frame_count == 12
        assert diagnostics[0].saved_clip_move_counts == [6]
