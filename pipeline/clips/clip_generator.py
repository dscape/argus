"""Generate training clips from matched videos + PGNs."""

import io
import json
import logging
import os
from pathlib import Path

import chess
import chess.pgn
import cv2
import numpy as np
import torch

from pipeline.clips.board_detector import BoardDetector
from pipeline.clips.move_detector import MoveDetector
from pipeline.clips.pgn_aligner import align_pgn_to_detections
from pipeline.db.connection import get_conn
from pipeline.download.video_downloader import get_video_path

logger = logging.getLogger(__name__)

# Import from argus for training format compatibility
try:
    from argus.chess.move_vocabulary import get_vocabulary
    from argus.chess.constraint_mask import get_legal_mask

    VOCAB = get_vocabulary()
except ImportError:
    logger.warning("argus package not installed. Training clip generation will use basic format.")
    VOCAB = None

OUTPUT_DIR = os.path.join("data", "training_clips")
FRAME_SIZE = 224  # Match argus training format


class ClipGenerator:
    """Generate training clips from video + PGN pairs."""

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        base_fps: float = 2.0,
        move_fps: float = 10.0,
        move_window_seconds: float = 2.0,
    ):
        """
        Args:
            output_dir: Directory for output .pt files.
            base_fps: Frame extraction rate during non-move periods.
            move_fps: Frame extraction rate around detected moves.
            move_window_seconds: Seconds around a move to use higher FPS.
        """
        self.output_dir = output_dir
        self.base_fps = base_fps
        self.move_fps = move_fps
        self.move_window_seconds = move_window_seconds
        self.board_detector = BoardDetector(target_size=512)
        self.move_detector = MoveDetector()

    def generate_clip(
        self,
        video_path: str,
        pgn_moves: str,
        video_id: str,
        game_id: int,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
    ) -> dict | None:
        """Generate a training clip from a video + PGN.

        Args:
            video_path: Path to the downloaded MP4.
            pgn_moves: PGN movetext.
            video_id: YouTube video ID.
            game_id: Database game ID.
            start_seconds: Start offset (from chapter timestamp).
            end_seconds: End offset (next chapter or end of video).

        Returns:
            Dict with clip metadata, or None if generation failed.
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            logger.error(f"Invalid FPS for {video_path}")
            cap.release()
            return None

        # Seek to start
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps) if end_seconds else total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read reference frame for board detection
        ret, ref_frame = cap.read()
        if not ret:
            cap.release()
            return None

        board_info = self.board_detector.detect(ref_frame)
        if board_info is None:
            logger.warning(f"Board detection failed for {video_id}")
            cap.release()
            return None

        transform = board_info["transform"]

        # Extract board frames at base FPS
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        board_frames = []
        frame_indices = []
        frame_skip = max(1, int(fps / self.base_fps))

        current_frame = start_frame
        while current_frame < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            warped = self.board_detector.warp_board(frame, transform)
            board_frames.append(warped)
            frame_indices.append(current_frame)
            current_frame += frame_skip

        cap.release()

        if len(board_frames) < 10:
            logger.warning(f"Too few frames extracted for {video_id}")
            return None

        # Detect moves via frame differencing
        detected_moves = self.move_detector.detect_moves(
            board_frames, self.base_fps, start_seconds
        )

        # Align with PGN
        alignment = align_pgn_to_detections(
            pgn_moves, detected_moves, fps
        )

        if alignment.quality < 0.3:
            logger.warning(
                f"Low alignment quality ({alignment.quality:.2f}) for "
                f"video={video_id}, game={game_id}"
            )

        # Build training clip
        clip_data = self._build_training_clip(
            board_frames, frame_indices, alignment, pgn_moves, fps
        )

        if clip_data is None:
            return None

        # Save
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{video_id}_{game_id}.pt"
        filepath = os.path.join(self.output_dir, filename)
        torch.save(clip_data, filepath)

        # Record in database
        self._save_to_db(
            game_id, video_id, filepath,
            len(board_frames), alignment.total_pgn_moves,
            alignment.quality, alignment.error_count > 0,
        )

        return {
            "filepath": filepath,
            "num_frames": len(board_frames),
            "num_moves": alignment.total_pgn_moves,
            "aligned_moves": alignment.aligned_count,
            "quality": alignment.quality,
            "has_errors": alignment.error_count > 0,
        }

    def _build_training_clip(
        self,
        board_frames: list[np.ndarray],
        frame_indices: list[int],
        alignment,
        pgn_moves: str,
        fps: float,
    ) -> dict | None:
        """Build a training clip dict compatible with ArgusDataset."""
        num_frames = len(board_frames)

        # Resize frames to training size
        resized = []
        for frame in board_frames:
            resized_frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            # Convert BGR to RGB and to tensor format (C, H, W)
            rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            resized.append(rgb)

        frames_tensor = torch.from_numpy(
            np.stack(resized)
        ).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Build move targets
        if VOCAB is None:
            # Basic format without argus vocab
            return {
                "frames": frames_tensor.to(torch.uint8),
                "pgn_moves": pgn_moves,
                "alignment_quality": alignment.quality,
                "detected_moves": len(alignment.moves),
            }

        no_move_idx = VOCAB.NO_MOVE_IDX
        vocab_size = VOCAB.size

        move_targets = torch.full((num_frames,), no_move_idx, dtype=torch.long)
        detect_targets = torch.zeros(num_frames, dtype=torch.float32)
        move_mask = torch.ones(num_frames, dtype=torch.float32)

        # Build legal masks by replaying the game
        legal_masks = torch.zeros(num_frames, vocab_size, dtype=torch.bool)
        game = chess.pgn.read_game(io.StringIO(pgn_moves))
        if game is None:
            return None

        board = game.board()
        current_move_idx = 0
        move_node = game
        move_list = list(game.mainline_moves())

        # Create a map from aligned frame indices to move info
        move_frame_map = {}
        for am in alignment.moves:
            move_frame_map[am.frame_idx] = am

        for i, frame_idx in enumerate(frame_indices):
            # Get legal mask for current position
            legal_mask = get_legal_mask(board)
            legal_masks[i] = legal_mask

            # Check if this frame has an aligned move
            if frame_idx in move_frame_map:
                am = move_frame_map[frame_idx]
                if current_move_idx < len(move_list):
                    move = move_list[current_move_idx]
                    uci = move.uci()
                    idx = VOCAB.move_to_index.get(uci)
                    if idx is not None:
                        move_targets[i] = idx
                        detect_targets[i] = 1.0
                    board.push(move)
                    current_move_idx += 1

        return {
            "frames": frames_tensor.to(torch.uint8),
            "move_targets": move_targets,
            "detect_targets": detect_targets,
            "legal_masks": legal_masks,
            "move_mask": move_mask,
        }

    def _save_to_db(
        self,
        game_id: int,
        video_id: str,
        filepath: str,
        num_frames: int,
        num_moves: int,
        quality: float,
        has_errors: bool,
    ):
        """Save clip metadata to the training_clips table."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO training_clips
                        (game_id, video_id, file_path, num_frames,
                         num_moves, alignment_quality, has_errors)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (game_id, video_id, filepath, num_frames,
                     num_moves, quality, has_errors),
                )
                conn.commit()


def generate_all_clips(
    min_confidence: float = 70.0,
    limit: int | None = None,
    output_dir: str = OUTPUT_DIR,
):
    """Generate training clips for all matched and downloaded games."""
    generator = ClipGenerator(output_dir=output_dir)

    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT gvl.game_id, gvl.video_id, gvl.timestamp_seconds,
                       g.pgn_moves, v.channel_handle
                FROM game_video_links gvl
                JOIN games g ON g.id = gvl.game_id
                JOIN youtube_videos v ON v.video_id = gvl.video_id
                WHERE gvl.match_confidence >= %s
                  AND NOT EXISTS (
                      SELECT 1 FROM training_clips tc
                      WHERE tc.game_id = gvl.game_id
                        AND tc.video_id = gvl.video_id
                  )
                ORDER BY gvl.match_confidence DESC
            """
            params = [min_confidence]

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            matches = cur.fetchall()

    if not matches:
        print("No matched games ready for clip generation.")
        return

    print(f"Generating clips for {len(matches)} game-video matches...")
    generated = 0
    skipped = 0
    failed = 0

    for game_id, video_id, timestamp, pgn_moves, channel_handle in matches:
        video_path = get_video_path(video_id, channel_handle)
        if video_path is None:
            skipped += 1
            continue

        result = generator.generate_clip(
            video_path=video_path,
            pgn_moves=pgn_moves,
            video_id=video_id,
            game_id=game_id,
            start_seconds=float(timestamp or 0),
        )

        if result:
            generated += 1
            logger.info(
                f"Generated clip: game={game_id}, quality={result['quality']:.2f}, "
                f"moves={result['aligned_moves']}/{result['num_moves']}"
            )
        else:
            failed += 1

    print(f"\nClip generation: {generated} generated, {skipped} skipped (not downloaded), {failed} failed")
