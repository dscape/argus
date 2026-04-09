"""Generate training clips from videos with 2D board overlays.

Combines OTB camera crops (training frames) with overlay-extracted ground truth
(moves) to produce .pt training clips compatible with ArgusDataset.
"""

import logging
import os
import subprocess

import chess
import chess.pgn
import cv2
import numpy as np
import torch

from pipeline.overlay.calibration import LayoutCalibration, get_calibration
from pipeline.overlay.grid_detector import detect_grid
from pipeline.overlay.overlay_move_detector import GameSegment, detect_moves
from pipeline.overlay.sequence_reader import LockedOverlaySequenceReader

logger = logging.getLogger(__name__)

# Import from argus for training format compatibility
try:
    from argus.chess.constraint_mask import get_legal_mask
    from argus.chess.move_vocabulary import get_vocabulary

    VOCAB = get_vocabulary()
except ImportError:
    logger.warning("argus package not installed. Clip generation will use basic format.")
    VOCAB = None

OUTPUT_DIR = os.path.join("data", "argus", "training_clips")
FRAME_SIZE = 224


class OverlayClipGenerator:
    """Generate training clips from overlay videos."""

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        base_fps: float = 2.0,
        device: str = "cpu",
        min_moves_per_segment: int = 5,
    ):
        self.output_dir = output_dir
        self.base_fps = base_fps
        self.device = device
        self.min_moves_per_segment = min_moves_per_segment

    def generate_clips(
        self,
        video_path: str,
        calibration: LayoutCalibration,
        video_id: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict]:
        """Generate training clips from a video with 2D overlay.

        Args:
            video_path: Path to the video file.
            calibration: Layout calibration with overlay/camera crop coordinates.
            video_id: Identifier for naming output files.
            start_time: If set, start processing from this time (seconds).
            end_time: If set, stop processing at this time (seconds).

        Returns:
            List of dicts with clip metadata for each generated clip.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or total_frames <= 0:
            logger.error(f"Invalid video properties: fps={fps}, frames={total_frames}")
            cap.release()
            return []

        # Scale calibration to actual video resolution
        cal = calibration.scale_to_resolution(width, height)

        # Extract frames
        frame_skip = max(1, int(fps / self.base_fps))
        overlay_crops = []
        camera_crops = []
        frame_indices = []
        fens = []

        # Compute frame range from time bounds
        first_frame = int(start_time * fps) if start_time is not None else 0
        last_frame = int(end_time * fps) if end_time is not None else total_frames
        first_frame = max(0, first_frame)
        last_frame = min(total_frames, last_frame)
        clip_start_time = first_frame / fps if fps > 0 else 0.0

        logger.info(
            f"Processing {video_path}: frames {first_frame}-{last_frame} at {fps:.1f} FPS, "
            f"sampling every {frame_skip} frames"
        )

        sequence_reader: LockedOverlaySequenceReader | None = None

        current_frame = first_frame
        while current_frame < last_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            # Crop overlay region
            ox, oy, ow, oh = cal.overlay
            overlay_crop = frame[oy : oy + oh, ox : ox + ow]

            # Crop camera region
            cx, cy, cw, ch = cal.camera
            camera_crop = frame[cy : cy + ch, cx : cx + cw]

            # Lock grid once, then use cheap per-square gating plus partial reads.
            fen: str | None = None
            if sequence_reader is None:
                grid = detect_grid(overlay_crop)
                if grid is not None:
                    sequence_reader = LockedOverlaySequenceReader(grid, device=self.device)
                    fen = sequence_reader.read(overlay_crop).fen
            else:
                fen = sequence_reader.read(overlay_crop).fen

            overlay_crops.append(overlay_crop)
            camera_crops.append(camera_crop)
            frame_indices.append(current_frame)
            fens.append(fen)

            current_frame += frame_skip

        cap.release()

        if len(fens) < 10:
            logger.warning(f"Too few frames extracted: {len(fens)}")
            return []

        readable = sum(1 for f in fens if f is not None)
        logger.info(f"Extracted {len(fens)} frames, {readable} readable FENs")
        if sequence_reader is not None:
            logger.info(
                "Overlay read stats: %d full, %d partial, %d cached, %d suppressed",
                sequence_reader.num_full_reads,
                sequence_reader.num_partial_reads,
                sequence_reader.num_cached_reads,
                sequence_reader.num_suppressed_reads,
            )

        # Detect moves from FEN sequence
        segments = detect_moves(
            fens=fens,
            frame_indices=frame_indices,
            fps=fps,
            start_time=clip_start_time,
        )

        if not segments:
            logger.warning("No game segments detected")
            return []

        logger.info(f"Detected {len(segments)} game segment(s)")

        # Generate a .pt clip for each game segment
        results = []
        os.makedirs(self.output_dir, exist_ok=True)

        for game_idx, segment in enumerate(segments):
            if segment.num_moves < self.min_moves_per_segment:
                logger.info(f"Skipping segment {game_idx}: only {segment.num_moves} moves")
                continue

            clip_data = self._build_training_clip(
                camera_crops=camera_crops,
                frame_indices=frame_indices,
                segment=segment,
                fps=fps,
                frame_skip=frame_skip,
                move_delay_seconds=cal.move_delay_seconds,
            )

            if clip_data is None:
                continue

            # Generate video_id from path if not provided
            vid = video_id or os.path.splitext(os.path.basename(video_path))[0]
            filename = f"clip_overlay_{vid}_{game_idx}.pt"
            filepath = os.path.join(self.output_dir, filename)

            torch.save(clip_data, filepath)
            logger.info(
                f"Saved clip: {filename} ({segment.num_moves} moves, "
                f"{clip_data['frames'].shape[0]} frames)"
            )

            results.append({
                "filepath": filepath,
                "num_frames": clip_data["frames"].shape[0],
                "num_moves": segment.num_moves,
                "game_index": game_idx,
                "pgn_moves": segment.pgn_moves,
            })

        return results

    def _build_training_clip(
        self,
        camera_crops: list[np.ndarray],
        frame_indices: list[int],
        segment: GameSegment,
        fps: float,
        frame_skip: int = 1,
        move_delay_seconds: float = 0.0,
    ) -> dict | None:
        """Build a training clip dict compatible with ArgusDataset.

        Args:
            camera_crops: All camera crops from the video.
            frame_indices: Frame index for each crop.
            segment: Detected game segment with moves.
            fps: Video FPS.
            frame_skip: Frames skipped between samples (for delay calculation).
            move_delay_seconds: Broadcast delay — shift move timestamps backward
                by this many seconds to align with the OTB camera moment the
                move was actually played (overlay updates after OTB).

        Returns:
            Dict with frames, move_targets, detect_targets, legal_masks, move_mask.
        """
        # Find the frame range for this segment
        start_idx = None
        end_idx = None
        for i, fidx in enumerate(frame_indices):
            if fidx >= segment.start_frame and start_idx is None:
                start_idx = i
            if fidx <= segment.end_frame:
                end_idx = i

        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return None

        segment_cameras = camera_crops[start_idx : end_idx + 1]
        segment_frame_indices = frame_indices[start_idx : end_idx + 1]
        num_frames = len(segment_cameras)

        if num_frames < 5:
            return None

        # Resize camera frames to training size
        resized = []
        for crop in segment_cameras:
            r = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))
            rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
            resized.append(rgb)

        frames_tensor = torch.from_numpy(
            np.stack(resized)
        ).permute(0, 3, 1, 2).to(torch.uint8)  # (T, C, H, W)

        delay_sample_steps = int(move_delay_seconds * fps / frame_skip) if move_delay_seconds > 0 else 0
        delay_frame_offset = delay_sample_steps * frame_skip
        frame_timestamps = torch.tensor(
            [frame_idx / fps for frame_idx in segment_frame_indices],
            dtype=torch.float32,
        )

        adjusted_move_frames = [
            max(move.frame_idx - delay_frame_offset, segment.start_frame)
            for move in segment.moves
        ]
        move_timestamps = torch.tensor(
            [frame_idx / fps for frame_idx in adjusted_move_frames],
            dtype=torch.float32,
        )
        initial_board_fen = segment.moves[0].fen_before if segment.moves else chess.STARTING_BOARD_FEN
        clip_metadata = {
            "initial_board_fen": initial_board_fen,
            "pgn_moves": segment.pgn_moves,
            "num_moves": segment.num_moves,
            "move_ucis": [move.move_uci for move in segment.moves],
            "move_sans": [move.move_san for move in segment.moves],
            "frame_indices": torch.tensor(segment_frame_indices, dtype=torch.long),
            "frame_timestamps_seconds": frame_timestamps,
            "move_frame_indices": torch.tensor(adjusted_move_frames, dtype=torch.long),
            "move_timestamps_seconds": move_timestamps,
            "segment_start_time_seconds": float(segment.start_time),
            "segment_end_time_seconds": float(segment.end_time),
        }

        if VOCAB is None:
            return {
                "frames": frames_tensor,
                **clip_metadata,
            }

        from argus.chess.move_vocabulary import NO_MOVE_IDX

        no_move_idx = NO_MOVE_IDX
        vocab_size = VOCAB.size

        move_targets = torch.full((num_frames,), no_move_idx, dtype=torch.long)
        detect_targets = torch.zeros(num_frames, dtype=torch.float32)
        legal_masks = torch.zeros(num_frames, vocab_size, dtype=torch.bool)
        move_mask = torch.ones(num_frames, dtype=torch.float32)
        move_confidence = torch.ones(num_frames, dtype=torch.float32)

        # Build a map from frame index to move.
        move_frame_map = {}
        for adjusted_idx, move in zip(adjusted_move_frames, segment.moves):
            move_frame_map[adjusted_idx] = move

        # Replay the game to generate legal masks
        board = chess.Board()
        board.set_board_fen(initial_board_fen)

        for i, frame_idx in enumerate(segment_frame_indices):
            # Generate legal mask for current position
            legal_mask = get_legal_mask(board)
            legal_masks[i] = legal_mask

            # Check if this frame has a move
            if frame_idx in move_frame_map:
                m = move_frame_map[frame_idx]
                uci = m.move_uci
                idx = VOCAB.uci_to_index(uci) if VOCAB.contains(uci) else None
                if idx is None:
                    logger.warning("Move %s missing from vocabulary at frame %d", uci, frame_idx)
                    return None

                move_targets[i] = idx
                detect_targets[i] = 1.0
                move_confidence[i] = m.confidence

                # Push the move on the board
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    logger.warning("Invalid UCI move: %s", uci)
                    return None

                if move not in board.legal_moves:
                    logger.warning("Move %s not legal at frame %d", uci, frame_idx)
                    return None

                board.push(move)

        return {
            "frames": frames_tensor,
            "move_targets": move_targets,
            "detect_targets": detect_targets,
            "legal_masks": legal_masks,
            "move_mask": move_mask,
            "move_confidence": move_confidence,
            **clip_metadata,
        }


def download_video(url: str, output_dir: str = "data/videos/overlay") -> str | None:
    """Download a video via yt-dlp. Returns the output file path."""
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "best[ext=mp4]/best",
                "-o", output_template,
                "--no-warnings",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            return None

        # Find the downloaded file
        for line in result.stdout.splitlines():
            if "Destination:" in line:
                return line.split("Destination:")[-1].strip()
            if "has already been downloaded" in line:
                # Extract path from "... already been downloaded and merged"
                parts = line.split("[download]")[-1].strip()
                path = parts.split(" has already")[0].strip()
                if os.path.exists(path):
                    return path

        # Fallback: look for the file
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                return os.path.join(output_dir, f)

        return None

    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out for {url}")
        return None


def _get_db_clips(video_id: str) -> list[dict]:
    """Query per-clip calibrations from the video_clips DB table."""
    try:
        from pipeline.db.connection import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, start_time, end_time,
                           overlay_bbox, camera_bbox, ref_resolution,
                           board_flipped, board_theme
                    FROM video_clips
                    WHERE video_id = %s
                    ORDER BY clip_index
                    """,
                    (video_id,),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.debug(f"Could not read video_clips from DB: {e}")
        return []


def generate_from_video(
    video_path_or_url: str,
    channel_handle: str,
    output_dir: str = OUTPUT_DIR,
    base_fps: float = 2.0,
    min_moves_per_segment: int = 5,
) -> list[dict]:
    """Generate training clips from a video path or URL.

    Checks the ``video_clips`` DB table first for per-clip calibrations.
    Falls back to channel-level YAML calibration if no DB entries exist.

    Args:
        video_path_or_url: Local file path or YouTube URL.
        channel_handle: Channel handle for calibration lookup.
        output_dir: Output directory for .pt files.
        base_fps: Frame sampling rate.
        min_moves_per_segment: Minimum detected moves required to save a clip.

    Returns:
        List of clip metadata dicts.
    """
    # Download if URL
    video_path = video_path_or_url
    is_url = video_path_or_url.startswith(("http://", "https://"))

    if is_url:
        logger.info(f"Downloading video: {video_path_or_url}")
        video_path = download_video(video_path_or_url)
        if video_path is None:
            logger.error("Failed to download video")
            return []

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # 1. Try per-clip DB calibrations
    db_clips = _get_db_clips(video_id)
    if db_clips:
        logger.info(f"Using {len(db_clips)} clip(s) from DB for {video_id}")
        generator = OverlayClipGenerator(
            output_dir=output_dir,
            base_fps=base_fps,
            min_moves_per_segment=min_moves_per_segment,
        )
        results: list[dict] = []
        for clip_row in db_clips:
            cal = LayoutCalibration(
                overlay=tuple(clip_row["overlay_bbox"]),
                camera=tuple(clip_row["camera_bbox"]),
                ref_resolution=tuple(clip_row["ref_resolution"]),
                board_flipped=clip_row["board_flipped"],
                board_theme=clip_row["board_theme"],
            )
            clip_results = generator.generate_clips(
                video_path,
                cal,
                video_id=f"{video_id}_clip{clip_row['id']}",
                start_time=clip_row["start_time"],
                end_time=clip_row["end_time"],
            )
            results.extend(clip_results)
        return results

    # 2. Fall back to channel-level YAML calibration
    calibration = get_calibration(channel_handle)
    if calibration is None:
        logger.error(
            f"No calibration found for {channel_handle}. "
            f"Run auto-segment + auto-calibrate, or 'pipeline calibrate' first."
        )
        return []

    generator = OverlayClipGenerator(
        output_dir=output_dir,
        base_fps=base_fps,
        min_moves_per_segment=min_moves_per_segment,
    )
    return generator.generate_clips(video_path, calibration, video_id=video_id)
