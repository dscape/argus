"""Generate training clips from videos with 2D board overlays.

Combines OTB camera crops (training frames) with overlay-extracted ground truth
(moves) to produce .pt training clips compatible with ArgusDataset.
"""

import io
import logging
import os
import subprocess

import chess
import chess.pgn
import cv2
import numpy as np
import torch

from pipeline.overlay.calibration import LayoutCalibration, get_calibration
from pipeline.overlay.overlay_move_detector import GameSegment, detect_moves
from pipeline.overlay.overlay_reader import OverlayReader

logger = logging.getLogger(__name__)

# Import from argus for training format compatibility
try:
    from argus.chess.constraint_mask import get_legal_mask
    from argus.chess.move_vocabulary import get_vocabulary

    VOCAB = get_vocabulary()
except ImportError:
    logger.warning("argus package not installed. Clip generation will use basic format.")
    VOCAB = None

OUTPUT_DIR = os.path.join("data", "training_clips")
FRAME_SIZE = 224


class OverlayClipGenerator:
    """Generate training clips from overlay videos."""

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        base_fps: float = 2.0,
    ):
        self.output_dir = output_dir
        self.base_fps = base_fps

    def generate_clips(
        self,
        video_path: str,
        calibration: LayoutCalibration,
        video_id: str = "",
    ) -> list[dict]:
        """Generate training clips from a video with 2D overlay.

        Args:
            video_path: Path to the video file.
            calibration: Layout calibration with overlay/camera crop coordinates.
            video_id: Identifier for naming output files.

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

        # Initialize overlay reader
        reader = OverlayReader(board_theme=cal.board_theme)

        # Extract frames
        frame_skip = max(1, int(fps / self.base_fps))
        overlay_crops = []
        camera_crops = []
        frame_indices = []
        fens = []

        logger.info(
            f"Processing {video_path}: {total_frames} frames at {fps:.1f} FPS, "
            f"sampling every {frame_skip} frames"
        )

        current_frame = 0
        while current_frame < total_frames:
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

            # Read board from overlay
            fen = reader.read_fen(overlay_crop, flipped=cal.board_flipped)

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

        # Detect moves from FEN sequence
        segments = detect_moves(
            fens=fens,
            frame_indices=frame_indices,
            fps=fps,
            start_time=0.0,
        )

        if not segments:
            logger.warning("No game segments detected")
            return []

        logger.info(f"Detected {len(segments)} game segment(s)")

        # Generate a .pt clip for each game segment
        results = []
        os.makedirs(self.output_dir, exist_ok=True)

        for game_idx, segment in enumerate(segments):
            if segment.num_moves < 5:
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
            filename = f"overlay_{vid}_{game_idx}.pt"
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

        if VOCAB is None:
            return {
                "frames": frames_tensor,
                "pgn_moves": segment.pgn_moves,
                "num_moves": segment.num_moves,
            }

        from argus.chess.move_vocabulary import NO_MOVE_IDX

        no_move_idx = NO_MOVE_IDX
        vocab_size = VOCAB.size

        move_targets = torch.full((num_frames,), no_move_idx, dtype=torch.long)
        detect_targets = torch.zeros(num_frames, dtype=torch.float32)
        legal_masks = torch.zeros(num_frames, vocab_size, dtype=torch.bool)
        move_mask = torch.ones(num_frames, dtype=torch.float32)

        # Build a map from frame index to move.
        # Apply move delay: shift overlay-detected move timestamps backward
        # so they align with when the move was actually played OTB.
        delay_frames = int(move_delay_seconds * fps / frame_skip) if move_delay_seconds > 0 else 0
        move_frame_map = {}
        for m in segment.moves:
            adjusted_idx = max(m.frame_idx - delay_frames, segment.start_frame)
            move_frame_map[adjusted_idx] = m

        # Replay the game to generate legal masks
        board = chess.Board()

        for i, frame_idx in enumerate(segment_frame_indices):
            # Generate legal mask for current position
            legal_mask = get_legal_mask(board)
            legal_masks[i] = legal_mask

            # Check if this frame has a move
            if frame_idx in move_frame_map:
                m = move_frame_map[frame_idx]
                uci = m.move_uci
                idx = VOCAB.uci_to_index(uci) if VOCAB.contains(uci) else None
                if idx is not None:
                    move_targets[i] = idx
                    detect_targets[i] = 1.0

                # Push the move on the board
                try:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        logger.warning(f"Move {uci} not legal at frame {frame_idx}")
                except ValueError:
                    logger.warning(f"Invalid UCI move: {uci}")

        return {
            "frames": frames_tensor,
            "move_targets": move_targets,
            "detect_targets": detect_targets,
            "legal_masks": legal_masks,
            "move_mask": move_mask,
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


def generate_from_video(
    video_path_or_url: str,
    channel_handle: str,
    output_dir: str = OUTPUT_DIR,
    base_fps: float = 2.0,
) -> list[dict]:
    """Generate training clips from a video path or URL.

    Args:
        video_path_or_url: Local file path or YouTube URL.
        channel_handle: Channel handle for calibration lookup.
        output_dir: Output directory for .pt files.
        base_fps: Frame sampling rate.

    Returns:
        List of clip metadata dicts.
    """
    calibration = get_calibration(channel_handle)
    if calibration is None:
        logger.error(
            f"No calibration found for {channel_handle}. "
            f"Run 'pipeline overlay-calibrate' first."
        )
        return []

    # Download if URL
    video_path = video_path_or_url
    is_url = video_path_or_url.startswith(("http://", "https://"))

    if is_url:
        logger.info(f"Downloading video: {video_path_or_url}")
        video_path = download_video(video_path_or_url)
        if video_path is None:
            logger.error("Failed to download video")
            return []

    # Extract video ID for naming
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    generator = OverlayClipGenerator(output_dir=output_dir, base_fps=base_fps)
    return generator.generate_clips(video_path, calibration, video_id=video_id)
