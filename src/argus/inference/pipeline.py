"""End-to-end inference pipeline: video file → per-game PGN."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.chess.pgn_writer import PGNWriter
from argus.inference.tracker import MultiGameTracker
from argus.model.argus_model import ArgusModel
from argus.types import GameTrack

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Processes video and outputs per-game PGN files.

    Supports two modes:
    1. Single-board: process a sequence of board crop images.
    2. Multi-board: process full frames with board detection.
    """

    def __init__(
        self,
        model: ArgusModel,
        device: str | torch.device = "cuda",
        detect_threshold: float = 0.5,
        move_confidence_threshold: float = 0.3,
        fps: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.detect_threshold = detect_threshold
        self.move_confidence_threshold = move_confidence_threshold
        self.fps = fps
        self.vocab = get_vocabulary()
        self.tracker = MultiGameTracker()

    @torch.no_grad()
    def process_crops(
        self,
        crop_sequence: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
    ) -> GameTrack:
        """Process a single-board crop sequence.

        Args:
            crop_sequence: (T, C, H, W) board crops.
            legal_masks: (T, VOCAB_SIZE) optional legal masks.
                If None, uses the tracker's internal state machine.

        Returns:
            GameTrack with reconstructed PGN.
        """
        self.tracker.reset()

        T = crop_sequence.shape[0]
        crops = crop_sequence.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        if legal_masks is not None:
            masks = legal_masks.unsqueeze(0).to(self.device)
        else:
            masks = self.tracker.get_legal_masks(board_id=0, batch_size=1, seq_len=T)
            masks = masks.to(self.device)

        output = self.model(crops=crops, legal_masks=masks)

        move_probs = output.move_probs.squeeze(0).squeeze(1)  # (T, VOCAB_SIZE)
        detect_logits = output.detect_logits.squeeze(0).squeeze(1)  # (T,)
        detect_probs = torch.sigmoid(detect_logits)

        # Extract moves
        game = self.tracker.get_or_create_game(0)

        for t in range(T):
            if detect_probs[t] > self.detect_threshold:
                pred_idx = move_probs[t].argmax().item()
                confidence = move_probs[t, pred_idx].item()

                if (
                    pred_idx != NO_MOVE_IDX
                    and pred_idx < self.vocab.num_moves
                    and confidence > self.move_confidence_threshold
                ):
                    uci = self.vocab.index_to_uci(pred_idx)
                    self.tracker.push_move(0, uci, confidence, t)

        return self.tracker.finalize_game(0)

    @torch.no_grad()
    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[GameTrack]:
        """Process a video file and extract PGN for all detected games.

        Args:
            video_path: Path to input video.
            output_dir: Directory to save PGN files. If None, don't save.

        Returns:
            List of GameTrack for each detected game.
        """
        try:
            import av
        except ImportError:
            raise ImportError("PyAV is required for video processing: pip install av")

        video_path = Path(video_path)
        logger.info(f"Processing video: {video_path}")

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate)
        frame_skip = max(1, int(video_fps / self.fps))

        self.tracker.reset()
        frame_idx = 0
        processed = 0

        for frame in container.decode(video=0):
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # Convert to tensor
            img = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

            # For single board mode, treat as a 1-frame crop sequence
            if not self.model.use_detector:
                crops = tensor.unsqueeze(1)  # (1, 1, C, H, W)
                masks = self.tracker.get_legal_masks(0, 1, 1).to(self.device)
                output = self.model(crops=crops, legal_masks=masks)

                move_probs = output.move_probs.squeeze()  # (VOCAB_SIZE,)
                detect_logit = output.detect_logits.squeeze()
                detect_prob = torch.sigmoid(detect_logit).item()

                if detect_prob > self.detect_threshold:
                    pred_idx = move_probs.argmax().item()
                    confidence = move_probs[pred_idx].item()
                    if (
                        pred_idx != NO_MOVE_IDX
                        and pred_idx < self.vocab.num_moves
                        and confidence > self.move_confidence_threshold
                    ):
                        uci = self.vocab.index_to_uci(pred_idx)
                        self.tracker.push_move(0, uci, confidence, processed)
            else:
                # Multi-board: run detection + tracking
                # This requires full frame processing
                frames_batch = tensor.unsqueeze(1)  # (1, 1, C, H, W)
                output = self.model(frames=frames_batch)
                # TODO: implement multi-board inference with tracker

            processed += 1
            frame_idx += 1

        container.close()

        # Finalize all games
        tracks = self.tracker.finalize_all()

        # Save PGN files
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for track in tracks:
                pgn_path = out / f"game_board{track.board_id}.pgn"
                pgn_path.write_text(track.pgn)
                logger.info(f"Saved {pgn_path} ({track.moves.__len__()} moves)")

        logger.info(f"Processed {processed} frames, found {len(tracks)} games")
        return tracks
