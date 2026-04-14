#!/usr/bin/env python3
"""Evaluate a direct physical move model on held-out annotated sequences."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.replay import build_replay_board
from pipeline.physical.move_data import (
    build_board_hypotheses_from_piece_fen,
    load_eval_move_sequences,
)
from pipeline.physical.square_probe import evaluate_probe
from pipeline.shared import SQUARE_CLASS_NAMES, board_to_class_ids
from scripts.eval_physical_board_tracker import (
    FramePrediction,
    IdentityProbe,
    compute_tracker_sequence_metrics,
)

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.data.transforms import ValidationTransform
from argus.device import resolve_device
from argus.model.argus_model import ArgusModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "physical_move_model_eval.json"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ArgusModel.from_config(checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = ValidationTransform()
    vocab = get_vocabulary()
    sequences = load_eval_move_sequences(image_size=args.image_size)

    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    square_predictions: list[int] = []
    square_targets: list[int] = []
    board_annotation_ids: list[str] = []

    with torch.no_grad():
        for sequence in sequences:
            candidate_boards = build_board_hypotheses_from_piece_fen(sequence.initial_board_fen)

            for frame_offset in range(sequence.frames.shape[0]):
                board, candidate_boards, accepted_move_uci = decode_frame(
                    model=model,
                    sequence=sequence,
                    candidate_boards=candidate_boards,
                    frame_offset=frame_offset,
                    clip_length=args.clip_length,
                    decoder_mode=args.decoder_mode,
                    detect_threshold=args.detect_threshold,
                    move_confidence_threshold=args.move_confidence_threshold,
                    move_score_margin=args.move_score_margin,
                    transform=transform,
                    device=device,
                    vocab=vocab,
                )

                predicted_labels = tuple(board_to_class_ids(build_replay_board(board.board_fen())))
                target_labels = sequence.labels[frame_offset]
                annotation_id = (
                    f"{Path(sequence.clip_path).stem}_frame"
                    f"{sequence.frame_indices[frame_offset]:04d}"
                )
                predictions_by_clip[sequence.clip_path].append(
                    FramePrediction(
                        annotation_id=annotation_id,
                        frame_index=int(sequence.frame_indices[frame_offset]),
                        target_labels=target_labels,
                        predicted_labels=predicted_labels,
                        move_uci=accepted_move_uci,
                    )
                )
                square_predictions.extend(predicted_labels)
                square_targets.extend(target_labels)
                board_annotation_ids.extend([annotation_id] * 64)

    logits = torch.zeros((len(square_predictions), len(SQUARE_CLASS_NAMES)), dtype=torch.float32)
    for index, class_id in enumerate(square_predictions):
        logits[index, class_id] = 1.0
    metrics = evaluate_probe(
        IdentityProbe(),
        logits,
        torch.tensor(square_targets, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )
    move_recall, static_false_change_rate, diagnostics = compute_tracker_sequence_metrics(
        predictions_by_clip,
        tolerance=args.move_match_tolerance,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "clip_length": args.clip_length,
        "decoder_mode": args.decoder_mode,
        "detect_threshold": args.detect_threshold,
        "move_confidence_threshold": args.move_confidence_threshold,
        "move_score_margin": args.move_score_margin,
        "move_match_tolerance": args.move_match_tolerance,
        "metrics": metrics.to_dict(),
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": static_false_change_rate,
        "sequence_diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))


def decode_frame(
    *,
    model: ArgusModel,
    sequence,
    candidate_boards,
    frame_offset: int,
    clip_length: int,
    decoder_mode: str,
    detect_threshold: float,
    move_confidence_threshold: float,
    move_score_margin: float,
    transform: ValidationTransform,
    device: torch.device,
    vocab,
):
    candidate_predictions = [
        _predict_frame(model, sequence, board, frame_offset, clip_length, transform, device, vocab)
        for board in candidate_boards
    ]

    if decoder_mode == "detect_threshold":
        best_prediction = max(
            candidate_predictions,
            key=lambda prediction: prediction[1] * prediction[3],
        )
        board, detect_prob, predicted_index, predicted_prob, _, _ = best_prediction
        if (
            detect_prob >= detect_threshold
            and predicted_prob >= move_confidence_threshold
            and predicted_index not in {NO_MOVE_IDX, vocab.size - 1}
        ):
            move_uci = vocab.index_to_uci(predicted_index)
            move = chess_move_from_uci(move_uci)
            if move is not None and move in board.legal_moves:
                board.push(move)
                return board, [board], move_uci
        if len(candidate_boards) == 1:
            return board, [board], None
        return candidate_boards[0], candidate_boards, None

    best_prediction = max(
        candidate_predictions,
        key=lambda prediction: prediction[5] - prediction[4],
    )
    board, _, predicted_index, predicted_prob, stay_score, move_score = best_prediction
    if (
        predicted_index not in {NO_MOVE_IDX, vocab.size - 1}
        and predicted_prob >= move_confidence_threshold
        and move_score - stay_score >= move_score_margin
    ):
        move_uci = vocab.index_to_uci(predicted_index)
        move = chess_move_from_uci(move_uci)
        if move is not None and move in board.legal_moves:
            board.push(move)
            return board, [board], move_uci
    if len(candidate_boards) == 1:
        return board, [board], None
    return candidate_boards[0], candidate_boards, None


def _predict_frame(
    model: ArgusModel,
    sequence,
    board,
    frame_offset: int,
    clip_length: int,
    transform: ValidationTransform,
    device: torch.device,
    vocab,
):
    window_start = max(0, frame_offset - clip_length + 1)
    frames_window = sequence.frames[window_start : frame_offset + 1].clone()
    frames_window = transform(frames_window).unsqueeze(0).to(device)
    legal_mask = get_legal_mask(board).unsqueeze(0).repeat(frames_window.shape[1], 1)
    output = model(
        crops=frames_window,
        legal_masks=legal_mask.unsqueeze(0).to(device),
    )
    move_probs = output.move_probs.squeeze(0).squeeze(1)[-1]
    detect_logit = output.detect_logits.squeeze(0).squeeze(1)[-1]
    detect_prob = float(torch.sigmoid(detect_logit).item())
    predicted_index = int(move_probs.argmax().item())
    predicted_prob = float(move_probs[predicted_index].item())
    stay_score = float(torch.log(move_probs[NO_MOVE_IDX].clamp_min(1e-8)).item())
    legal_move_probs = move_probs[: vocab.num_moves]
    if int(torch.isfinite(legal_move_probs).sum().item()) == 0:
        move_score = stay_score
    else:
        move_score = float(torch.log(legal_move_probs.max().clamp_min(1e-8)).item())
    return (
        board.copy(stack=False),
        detect_prob,
        predicted_index,
        predicted_prob,
        stay_score,
        move_score,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate direct physical move model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument(
        "--decoder-mode",
        choices=("detect_threshold", "move_prob_margin"),
        default="detect_threshold",
    )
    parser.add_argument("--detect-threshold", type=float, default=0.5)
    parser.add_argument("--move-confidence-threshold", type=float, default=0.3)
    parser.add_argument("--move-score-margin", type=float, default=1.0)
    parser.add_argument("--move-match-tolerance", type=int, default=1)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_PATH)
    return parser


def chess_move_from_uci(uci: str):
    import chess

    try:
        return chess.Move.from_uci(uci)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
