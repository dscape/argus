#!/usr/bin/env python3
"""Evaluate a direct physical move model on held-out annotated sequences."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.overlay.replay import build_replay_board
from pipeline.physical.move_data import (
    build_board_hypotheses_from_piece_fen,
    load_eval_move_sequences,
)
from pipeline.physical.square_probe import evaluate_probe
from pipeline.shared import (
    SQUARE_CLASS_NAMES,
    LegalSequenceBeamDecoder,
    SegmentalLegalSequenceDecoder,
    board_to_class_ids,
)
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
    sequences = load_eval_move_sequences(
        image_size=args.image_size,
        observation_mode=args.observation_mode,
        oblique_crop_margin=args.oblique_crop_margin,
    )

    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    decoder_diagnostics_by_clip: dict[str, dict[str, object]] = {}
    decoder_anomalies_by_clip: dict[str, list[dict[str, object]]] = {}
    square_predictions: list[int] = []
    square_targets: list[int] = []
    board_annotation_ids: list[str] = []

    with torch.no_grad():
        for sequence in sequences:
            if args.decoder_mode == "beam":
                sequence_predictions = decode_sequence_with_beam(
                    model=model,
                    sequence=sequence,
                    transform=transform,
                    device=device,
                    beam_size=args.beam_size,
                    top_move_candidates=args.beam_top_moves,
                    top_board_candidates=args.beam_top_board_moves,
                    board_weight=args.beam_board_weight,
                    move_weight=args.beam_move_weight,
                    detect_weight=args.beam_detect_weight,
                    move_score_margin=args.beam_move_score_margin,
                    lookahead_window=args.beam_lookahead_window,
                    lookahead_weight=args.beam_lookahead_weight,
                    lookahead_decay=args.beam_lookahead_decay,
                )
            elif args.decoder_mode == "segmental":
                sequence_predictions, sequence_decoder_diagnostics = (
                    decode_sequence_with_segmental_decoder(
                        model=model,
                        sequence=sequence,
                        transform=transform,
                        device=device,
                        beam_size=args.beam_size,
                        top_move_candidates=args.beam_top_moves,
                        top_board_candidates=args.beam_top_board_moves,
                        board_weight=args.beam_board_weight,
                        move_weight=args.beam_move_weight,
                        detect_weight=args.beam_detect_weight,
                        move_score_margin=args.beam_move_score_margin,
                        detect_peak_threshold=args.segmental_detect_peak_threshold,
                        board_change_peak_threshold=args.segmental_board_change_peak_threshold,
                        min_event_separation=args.segmental_min_event_separation,
                        secondary_min_event_separation=(
                            None
                            if args.segmental_secondary_min_event_separation <= 0
                            else args.segmental_secondary_min_event_separation
                        ),
                        secondary_peak_ratio=args.segmental_secondary_peak_ratio,
                        state_aware_proposal_passes=args.segmental_state_aware_proposal_passes,
                        anomaly_change_evidence_threshold=(
                            args.segmental_anomaly_change_evidence_threshold
                        ),
                        anomaly_settled_gain_threshold=(
                            args.segmental_anomaly_settled_gain_threshold
                        ),
                        segment_board_drop_worst_frames=(args.segmental_board_drop_worst_frames),
                        event_window_radius=args.segmental_event_window_radius,
                        max_event_proposals=args.segmental_max_event_proposals,
                        diagnostic_settled_horizon=args.segmental_diagnostic_settled_horizon,
                    )
                )
                sequence_anomalies = sequence_decoder_diagnostics.get(
                    "possible_illegal_segments",
                    [],
                )
                if sequence_anomalies:
                    decoder_anomalies_by_clip[sequence.clip_path] = sequence_anomalies
                if args.include_decoder_diagnostics:
                    decoder_diagnostics_by_clip[sequence.clip_path] = sequence_decoder_diagnostics
            else:
                candidate_boards = build_board_hypotheses_from_piece_fen(
                    sequence.initial_board_fen,
                    initial_side_to_move=sequence.initial_side_to_move,
                )
                sequence_predictions = []
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
                    sequence_predictions.append((board.copy(stack=False), accepted_move_uci))

            for frame_offset, (board, accepted_move_uci) in enumerate(sequence_predictions):
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
        "observation_mode": args.observation_mode,
        "decoder_mode": args.decoder_mode,
        "detect_threshold": args.detect_threshold,
        "move_confidence_threshold": args.move_confidence_threshold,
        "move_score_margin": args.move_score_margin,
        "beam_size": args.beam_size,
        "beam_top_moves": args.beam_top_moves,
        "beam_board_weight": args.beam_board_weight,
        "beam_move_weight": args.beam_move_weight,
        "beam_detect_weight": args.beam_detect_weight,
        "beam_top_board_moves": args.beam_top_board_moves,
        "beam_move_score_margin": args.beam_move_score_margin,
        "beam_lookahead_window": args.beam_lookahead_window,
        "beam_lookahead_weight": args.beam_lookahead_weight,
        "beam_lookahead_decay": args.beam_lookahead_decay,
        "segmental_detect_peak_threshold": args.segmental_detect_peak_threshold,
        "segmental_board_change_peak_threshold": args.segmental_board_change_peak_threshold,
        "segmental_min_event_separation": args.segmental_min_event_separation,
        "segmental_secondary_min_event_separation": args.segmental_secondary_min_event_separation,
        "segmental_secondary_peak_ratio": args.segmental_secondary_peak_ratio,
        "segmental_state_aware_proposal_passes": args.segmental_state_aware_proposal_passes,
        "segmental_anomaly_change_evidence_threshold": (
            args.segmental_anomaly_change_evidence_threshold
        ),
        "segmental_anomaly_settled_gain_threshold": (args.segmental_anomaly_settled_gain_threshold),
        "segmental_board_drop_worst_frames": args.segmental_board_drop_worst_frames,
        "segmental_event_window_radius": args.segmental_event_window_radius,
        "segmental_max_event_proposals": args.segmental_max_event_proposals,
        "segmental_diagnostic_settled_horizon": args.segmental_diagnostic_settled_horizon,
        "move_match_tolerance": args.move_match_tolerance,
        "metrics": metrics.to_dict(),
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": static_false_change_rate,
        "sequence_diagnostics": diagnostics,
    }
    if args.decoder_mode == "segmental":
        report["decoder_anomaly_summary"] = summarize_decoder_anomalies(decoder_anomalies_by_clip)
    if decoder_diagnostics_by_clip:
        report["decoder_diagnostics"] = decoder_diagnostics_by_clip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))


def summarize_decoder_anomalies(
    anomalies_by_clip: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    flattened = [
        {"clip_path": clip_path, **segment}
        for clip_path, segments in anomalies_by_clip.items()
        for segment in segments
    ]
    flattened.sort(
        key=lambda segment: (
            float(segment.get("anomaly_score", 0.0)),
            float(segment.get("change_evidence", 0.0)),
        ),
        reverse=True,
    )
    return {
        "clips_with_possible_illegal_move": len(anomalies_by_clip),
        "segments_with_possible_illegal_move": len(flattened),
        "top_possible_illegal_segments": flattened[:10],
    }


def decode_frame(
    *,
    model: ArgusModel,
    sequence: Any,
    candidate_boards: list[chess.Board],
    frame_offset: int,
    clip_length: int,
    decoder_mode: str,
    detect_threshold: float,
    move_confidence_threshold: float,
    move_score_margin: float,
    transform: ValidationTransform,
    device: torch.device,
    vocab: Any,
) -> tuple[chess.Board, list[chess.Board], str | None]:
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


def decode_sequence_with_beam(
    *,
    model: ArgusModel,
    sequence: Any,
    transform: ValidationTransform,
    device: torch.device,
    beam_size: int,
    top_move_candidates: int,
    top_board_candidates: int,
    board_weight: float,
    move_weight: float,
    detect_weight: float,
    move_score_margin: float,
    lookahead_window: int,
    lookahead_weight: float,
    lookahead_decay: float,
) -> list[tuple[chess.Board, str | None]]:
    frames = transform(sequence.frames.clone()).unsqueeze(0).to(device)
    board_corners = None
    if sequence.board_corners is not None:
        board_corners = sequence.board_corners.unsqueeze(0).to(device)
    output = model(crops=frames, board_corners=board_corners)
    if output.square_logits is None:
        raise ValueError(
            "Beam decoding requires square logits; train/eval with square head enabled"
        )

    decoded = LegalSequenceBeamDecoder(
        sequence.initial_board_fen,
        initial_side_to_move=sequence.initial_side_to_move,
        beam_size=beam_size,
        top_move_candidates=top_move_candidates,
        top_board_candidates=top_board_candidates,
        board_weight=board_weight,
        move_weight=move_weight,
        detect_weight=detect_weight,
        move_score_margin=move_score_margin,
        lookahead_window=lookahead_window,
        lookahead_weight=lookahead_weight,
        lookahead_decay=lookahead_decay,
    ).decode(
        output.square_logits.squeeze(0),
        sequence_move_logits=output.move_logits.squeeze(0).squeeze(1),
        sequence_detect_logits=output.detect_logits.squeeze(0).squeeze(1),
    )
    return [(chess.Board(frame.full_fen), frame.move_uci) for frame in decoded.frames]


def decode_sequence_with_segmental_decoder(
    *,
    model: ArgusModel,
    sequence: Any,
    transform: ValidationTransform,
    device: torch.device,
    beam_size: int,
    top_move_candidates: int,
    top_board_candidates: int,
    board_weight: float,
    move_weight: float,
    detect_weight: float,
    move_score_margin: float,
    detect_peak_threshold: float,
    board_change_peak_threshold: float,
    min_event_separation: int,
    secondary_min_event_separation: int | None,
    secondary_peak_ratio: float,
    state_aware_proposal_passes: int,
    anomaly_change_evidence_threshold: float,
    anomaly_settled_gain_threshold: float,
    segment_board_drop_worst_frames: int,
    event_window_radius: int,
    max_event_proposals: int,
    diagnostic_settled_horizon: int,
) -> tuple[list[tuple[chess.Board, str | None]], dict[str, object]]:
    frames = transform(sequence.frames.clone()).unsqueeze(0).to(device)
    board_corners = None
    if sequence.board_corners is not None:
        board_corners = sequence.board_corners.unsqueeze(0).to(device)
    output = model(crops=frames, board_corners=board_corners)
    if output.square_logits is None:
        raise ValueError(
            "Segmental decoding requires square logits; train/eval with square head enabled"
        )

    decoded = SegmentalLegalSequenceDecoder(
        sequence.initial_board_fen,
        initial_side_to_move=sequence.initial_side_to_move,
        beam_size=beam_size,
        top_move_candidates=top_move_candidates,
        top_board_candidates=top_board_candidates,
        board_weight=board_weight,
        move_weight=move_weight,
        detect_weight=detect_weight,
        move_score_margin=move_score_margin,
        detect_peak_threshold=detect_peak_threshold,
        board_change_peak_threshold=board_change_peak_threshold,
        min_event_separation=min_event_separation,
        secondary_min_event_separation=secondary_min_event_separation,
        secondary_peak_ratio=secondary_peak_ratio,
        state_aware_proposal_passes=state_aware_proposal_passes,
        anomaly_change_evidence_threshold=anomaly_change_evidence_threshold,
        anomaly_settled_gain_threshold=anomaly_settled_gain_threshold,
        segment_board_drop_worst_frames=segment_board_drop_worst_frames,
        event_window_radius=event_window_radius,
        max_event_proposals=max_event_proposals,
        diagnostic_settled_horizon=diagnostic_settled_horizon,
    ).decode(
        output.square_logits.squeeze(0),
        sequence_move_logits=output.move_logits.squeeze(0).squeeze(1),
        sequence_detect_logits=output.detect_logits.squeeze(0).squeeze(1),
    )
    return (
        [(chess.Board(frame.full_fen), frame.move_uci) for frame in decoded.frames],
        {} if decoded.diagnostics is None else decoded.diagnostics,
    )


def _predict_frame(
    model: ArgusModel,
    sequence: Any,
    board: chess.Board,
    frame_offset: int,
    clip_length: int,
    transform: ValidationTransform,
    device: torch.device,
    vocab: Any,
) -> tuple[chess.Board, float, int, float, float, float]:
    window_start = max(0, frame_offset - clip_length + 1)
    frames_window = sequence.frames[window_start : frame_offset + 1].clone()
    frames_window = transform(frames_window).unsqueeze(0).to(device)
    board_corners = None
    if sequence.board_corners is not None:
        board_corners = sequence.board_corners[window_start : frame_offset + 1].clone()
        board_corners = board_corners.unsqueeze(0).to(device)
    legal_mask = get_legal_mask(board).unsqueeze(0).repeat(frames_window.shape[1], 1)
    output = model(
        crops=frames_window,
        legal_masks=legal_mask.unsqueeze(0).to(device),
        board_corners=board_corners,
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
        "--observation-mode",
        choices=("rectified", "oblique", "native_oblique"),
        default="oblique",
    )
    parser.add_argument("--oblique-crop-margin", type=float, default=0.18)
    parser.add_argument(
        "--decoder-mode",
        choices=("beam", "segmental", "detect_threshold", "move_prob_margin"),
        default="beam",
    )
    parser.add_argument("--detect-threshold", type=float, default=0.5)
    parser.add_argument("--move-confidence-threshold", type=float, default=0.3)
    parser.add_argument("--move-score-margin", type=float, default=1.0)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--beam-top-moves", type=int, default=16)
    parser.add_argument("--beam-top-board-moves", type=int, default=0)
    parser.add_argument("--beam-board-weight", type=float, default=1.0)
    parser.add_argument("--beam-move-weight", type=float, default=1.0)
    parser.add_argument("--beam-detect-weight", type=float, default=1.0)
    parser.add_argument("--beam-move-score-margin", type=float, default=0.0)
    parser.add_argument("--beam-lookahead-window", type=int, default=1)
    parser.add_argument("--beam-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--beam-lookahead-decay", type=float, default=1.0)
    parser.add_argument("--segmental-detect-peak-threshold", type=float, default=0.1)
    parser.add_argument("--segmental-board-change-peak-threshold", type=float, default=0.03125)
    parser.add_argument("--segmental-min-event-separation", type=int, default=4)
    parser.add_argument("--segmental-secondary-min-event-separation", type=int, default=0)
    parser.add_argument("--segmental-secondary-peak-ratio", type=float, default=0.8)
    parser.add_argument("--segmental-state-aware-proposal-passes", type=int, default=0)
    parser.add_argument("--segmental-anomaly-change-evidence-threshold", type=float, default=0.25)
    parser.add_argument("--segmental-anomaly-settled-gain-threshold", type=float, default=0.0)
    parser.add_argument("--segmental-board-drop-worst-frames", type=int, default=0)
    parser.add_argument("--segmental-event-window-radius", type=int, default=1)
    parser.add_argument("--segmental-max-event-proposals", type=int, default=32)
    parser.add_argument("--segmental-diagnostic-settled-horizon", type=int, default=8)
    parser.add_argument("--include-decoder-diagnostics", action="store_true")
    parser.add_argument("--move-match-tolerance", type=int, default=1)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_PATH)
    return parser


def chess_move_from_uci(uci: str) -> chess.Move | None:
    try:
        return chess.Move.from_uci(uci)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
