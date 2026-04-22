"""Single-frame board inference: sample from top-k per square, score, take best."""

from __future__ import annotations

import numpy as np

import chess

from .constraints import satisfies_hard_class_constraints
from .state import BoardState, board_from_class_ids, log_softmax_rows, numpy_logits_to_constrained_class_ids, score_board

_DEFAULT_K = 3
_DEFAULT_NUM_SAMPLES = 200
_DEFAULT_SEED = 0


def per_square_top_k_indices(logp: np.ndarray, k: int) -> np.ndarray:
    k = min(k, logp.shape[1])
    if k < 1:
        raise ValueError("k must be at least 1")
    return np.argsort(-logp, axis=1)[:, :k]


def sample_class_ids_stochastic(
    logp: np.ndarray,
    top: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    cids: list[int] = []
    for s in range(64):
        choices = top[s]
        weights = np.exp(np.maximum(logp[s, choices] - logp[s, choices].max(), -30.0))
        weights /= float(np.sum(weights) + 1e-12)
        pick = int(rng.choice(choices, p=weights))
        cids.append(pick)
    return cids


def generate_candidates(
    logits: np.ndarray,
    *,
    k: int = _DEFAULT_K,
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    seed: int | None = _DEFAULT_SEED,
) -> list[tuple[chess.Board, list[int]]]:
    """Build a list of candidate boards (``num_samples`` stochastic samples plus a few heuristics)."""
    rng = np.random.default_rng(seed)
    logp = log_softmax_rows(np.asarray(logits, dtype=np.float64))
    top = per_square_top_k_indices(logp, k)
    cids0 = [int(np.argmax(logp[i])) for i in range(64)]
    cids1 = numpy_logits_to_constrained_class_ids(logits)
    out: list[tuple[chess.Board, list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    def _try(cids: list[int]) -> None:
        t = tuple(cids)
        if t in seen:
            return
        if satisfies_hard_class_constraints(cids):
            seen.add(t)
            out.append((board_from_class_ids(cids, turn=chess.WHITE), cids))

    for cids in (cids0, cids1):
        _try(cids)
    for _ in range(num_samples):
        _try(sample_class_ids_stochastic(logp, top, rng))
    if not out:
        out.append((board_from_class_ids(cids0, turn=chess.WHITE), cids0))
    return out


def infer_board(
    logits: np.ndarray,
    *,
    k: int = _DEFAULT_K,
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    seed: int | None = _DEFAULT_SEED,
) -> BoardState:
    """Map ``logits`` shaped ``(64, 13)`` to the highest-scoring plausible board.

    The search space is pruned with top-``k`` per square, random re-sampling, and the
    same lightweight class constraints used elsewhere in the repo (kings, pawns,
    no adjacent kings, no back-rank pawns). Turn defaults to **white** for the
    snapshot; temporal tracking (see :mod:`pipeline.board_inference.tracker`) is
    responsible for stateful legality and move continuity.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (64, 13):
        raise ValueError(f"Expected logits shaped (64, 13), got {logits.shape}")

    cands = generate_candidates(logits, k=k, num_samples=num_samples, seed=seed)
    best: chess.Board | None = None
    best_s = float("-inf")
    for b, _ in cands:
        s = score_board(b, logits)
        if s > best_s or best is None:
            best_s = s
            best = b
    assert best is not None
    return BoardState(best, best_s)
