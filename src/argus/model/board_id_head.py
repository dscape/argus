"""Board identity tracking via embedding similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class BoardIdentityTracker:
    """Tracks board identities across frames using embedding matching."""

    def __init__(self, similarity_threshold: float = 0.5, ema_momentum: float = 0.9) -> None:
        self.similarity_threshold = similarity_threshold
        self.ema_momentum = ema_momentum
        self._known_embeddings: dict[int, torch.Tensor] = {}
        self._next_id = 0
        self._active_ids: set[int] = set()
        self._frames_since_seen: dict[int, int] = {}

    def update(
        self, embeddings: torch.Tensor, confidences: torch.Tensor,
        confidence_threshold: float = 0.5,
    ) -> list[int]:
        valid_mask = confidences > confidence_threshold
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            for bid in list(self._active_ids):
                self._frames_since_seen[bid] = self._frames_since_seen.get(bid, 0) + 1
            return []

        valid_embs = F.normalize(embeddings[valid_indices], dim=-1)
        assigned_ids: list[int] = [-1] * len(valid_indices)

        if not self._known_embeddings:
            for i in range(len(valid_indices)):
                bid = self._next_id
                self._next_id += 1
                assigned_ids[i] = bid
                self._known_embeddings[bid] = valid_embs[i].detach().clone()
                self._active_ids.add(bid)
                self._frames_since_seen[bid] = 0
            return assigned_ids

        known_ids = sorted(self._active_ids)
        known_embs = torch.stack(
            [F.normalize(self._known_embeddings[bid].unsqueeze(0), dim=-1).squeeze(0) for bid in known_ids]
        )
        sim_matrix = valid_embs @ known_embs.T
        cost_matrix = -sim_matrix.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_detections: set[int] = set()
        matched_boards: set[int] = set()
        for row, col in zip(row_indices, col_indices):
            similarity = sim_matrix[row, col].item()
            if similarity >= self.similarity_threshold:
                bid = known_ids[col]
                assigned_ids[row] = bid
                matched_detections.add(row)
                matched_boards.add(bid)
                self._known_embeddings[bid] = (
                    self.ema_momentum * self._known_embeddings[bid]
                    + (1 - self.ema_momentum) * valid_embs[row].detach()
                )
                self._frames_since_seen[bid] = 0

        for i in range(len(valid_indices)):
            if i not in matched_detections:
                bid = self._next_id
                self._next_id += 1
                assigned_ids[i] = bid
                self._known_embeddings[bid] = valid_embs[i].detach().clone()
                self._active_ids.add(bid)
                self._frames_since_seen[bid] = 0

        for bid in self._active_ids:
            if bid not in matched_boards:
                self._frames_since_seen[bid] = self._frames_since_seen.get(bid, 0) + 1

        return assigned_ids

    def deactivate_stale(self, max_frames: int = 60) -> list[int]:
        stale = [bid for bid, frames in self._frames_since_seen.items() if frames > max_frames and bid in self._active_ids]
        for bid in stale:
            self._active_ids.discard(bid)
        return stale

    def reset(self) -> None:
        self._known_embeddings.clear()
        self._next_id = 0
        self._active_ids.clear()
        self._frames_since_seen.clear()

    @property
    def active_board_ids(self) -> list[int]:
        return sorted(self._active_ids)

    @property
    def num_active(self) -> int:
        return len(self._active_ids)
