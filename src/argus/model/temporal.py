"""Temporal module using Mamba-2 State Space Model."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaTemporalModule(nn.Module):
    """Mamba-2 SSM for temporal state tracking."""

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        d_state: int = 128,
        expand: int = 2,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        if input_dim is not None and input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if MAMBA_AVAILABLE:
            for _ in range(n_layers):
                self.layers.append(Mamba2(d_model=d_model, d_state=d_state, expand=expand))
                self.norms.append(nn.LayerNorm(d_model))
        else:
            for _ in range(n_layers):
                self.layers.append(nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True))
                self.norms.append(nn.LayerNorm(d_model))

        self.final_norm = nn.LayerNorm(d_model)
        self._is_mamba = MAMBA_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for norm, layer in zip(self.norms, self.layers):
            residual = x
            x = norm(x)
            if self._is_mamba:
                x = layer(x)
            else:
                x, _ = layer(x)
            x = x + residual
        return self.final_norm(x)
