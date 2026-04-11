"""Device-selection helpers."""

from __future__ import annotations

import torch


def default_device() -> str:
    """Return the preferred accelerator for this machine.

    Preference order:
    1. CUDA
    2. Apple MPS
    3. CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | torch.device | None = "auto") -> str:
    """Resolve a requested device, falling back to the best available option."""
    if isinstance(device, torch.device):
        device = str(device)

    if device is None or device == "auto":
        return default_device()
    if device.startswith("cuda"):
        return device if torch.cuda.is_available() else default_device()
    if device.startswith("mps"):
        return device if _mps_available() else default_device()
    return device


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
