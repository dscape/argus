from __future__ import annotations

import torch

from argus.device import default_device, resolve_device


def test_resolve_device_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "mps"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"


def test_resolve_device_falls_back_to_best_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == "mps"
    assert resolve_device("cuda") == "mps"
    assert resolve_device(torch.device("cpu")) == "cpu"
