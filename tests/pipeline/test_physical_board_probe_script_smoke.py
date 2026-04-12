from __future__ import annotations

from pathlib import Path


def test_train_physical_board_probe_script_exists() -> None:
    assert Path("scripts/train_physical_board_probe.py").exists()
