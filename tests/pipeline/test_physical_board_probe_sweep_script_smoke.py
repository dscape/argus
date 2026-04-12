from __future__ import annotations

from pathlib import Path


def test_sweep_physical_board_probe_script_exists() -> None:
    assert Path("scripts/sweep_physical_board_probe.py").exists()
