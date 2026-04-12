from __future__ import annotations

from pathlib import Path


def test_build_physical_board_probe_ensemble_script_exists() -> None:
    assert Path("scripts/build_physical_board_probe_ensemble.py").exists()
