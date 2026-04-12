from __future__ import annotations

from pathlib import Path


def test_eval_physical_board_runtime_script_exists() -> None:
    assert Path("scripts/eval_physical_board_runtime.py").exists()
