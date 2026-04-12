from __future__ import annotations

from pathlib import Path


def test_export_physical_real_board_dataset_script_exists() -> None:
    assert Path("scripts/export_physical_real_board_dataset.py").exists()
