"""Trainer receipts: which clips / annotations a run actually consumed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

RECEIPT_FILENAME = "used_manifest.jsonl"


def write_training_receipt(
    output_dir: str | Path,
    *,
    kind: str,
    entries: Iterable[dict[str, str | int | None]],
) -> Path:
    """Write `used_manifest.jsonl` next to a checkpoint.

    `kind` labels the trainer (e.g. "board_probe", "square_classifier",
    "physical_move_model"). `entries` each carry at least one identifying
    field — `clip_path`, `annotation_id`, or `source_video_id` — so the
    audit script can reconcile them against what sits on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RECEIPT_FILENAME
    written_at = datetime.now(timezone.utc).isoformat()
    lines = [
        json.dumps(
            {"kind": kind, "written_at": written_at, **entry},
            sort_keys=True,
        )
        for entry in entries
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def read_training_receipt(path: str | Path) -> list[dict[str, str | int | None]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, str | int | None]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
