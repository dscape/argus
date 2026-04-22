# autoresearch

This is an Argus-local, autoresearch-style workspace focused on the **highest-ceiling current direction**:

- baseline: `outputs/2026-04-16/tracker_sweep_rectified_realplusmanual/eval_w2_m10.json`
- current `board_exact`: `0.216568`
- dominant failure mode from the latest tagged study:
  - `outputs/2026-04-16/physical_board_failure_study_rectified_realplusmanual_w2_m10/`
  - `12 / 13` sampled failures were `temporal in-between / move execution ambiguity`
  - `1 / 13` was `piece classifier / square evidence`
- conclusion: do **decoder / temporal-tolerance research**, not new classifier / encoder / data sweeps here

## Files

- `prepare.py` — fixed prep + evaluation harness. Do not modify.
- `train.py` — the only file you modify.
- `program.md` — these instructions.
- `pyproject.toml` — shape-only, present because autoresearch expects it.
- `results.tsv` — local experiment log.

## Setup

1. Read:
   - `autoresearch/program.md`
   - `autoresearch/prepare.py`
   - `autoresearch/train.py`
   - `outputs/plan.md`
   - `outputs/2026-04-16/tracker_sweep_rectified_realplusmanual/eval_w2_m10.json`
2. Verify the cache exists. If not, run:
   - `.venv/bin/python autoresearch/prepare.py`
3. Confirm that `autoresearch/results.tsv` exists and has a header.
4. Confirm `autoresearch/best_train.py` exists.

## Fixed contract

- Optimize **`board_exact` first** on the fixed cached held-out rectified-board logits.
- Tie-break in this order:
  1. lower `static_false_change_rate`
  2. higher `move_detection_recall`
  3. higher `macro_f1`
  4. higher `non_empty_accuracy`
- Do not modify `prepare.py`.
- Do not add dependencies.
- Do not switch this workspace to probe training / data generation / classifier architecture work.
- Keep experiments focused on the temporal ambiguity bucket: transition-frame misses, delayed commits, long desyncs, dwell-aware decoding, settle windows, hysteresis, proposal quality, event clustering, state-aware timing.

## Run command

Redirect all output to a log:

```bash
.venv/bin/python autoresearch/train.py > autoresearch/run.log 2>&1
```

Then inspect:

```bash
grep "^board_exact:\|^status:\|^snapshot:\|^report:" autoresearch/run.log
```

## Logging / keep-discard behavior

- `train.py` automatically:
  - snapshots itself into `autoresearch/snapshots/`
  - writes a JSON report into `autoresearch/runs/`
  - appends a row to `autoresearch/results.tsv`
  - updates `autoresearch/best_train.py` on `keep`
  - restores `autoresearch/train.py` from `autoresearch/best_train.py` on `discard`
- If the run crashes before logging, restore manually:

```bash
cp autoresearch/best_train.py autoresearch/train.py
```

## First run

The first run must be the current baseline in `train.py` as-is.

## Controller lanes

- `autoresearch/controller.sh` — main cheap loop, 120s hard timeout
- `autoresearch/controller_slow.sh` — off-budget slow lane for state-aware experiments, 360s hard timeout
- Run only one controller at a time; both lanes edit the same `autoresearch/train.py` workspace.

## Research loop

Loop forever:

1. Edit `autoresearch/train.py`.
2. Run `.venv/bin/python autoresearch/train.py > autoresearch/run.log 2>&1`.
3. Read the summary lines from `autoresearch/run.log`.
4. If it crashed, inspect `tail -n 80 autoresearch/run.log`, restore `train.py` from `best_train.py`, and try a different idea.
5. If it ran, trust the auto-logged `keep` / `discard` result and continue immediately.

Do not stop to ask the human whether to continue. Keep going until interrupted.
