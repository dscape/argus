#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTH_DIR="$ROOT_DIR/autoresearch"
SESSION_FILE="${AUTORESEARCH_SESSION_FILE:-$AUTH_DIR/session-store/overnight-edit-loop.jsonl}"
STOP_FILE="${AUTORESEARCH_STOP_FILE:-$AUTH_DIR/STOP}"
ITERATION_FILE="${AUTORESEARCH_ITERATION_FILE:-$AUTH_DIR/controller.iteration}"
RUN_LOG_FILE="${AUTORESEARCH_RUN_LOG_FILE:-$AUTH_DIR/run.log}"
TIMEOUT_SECONDS="${AUTORESEARCH_TIMEOUT_SECONDS:-120}"
LANE_NAME="${AUTORESEARCH_LANE_NAME:-main}"
LANE_FOCUS="${AUTORESEARCH_LANE_FOCUS:-stay focused on decoder-only temporal ambiguity work}"
LANE_SCOPE="${AUTORESEARCH_LANE_SCOPE:-use only small, reviewable changes; one or two knob changes per experiment}"
LANE_BUDGET_GUIDANCE="${AUTORESEARCH_LANE_BUDGET_GUIDANCE:-cached experiments must stay cheap: target under ~30s, hard budget 120s}"
RUN_LOG_RELATIVE="autoresearch/$(basename "$RUN_LOG_FILE")"

mkdir -p "$AUTH_DIR/session-store"
cd "$ROOT_DIR"

edit_prompt() {
  cat <<EOF
Continue the autoresearch loop.

This invocation should ONLY prepare the next experiment:
1. inspect:
   - 'autoresearch/results.tsv'
   - '$RUN_LOG_RELATIVE'
   - 'autoresearch/best_train.py'
   - 'autoresearch/train.py'
2. optionally read repo files needed for context
3. edit only 'autoresearch/train.py' to define exactly one next experiment
4. stop without running 'autoresearch/train.py'

Important constraints:
- do not run 'autoresearch/train.py'; the outer controller will run it after you exit
- do not create another controller, background process, shell loop, or supervisor
- do not inspect 'autoresearch/overnight.log' or 'autoresearch/overnight.pid'
- keep changes confined to 'autoresearch/' unless you only need to read repo context
- $LANE_FOCUS
- $LANE_SCOPE
- $LANE_BUDGET_GUIDANCE
- avoid search explosion; keep within these caps unless the current file already violates them and you are shrinking it:
  - 'beam_size <= 8'
  - 'top_move_candidates <= 16'
  - 'top_board_candidates <= 4'
  - 'max_event_proposals <= 24'
  - 'event_window_radius <= 2'
  - 'state_aware_proposal_passes <= 1'
  - 'min_event_separation >= 2'
- if 'autoresearch/train.py' currently contains a risky or runaway config, first restore it toward 'autoresearch/best_train.py', then make the next small experiment
EOF
}

run_experiment() {
  RUN_LOG_FILE="$RUN_LOG_FILE" TIMEOUT_SECONDS="$TIMEOUT_SECONDS" .venv/bin/python - <<'PY'
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

log_path = Path(os.environ["RUN_LOG_FILE"])
timeout_seconds = float(os.environ["TIMEOUT_SECONDS"])
with log_path.open("w") as handle:
    try:
        completed = subprocess.run(
            [".venv/bin/python", "autoresearch/train.py"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        sys.exit(124)
    sys.exit(completed.returncode)
PY
}

iteration=0
if [[ -f "$ITERATION_FILE" ]]; then
  iteration="$(cat "$ITERATION_FILE")"
fi

while [[ ! -f "$STOP_FILE" ]]; do
  iteration=$((iteration + 1))
  printf '%s\n' "$iteration" > "$ITERATION_FILE"
  printf '\n[%s] %s controller iteration %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LANE_NAME" "$iteration"

  PI_CACHE_RETENTION=long pi -p --thinking high --session "$SESSION_FILE" "$(edit_prompt)"

  printf '[%s] running autoresearch/train.py with %ss timeout\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TIMEOUT_SECONDS"
  set +e
  run_experiment
  status=$?
  set -e

  if [[ $status -eq 124 ]]; then
    printf '[%s] experiment timed out; restoring best_train.py\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cp "$AUTH_DIR/best_train.py" "$AUTH_DIR/train.py"
  elif [[ $status -ne 0 ]]; then
    printf '[%s] experiment exited with status %s; restoring best_train.py\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"
    cp "$AUTH_DIR/best_train.py" "$AUTH_DIR/train.py"
  else
    printf '[%s] experiment finished successfully\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    grep '^experiment:\|^decoder:\|^board_exact:\|^move_detection_recall:\|^static_false_change_rate:\|^status:\|^snapshot:\|^report:' "$RUN_LOG_FILE" || true
  fi

  sleep 1
done

printf '[%s] stop file detected, controller exiting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
