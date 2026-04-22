#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTH_DIR="$ROOT_DIR/autoresearch"

export AUTORESEARCH_SESSION_FILE="$AUTH_DIR/session-store/overnight-slow-edit-loop.jsonl"
export AUTORESEARCH_STOP_FILE="$AUTH_DIR/STOP_SLOW"
export AUTORESEARCH_ITERATION_FILE="$AUTH_DIR/controller_slow.iteration"
export AUTORESEARCH_RUN_LOG_FILE="$AUTH_DIR/run_slow.log"
export AUTORESEARCH_TIMEOUT_SECONDS=360
export AUTORESEARCH_LANE_NAME="slow"
export AUTORESEARCH_LANE_FOCUS="stay focused on slow-lane state-aware temporal ambiguity work"
export AUTORESEARCH_LANE_SCOPE="restrict this lane to state-aware proposal/refinement experiments or similarly slow state-aware fallback work; do not spend it on cheap experiments already suited for autoresearch/controller.sh"
export AUTORESEARCH_LANE_BUDGET_GUIDANCE="slow-lane experiments may run off-budget for the main loop: target under ~300s, hard budget 360s"

exec "$AUTH_DIR/controller.sh"
