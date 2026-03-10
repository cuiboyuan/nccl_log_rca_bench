#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_ID="${1:-slow_$(date +%Y%m%d_%H%M%S)}"
FAULT_RANK="${2:-0}"
MASTER_PORT="${3:-29602}"
DELAY_SECONDS="${4:-15}"
NUM_ITERATION="${5:-5}"
WORKLOAD_TIMEOUT="${6:-120}" # in seconds

RUN_DIR="$PROJECT_ROOT/phase2_runs/$RUN_ID"
LABEL_CSV="$PROJECT_ROOT/labels/phase2_labels.csv"

mkdir -p "$RUN_DIR"

export PYTHONPATH="$PROJECT_ROOT/workload:${PYTHONPATH:-}"

python3 "$PROJECT_ROOT/workload/run_experiment.py" \
  "$RUN_ID" \
  "$RUN_DIR" \
  "$FAULT_RANK" \
  "$MASTER_PORT" \
  "$LABEL_CSV" \
  "fail_slow" \
  "$NUM_ITERATION" \
  "$WORKLOAD_TIMEOUT" \
  "$DELAY_SECONDS" \
  > "$RUN_DIR/stdout.log" \
  2> >(tee "$RUN_DIR/stderr.log" >&2) # print to both terminal and file

echo "Slow run finished: $RUN_ID"
echo "Artifacts: $RUN_DIR"
