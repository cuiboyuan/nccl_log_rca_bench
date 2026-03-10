#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_ID="${1:-oom_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${2:-29601}"

RUN_DIR="$PROJECT_ROOT/phase1_runs/$RUN_ID"
LABEL_CSV="$PROJECT_ROOT/labels/phase1_labels.csv"

mkdir -p "$RUN_DIR"

export PYTHONPATH="$PROJECT_ROOT/workload:${PYTHONPATH:-}"

python3 "$PROJECT_ROOT/workload/run_experiment.py" \
  "$RUN_ID" \
  "$RUN_DIR" \
  0 \
  "$MASTER_PORT" \
  "$LABEL_CSV" \
  "normal" \
  > "$RUN_DIR/stdout.log" \
  2> >(tee "$RUN_DIR/stderr.log" >&2) # print to both terminal and file

echo "Normal run finished: $RUN_ID"
echo "Artifacts: $RUN_DIR"
