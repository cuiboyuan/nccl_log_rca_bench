#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_ID="${1:-slow_$(date +%Y%m%d_%H%M%S)}"
FAULT_RANK="${2:-0}"
DELAY_SECONDS="${3:-15}"
MASTER_PORT="${4:-29602}"

RUN_DIR="$PROJECT_ROOT/phase2_runs/$RUN_ID"
LABEL_CSV="$PROJECT_ROOT/labels/phase2_labels.csv"

mkdir -p "$RUN_DIR"

export PYTHONPATH="$PROJECT_ROOT/workload:${PYTHONPATH:-}"

python3 "$PROJECT_ROOT/workload/fault_slow.py" \
  "$RUN_ID" \
  "$RUN_DIR" \
  "$FAULT_RANK" \
  "$DELAY_SECONDS" \
  "$MASTER_PORT" \
  "$LABEL_CSV" \
  > "$RUN_DIR/stdout.log" \
  2> "$RUN_DIR/stderr.log"

echo "Slow run finished: $RUN_ID"
echo "Artifacts: $RUN_DIR"
