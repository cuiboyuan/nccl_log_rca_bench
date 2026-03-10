#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_RUNS=3

echo "=== Phase 1 batch start ==="

for i in $(seq 1 $NUM_RUNS); do
  RUN_ID="normal_trial_${i}_$(date +%Y%m%d_%H%M%S)"
  PORT=$((29610 + i))
  echo "Running Normal trial $i, port=$PORT"
  "$PROJECT_ROOT/scripts/run_normal.sh" "$RUN_ID" "$PORT" || true
done

echo "=== Phase 1 batch complete ==="
