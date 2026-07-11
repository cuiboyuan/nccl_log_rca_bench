#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_RUNS=10

NUM_ITERATION=(10 20 30 40 50 60 70 80 90 100)
WORKLOAD_TIMEOUT=(60 120 180 240 300 300 420 480 540 600) # in seconds

echo "=== Phase 1 batch start ==="

for i in $(seq 1 $NUM_RUNS); do
  for j in $(seq 0 $((${#NUM_ITERATION[@]} - 1))); do
    RUN_ID="normal_trial_${i}_iter${NUM_ITERATION[$j]}_timeout${WORKLOAD_TIMEOUT[$j]}_$(date +%Y%m%d_%H%M%S)"
    PORT=$((29610 + i))
    echo "Running Normal trial $i, iteration=${NUM_ITERATION[$j]}, timeout=${WORKLOAD_TIMEOUT[$j]}, port=$PORT"
    "$PROJECT_ROOT/scripts/run_normal.sh" "$RUN_ID" "$PORT" "${NUM_ITERATION[$j]}" "${WORKLOAD_TIMEOUT[$j]}" || true
  done
done

echo "=== Phase 1 batch complete ==="
