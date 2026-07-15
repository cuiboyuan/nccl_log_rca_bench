#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_RUNS=5

NUM_ITERATION=(10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
WORKLOAD_TIMEOUT=(60 90 120 150 180 210 240 270 300 330 360 390 420 450 480 510 540 570 600) # in seconds

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
