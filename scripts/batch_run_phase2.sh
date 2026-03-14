#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_OOM_RUNS=3
NUM_SLOW_RUNS=3

OOM_RANKS=(0 1 0)
SLOW_RANKS=(0 1 0)
DELAYS=(10 15 20)

OOM_ITERATION=(2 3 4)
NUM_ITERATION=5
WORKLOAD_TIMEOUT=20 # in seconds

echo "=== Phase 2 batch start ==="

for i in $(seq 1 $NUM_OOM_RUNS); do
  RUN_ID="oom_trial_${i}_$(date +%Y%m%d_%H%M%S)"
  FAULT_RANK="${OOM_RANKS[$((i-1))]}"
  PORT=$((29610 + i))
  OOM_ITER="${OOM_ITERATION[$((i-1))]}"
  echo "Running OOM trial $i, rank=$FAULT_RANK, port=$PORT, injected at iter=$OOM_ITER, iteration=$NUM_ITERATION, timeout=$WORKLOAD_TIMEOUT"
  "$PROJECT_ROOT/scripts/run_fault_oom.sh" "$RUN_ID" "$FAULT_RANK" "$PORT" "$OOM_ITER" "$NUM_ITERATION" "$WORKLOAD_TIMEOUT" || true
done

for i in $(seq 1 $NUM_SLOW_RUNS); do
  RUN_ID="slow_trial_${i}_$(date +%Y%m%d_%H%M%S)"
  FAULT_RANK="${SLOW_RANKS[$((i-1))]}"
  DELAY="${DELAYS[$((i-1))]}"
  PORT=$((29630 + i))
  echo "Running slow trial $i, rank=$FAULT_RANK, port=$PORT, delay=$DELAY, iteration=$NUM_ITERATION, timeout=$WORKLOAD_TIMEOUT"
  "$PROJECT_ROOT/scripts/run_fault_slow.sh" "$RUN_ID" "$FAULT_RANK" "$PORT" "$DELAY" "$NUM_ITERATION" "$WORKLOAD_TIMEOUT" || true
done

echo "=== Phase 2 batch complete ==="
