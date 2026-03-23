#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_ITERATION=20
WORKLOAD_TIMEOUT=120 # in seconds

NUM_OOM_RUNS=10
OOM_RANKS=(0 1)
OOM_ITERATION=(2 10 18)

NUM_SLOW_RUNS=10
SLOW_RANKS=(0 1)
DELAYS=(6 12 30)


echo "=== Phase 2 batch start ==="

for i in $(seq 1 $NUM_OOM_RUNS); do
  for FAULT_RANK in "${OOM_RANKS[@]}"; do
    for OOM_ITER in "${OOM_ITERATION[@]}"; do
      RUN_ID="oom_trial_${i}_${FAULT_RANK}_${OOM_ITER}_$(date +%Y%m%d_%H%M%S)"
      PORT=$((29610 + i))
      echo "Running OOM trial $i, rank=$FAULT_RANK, port=$PORT, injected at iter=$OOM_ITER, iteration=$NUM_ITERATION, timeout=$WORKLOAD_TIMEOUT"
      "$PROJECT_ROOT/scripts/run_fault_oom.sh" "$RUN_ID" "$FAULT_RANK" "$PORT" "$OOM_ITER" "$NUM_ITERATION" "$WORKLOAD_TIMEOUT" || true
    done
  done
done

for i in $(seq 1 $NUM_SLOW_RUNS); do
  for FAULT_RANK in "${SLOW_RANKS[@]}"; do
    for DELAY in "${DELAYS[@]}"; do
      RUN_ID="slow_trial_${i}_${FAULT_RANK}_${DELAY}_$(date +%Y%m%d_%H%M%S)"
      PORT=$((29630 + i))
      echo "Running slow trial $i, rank=$FAULT_RANK, port=$PORT, delay=$DELAY, iteration=$NUM_ITERATION, timeout=$WORKLOAD_TIMEOUT"
      "$PROJECT_ROOT/scripts/run_fault_slow.sh" "$RUN_ID" "$FAULT_RANK" "$PORT" "$DELAY" "$NUM_ITERATION" "$WORKLOAD_TIMEOUT" || true
    done
  done
done

echo "=== Phase 2 batch complete ==="
