# NCCL Log RCA Benchmark

This repository generates NCCL debug logs for distributed PyTorch workloads under:
- Normal operation (baseline)
- Fail-stop fault (CUDA OOM on one rank)
- Fail-slow fault (artificial delay on one rank)

It is designed for repeatable run collection and label generation for root-cause analysis (RCA) experiments.
## Repository Layout

- `workload/`: Python workload and fault injection logic
- `scripts/`: Shell entrypoints for single runs and batch runs
- `labels/`: CSV labels written during runs
- `phase1_runs/`: Normal run artifacts
- `phase2_runs/`: Faulted run artifacts (OOM and slow)
- `baseline_logs/`: Example NCCL logs
- `legacy_README.md`: Older setup notes

## Requirements

- Linux host
- NVIDIA GPUs (at least 2 visible GPUs)
- CUDA-capable PyTorch with NCCL backend
- Python 3.11+

The workload checks `torch.cuda.device_count()` and fails if fewer than 2 GPUs are available.

## What Each Scenario Does

- `normal`:
  - Runs collective communication with no injected fault
  - Output status is usually `completed`

- `fail_stop` (OOM):
  - Selected rank allocates GPU memory until CUDA OOM occurs
  - Intended to trigger collective failure behavior
  - Output status is often `fault_triggered_or_failed`

- `fail_slow` (straggler):
  - Selected rank sleeps before collectives each iteration
  - Simulates a slow worker
  - Output status is usually `completed`

## Quick Start

Run from repository root:

```bash
cd /workspace/nccl_log_rca_bench
```

### 1) Phase 1 (normal baseline) batch

```bash
bash scripts/batch_run_phase1.sh
```

Defaults inside script:
- 3 runs
- 2 iterations per run
- 60 second process-group timeout

### 2) Phase 2 (faulted) batch

```bash
bash scripts/batch_run_phase2.sh
```

Defaults inside script:
- OOM: 3 runs with predefined fault ranks
- Slow: 3 runs with predefined fault ranks and delays
- 2 iterations per run
- 60 second process-group timeout

## Single Run Commands

### Normal run

```bash
bash scripts/run_normal.sh [RUN_ID] [MASTER_PORT] [NUM_ITERATION] [WORKLOAD_TIMEOUT]
```

Example:

```bash
bash scripts/run_normal.sh normal_test_001 29611 2 60
```

### OOM run (fail-stop)

```bash
bash scripts/run_fault_oom.sh [RUN_ID] [FAULT_RANK] [MASTER_PORT] [INJECTED_ITERATION] [NUM_ITERATION] [WORKLOAD_TIMEOUT]
```

Example:

```bash
bash scripts/run_fault_oom.sh oom_test_001 0 29621 0 2 60
```

### Slow run (fail-slow)

```bash
bash scripts/run_fault_slow.sh [RUN_ID] [FAULT_RANK] [MASTER_PORT] [DELAY_SECONDS] [NUM_ITERATION] [WORKLOAD_TIMEOUT]
```

Example:

```bash
bash scripts/run_fault_slow.sh slow_test_001 1 29631 15 2 60
```

## Runtime Behavior and Logging

During each run:
- `workload/run_experiment.py` writes `metadata.json` (start and final status)
- NCCL debug env vars are configured in `workload/phase2_common.py`
- NCCL logs are written as `nccl_logs_<hostname>_<pid>.txt` in each run directory
- Python stdout/stderr are captured to `stdout.log` and `stderr.log`
- A row is appended to the scenario label CSV in `labels/`

NCCL-related environment variables set by workload code:
- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=<provided port>`
- `NCCL_P2P_DISABLE=1`
- `NCCL_DEBUG=TRACE`
- `NCCL_DEBUG_SUBSYS=ALL`
- `NCCL_DEBUG_FILE=<run_dir>/nccl_logs_%h_%p.txt`

## Artifacts Produced Per Run

Each run directory under `phase1_runs/` or `phase2_runs/` contains:
- `metadata.json`
- `stdout.log`
- `stderr.log`
- one or more `nccl_logs_*.txt`

Example:

```text
phase2_runs/oom_trial_1_YYYYMMDD_HHMMSS/
  metadata.json
  stdout.log
  stderr.log
  nccl_logs_<hostname>_<pid>.txt
  nccl_logs_<hostname>_<pid>.txt
```

## Labels

Label files are:
- `labels/phase1_labels.csv`
- `labels/phase2_labels.csv`

Columns written by current code:
- `run_id`
- `scenario`
- `fault_type`
- `affected_rank`
- `delay_seconds`
- `oom_injection`
- `world_size`
- `master_port`
- `fault_iteration`
- `total_iteration`
- `workload_timeout`
- `status`

Note: older CSV rows may have fewer columns from earlier code versions.

## Troubleshooting

- Port conflicts:
  - Use a different `MASTER_PORT` for concurrent or back-to-back runs.

- Not enough GPUs:
  - Ensure at least 2 GPUs are visible to PyTorch.

- NCCL hangs/timeouts:
  - Increase `WORKLOAD_TIMEOUT` and check `stderr.log` and `nccl_logs_*.txt`.

- OOM behavior is expected in fail-stop runs:
  - Those runs may terminate with non-completed status by design.
