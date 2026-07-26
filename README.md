# GF-Bench: GPU Failure Simulation Framework and Dataset

GF-Bench is a framework for generating controlled failures in multi-GPU workloads, collecting rank-local NVIDIA Collective Communications Library (NCCL) logs, and evaluating log-based anomaly detection methods.

The current framework focuses on two failure families that are observable through NCCL traces:

- **CUDA out-of-memory (OOM)** as a fail-stop failure;
- **persistent GPU straggler** as a fail-slow failure.

The accompanying empirical study evaluates three sequence-based methods—DeepLog, LogAnomaly, and LogBERT—and the LLM-based LogPrompt. It also compares Drain-parsed logs with raw, rank-preserving logs to study the trade-off between detection effectiveness and operational cost.

## Repository Structure

```text
nccl_log_rca_bench/
├── dataset/                        # Example runs included with the repository
│   ├── labels/                     # Ground-truth label CSVs
│   ├── phase1_runs/                # Example normal runs (remove before generating a full dataset)
│   └── phase2_runs/                # Example fault-injected runs (remove before generating a full dataset)
├── workload/
│   ├── run_experiment.py       # Central execution and process-supervision entry point
│   ├── generic_workload.py     # Shared NCCL workload and fault-injection interface
│   ├── fault_oom.py            # CUDA OOM fail-stop injector
│   ├── fault_slow.py           # Persistent-straggler fail-slow injector
│   ├── no_fault.py             # Normal execution
│   └── phase2_common.py        # NCCL environment, metadata, and label utilities
├── scripts/
│   ├── run_normal.sh           # Run one normal execution
│   ├── run_fault_oom.sh        # Run one CUDA OOM execution
│   ├── run_fault_slow.sh       # Run one straggler execution
│   ├── batch_run_phase1.sh     # Generate the normal-run dataset partition
│   └── batch_run_phase2.sh     # Generate the two fault partitions
├── evaluation/
│   ├── data_process.py         # Drain parsing, sequence construction, splits, and manifest
│   ├── evaluate.py             # Primary entry point for DeepLog, LogAnomaly, and LogBERT
│   ├── evaluate_logprompt.py   # LogPrompt evaluation on parsed or raw logs
│   ├── calculate_cost.py       # LLM token-cost calculation
│   ├── deeplog_nccl.py
│   ├── loganomaly_nccl.py
│   ├── logbert_nccl.py
│   └── logbert/                # Vendored implementations used by the baselines
├── incontext_examples.csv      # Example-run specification for LogPrompt
└── requirements.txt            # Captured development environment
```

## Requirements

### Hardware

- Linux environment with Bash;
- NVIDIA driver compatible with CUDA 12.4;
- at least **two visible CUDA GPUs**;
- sufficient available GPU memory for the selected workload.

For the paper configuration, expose exactly two GPUs.

`run_experiment.py` uses all visible GPUs as the distributed world size and rejects environments with fewer than two GPUs.

### Software

The paper environment used:

- PyTorch 2.4.1;
- CUDA 12.4;
- NCCL 2.20.5.

To install all dependencies, use:

```bash
pip install -r requirements.txt
```

Some system-bound entries in the captured requirements may require an Ubuntu environment or system packages.

## Generate the Full Initial Dataset

The repository includes a small example dataset under `dataset/` that can be used directly for evaluation without running data generation. To reproduce the full dataset, first remove the example data:

```bash
rm -rf dataset/phase1_runs dataset/phase2_runs dataset/labels
```

The initial experiment grid is encoded in two batch scripts.

```bash
bash scripts/batch_run_phase1.sh
bash scripts/batch_run_phase2.sh
```

The scripts generate:

| Partition | Configuration grid | Runs |
|---|---:|---:|
| Normal | 10 workload lengths × 10 trials | 100 |
| CUDA OOM | 2 ranks × 3 injection iterations × 10 trials | 60 |
| Straggler | 2 ranks × 3 delay values × 10 trials | 60 |

Each batch wrapper continues after an expected faulted run, allowing the remaining configurations to execute.

## Prepare the Dataset for Evaluation

The generation scripts write `phase1_runs/`, `phase2_runs/`, and `labels/` at the repository root. The logical layout is:
```text
phase1_runs/<run_id>/                 # normal runs
phase2_runs/<run_id>/                 # fault-injected runs
├── metadata.json                     # scenario and execution metadata
├── stdout.log                        # workload output
├── stderr.log                        # process errors
└── nccl_logs_<host>_<pid>.txt        # per-rank NCCL TRACE logs

labels/
├── phase1_labels.csv
└── phase2_labels.csv
```

Evaluation code reads them from `dataset/`, so create the expected evaluation layout by:

```bash
mkdir -p dataset
mv phase1_runs dataset/phase1_runs
mv phase2_runs dataset/phase2_runs
mv labels dataset/labels
```

The resulting logical layout is:

```text
dataset/
├── phase1_runs
├── phase2_runs
└── labels
```

## Reproduce the Non-LLM Evaluation

First preprocess the data to obtain an even train/test partition and Drain-parsed input logs:

```bash
python evaluation/data_process.py  --normal-train-ratio 0.5 --abnormal-train-ratio 0.5
```

`evaluation/evaluate.py` is the primary single-file entry point for preprocessing, training, prediction, and metric calculation for DeepLog, LogAnomaly, and LogBERT. Since data is already processed, run evaluation of all three models:

```bash
python evaluation/evaluate.py --skip-preprocess
```

The unified evaluator writes model-specific CSV and JSON files under `output/nccl/`, including:

```text
output/nccl/
├── manifest.json
├── event_map.json
├── drain_input/
├── drain_output/
├── logbert_evaluation_results.csv
├── logbert_metrics.json
├── deeplog_evaluation_results.csv
├── deeplog_metrics.json
├── loganomaly_evaluation_results.csv
├── loganomaly_metrics.json
├── all_models_summary.csv
└── all_models_summary.json
```

## Reproduce the LogPrompt Evaluation

LogPrompt requires an OpenAI-compatible chat-completions endpoint.

```bash
export OPENAI_API_KEY="<your-api-key>"
```

The log data should already be processed by `evaluation/data_process.py`. If not, run preprocess again before issuing API requests:

```bash
python evaluation/data_process.py  --normal-train-ratio 0.5 --abnormal-train-ratio 0.5
```

### CoT with parsed logs

```bash
python evaluation/evaluate_logprompt.py \
  --model gpt-5.4 \
  --strategy CoT \
  --log-type parsed \
  --skip-preprocess
```

### CoT with raw logs

```bash
python evaluation/evaluate_logprompt.py \
  --model gpt-5.4 \
  --strategy CoT \
  --log-type raw \
  --skip-preprocess
```

### In-context learning with parsed logs

```bash
python evaluation/evaluate_logprompt.py \
  --model gpt-5.4 \
  --strategy InContext \
  --log-type parsed \
  --example-file incontext_examples.csv \
  --skip-preprocess
```

### In-context learning with raw logs

```bash
python evaluation/evaluate_logprompt.py \
  --model gpt-5.4 \
  --strategy InContext \
  --log-type raw \
  --example-file incontext_examples.csv \
  --skip-preprocess
```

`incontext_examples.csv` must contain valid run identifiers available in the prepared dataset:

```csv
run_id,label
<normal-training-run-id>,normal
<oom-training-run-id>,abnormal
<straggler-training-run-id>,abnormal
```

The committed example file refers to the run identifiers used in the authors' experiment. Replace those identifiers when evaluating a newly generated dataset.

Each LogPrompt configuration writes:

```text
logprompt_evaluation_results.csv  # predictions, token counts, and latency
logprompt_metrics.json            # overall and per-fault metrics
logprompt_responses.jsonl         # exact prompts and raw model responses
```

The default output directory encodes the prompt strategy, log representation, model, and rule configuration.

## Calculate LLM Cost

Use the result CSV produced by `evaluate_logprompt.py`:

```bash
python evaluation/calculate_cost.py \
  output/nccl/logprompt/<configuration>/logprompt_evaluation_results.csv
```

Write a cost-annotated CSV:

```bash
python evaluation/calculate_cost.py \
  output/nccl/logprompt/<configuration>/logprompt_evaluation_results.csv \
  --output output/nccl/logprompt/<configuration>/results_with_cost.csv
```

The script contains built-in GPT-5.4 prices used by the study and accepts explicit `--price-input`, `--price-cached-input`, and `--price-output` overrides. Because API prices may change, record the price configuration and evaluation date when reporting new results.

## Third-Party Code

The `evaluation/logbert/` directory contains adapted or vendored code used to evaluate LogBERT, DeepLog, and LogAnomaly. Its upstream license is retained in:

```text
evaluation/logbert/LICENSE
```

When redistributing or modifying that component, preserve its license and attribution requirements.
