# GF-Bench: GPU Failure Simulation and NCCL Log Anomaly Detection

GF-Bench is a research artifact for generating controlled failures in multi-GPU workloads, collecting rank-local NVIDIA Collective Communications Library (NCCL) logs, and evaluating log-based anomaly detection methods.

The current artifact focuses on two failure families that are observable through NCCL traces:

- **CUDA out-of-memory (OOM)** as a fail-stop failure;
- **persistent GPU straggler** as a fail-slow failure.

The accompanying empirical study evaluates three sequence-based methods—DeepLog, LogAnomaly, and LogBERT—and the LLM-based LogPrompt. It also compares Drain-parsed logs with raw, rank-preserving logs to study the trade-off between detection effectiveness and operational cost.

> **Scope.** The current task is run-level binary anomaly detection over controlled multi-GPU executions. The artifact is not a complete root-cause analysis system, hardware fault simulator, or production monitoring platform.

## Research Artifact at a Glance

| Component | Current implementation |
|---|---|
| Execution framework | PyTorch distributed execution with the NCCL backend |
| Process model | One worker process per visible GPU |
| Normal workload | Iterative local tensor operations followed by AllReduce |
| Fail-stop injection | Repeated GPU allocation until CUDA OOM |
| Fail-slow injection | Configurable delay on one rank before each collective operation |
| Collected artifacts | Per-process NCCL traces, stdout/stderr, metadata, and CSV labels |
| Log representation | Raw NCCL logs and Drain-extracted event templates |
| Evaluated methods | DeepLog, LogAnomaly, LogBERT, LogPrompt |
| Main metrics | Precision, recall, F1, accuracy, token usage, latency, and API cost |

The initial dataset described in the paper contains **220 two-GPU executions**: 100 normal runs, 60 CUDA OOM runs, and 60 straggler runs.

## Repository Structure

```text
nccl_log_rca_bench/
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
│   ├── batch_run_phase2.sh     # Generate the two fault partitions
│   └── clean_data.sh           # Remove generated data and evaluation outputs
├── evaluation/
│   ├── data_process.py         # Drain parsing, sequence construction, splits, and manifest
│   ├── evaluate.py             # Primary entry point for DeepLog, LogAnomaly, and LogBERT
│   ├── evaluate_logprompt.py   # LogPrompt evaluation on parsed or raw logs
│   ├── calculate_cost.py       # LLM token-cost calculation
│   ├── allreduce_count_baseline.py
│   ├── deeplog_nccl.py
│   ├── loganomaly_nccl.py
│   ├── logbert_nccl.py
│   └── logbert/                # Vendored implementations used by the baselines
├── incontext_examples.csv      # Example-run specification for LogPrompt
├── nccl_cot_rules.txt          # NCCL-specific rules for CoT prompting
└── requirements.txt            # Captured development environment
```

## Recommended Reading Order

Readers who want to understand the implementation without running the full experiment should start with the following files:

1. **`workload/run_experiment.py`** — selects the scenario, launches one process per GPU, enforces the global timeout, and writes run-level metadata.
2. **`workload/generic_workload.py`** — defines the common AllReduce workload and the `FaultInjection` interface.
3. **`workload/fault_oom.py`** and **`workload/fault_slow.py`** — implement the two current failure mechanisms.
4. **`scripts/batch_run_phase1.sh`** and **`scripts/batch_run_phase2.sh`** — define the parameter grid used to construct the initial dataset.
5. **`evaluation/data_process.py`** — converts rank-local NCCL logs into Drain templates, integer event sequences, train/test files, and a manifest.
6. **`evaluation/evaluate.py`** — provides the main one-command workflow for the three non-LLM baselines.
7. **`evaluation/evaluate_logprompt.py`** — implements parsed/raw LogPrompt evaluation and records prompts, responses, token usage, and metrics.

The vendored `evaluation/logbert/` directory contains upstream model code and is usually not the best starting point for understanding the GF-Bench-specific workflow.

## Requirements

### Hardware

- Linux environment with Bash;
- NVIDIA driver compatible with CUDA 12.4;
- at least **two visible CUDA GPUs**;
- sufficient available GPU memory for the selected workload.

For the paper configuration, expose exactly two GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

`run_experiment.py` uses all visible GPUs as the distributed world size and rejects environments with fewer than two GPUs.

### Software

The paper environment used:

- PyTorch 2.4.1;
- CUDA 12.4;
- NCCL 2.20.5.

The top-level `requirements.txt` is a captured development environment and includes notebook and Ubuntu-specific packages in addition to the benchmark runtime. On a clean machine, a smaller runtime environment is usually easier to reproduce:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  pandas numpy scipy scikit-learn requests tqdm matplotlib seaborn regex
```

To recreate the broader captured environment instead, use:

```bash
python -m pip install -r requirements.txt
```

Some system-bound entries in the captured requirements may require an Ubuntu environment or system packages.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/cuiboyuan/nccl_log_rca_bench.git
cd nccl_log_rca_bench
```

### 2. Run a small smoke experiment

The shell wrappers are the recommended user-facing entry points for individual executions.

#### Normal run

```bash
bash scripts/run_normal.sh smoke_normal 29601 5 120
```

Arguments:

```text
run_normal.sh RUN_ID MASTER_PORT NUM_ITERATIONS WORKLOAD_TIMEOUT_SECONDS
```

#### Persistent-straggler run

```bash
bash scripts/run_fault_slow.sh smoke_slow 0 29602 6 5 120
```

Arguments:

```text
run_fault_slow.sh RUN_ID FAULT_RANK MASTER_PORT DELAY_SECONDS NUM_ITERATIONS WORKLOAD_TIMEOUT_SECONDS
```

#### CUDA OOM run

```bash
bash scripts/run_fault_oom.sh smoke_oom 0 29603 2 5 120
```

Arguments:

```text
run_fault_oom.sh RUN_ID FAULT_RANK MASTER_PORT INJECTION_ITERATION NUM_ITERATIONS WORKLOAD_TIMEOUT_SECONDS
```

The OOM injection iteration is zero-based. The OOM scenario intentionally exhausts device memory and is expected to terminate one worker with a CUDA OOM exception.

### 3. Inspect the generated artifacts

A run directory contains files such as:

```text
phase1_runs/<run_id>/                 # normal runs
phase2_runs/<run_id>/                 # fault-injected runs
├── metadata.json                     # scenario and execution metadata
├── stdout.log                        # workload output
├── stderr.log                        # process errors
└── nccl_logs_<host>_<pid>.txt        # per-process NCCL TRACE logs

labels/
├── phase1_labels.csv
└── phase2_labels.csv
```

The NCCL environment is configured in `workload/phase2_common.py` with:

```text
NCCL_DEBUG=TRACE
NCCL_DEBUG_SUBSYS=ALL
NCCL_DEBUG_FILE=<run_dir>/nccl_logs_%h_%p.txt
NCCL_P2P_DISABLE=1
```

## Generate the Full Initial Dataset

The initial experiment grid is encoded in two batch scripts.

```bash
export CUDA_VISIBLE_DEVICES=0,1
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

The generation scripts currently write `phase1_runs/`, `phase2_runs/`, and `labels/` at the repository root, while the evaluation code reads them from `dataset/`. Create the expected evaluation layout using symbolic links:

```bash
mkdir -p dataset
ln -sfn ../phase1_runs dataset/phase1_runs
ln -sfn ../phase2_runs dataset/phase2_runs
ln -sfn ../labels dataset/labels
```

The resulting logical layout is:

```text
dataset/
├── phase1_runs -> ../phase1_runs
├── phase2_runs -> ../phase2_runs
└── labels      -> ../labels
```

A copied directory layout may be used instead of symbolic links.

## Reproduce the Non-LLM Evaluation

`evaluation/evaluate.py` is the primary single-file entry point for preprocessing, training, prediction, and metric calculation for DeepLog, LogAnomaly, and LogBERT.

The paper uses an even train/test partition, so pass `0.5` explicitly:

```bash
python evaluation/evaluate.py \
  --normal-train-ratio 0.5 \
  --abnormal-train-ratio 0.5
```

Run one baseline only:

```bash
python evaluation/evaluate.py \
  --model logbert \
  --normal-train-ratio 0.5 \
  --abnormal-train-ratio 0.5
```

Available model names are:

```text
logbert
deeplog
loganomaly
```

Reuse existing preprocessing and model checkpoints:

```bash
python evaluation/evaluate.py \
  --skip-preprocess \
  --skip-train
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

### Optional rule-based baseline

After preprocessing has produced `output/nccl/manifest.json`, run:

```bash
python evaluation/allreduce_count_baseline.py
```

This baseline flags a run when ranks emit different counts of canonical AllReduce entries.

## Reproduce the LogPrompt Evaluation

LogPrompt requires an OpenAI-compatible chat-completions endpoint. Do not commit API credentials.

```bash
export OPENAI_API_KEY="<your-api-key>"
```

Run preprocessing once with the paper split before issuing API requests:

```bash
python evaluation/data_process.py \
  --normal-train-ratio 0.5 \
  --abnormal-train-ratio 0.5
```

### CoT with parsed logs

```bash
python evaluation/evaluate_logprompt.py \
  --model gpt-5.4 \
  --strategy CoT \
  --log-type parsed \
  --cot-rules-file nccl_cot_rules.txt \
  --skip-preprocess
```

### CoT with raw logs

```bash
python evaluation/evaluate_logprompt.py \
  --model gpt-5.4 \
  --strategy CoT \
  --log-type raw \
  --cot-rules-file nccl_cot_rules.txt \
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

## Extending GF-Bench

### Add a new failure mechanism

1. Subclass `FaultInjection` from `workload/generic_workload.py`.
2. Implement:
   - `should_inject(iteration)` to define the target rank and trigger condition;
   - `inject()` to implement the failure behavior.
3. Add a scenario branch in `workload/run_experiment.py`.
4. Add a shell wrapper under `scripts/` with explicit configuration arguments.
5. Extend the label schema only when the new failure requires additional ground-truth fields.

### Add a new workload

Reuse the process setup in `generic_workload.py` and replace the local computation and collective-operation sequence. Preserve the following artifact contract where possible:

- one run directory per execution;
- rank-local NCCL logs;
- `metadata.json` for run configuration and outcome;
- one append-only label row per run;
- stable `run_id` values connecting labels, logs, and evaluation outputs.

### Add a new detector

A detector should consume either the raw run-level logs or the representation produced by `evaluation/data_process.py`, then emit at least:

```text
run_id
true_label
pred_label
fault_type
```

Use the same normal test set for each fault-specific evaluation so that false-positive counts remain comparable.

## Reproducibility Notes

- The paper configuration uses two NVIDIA A40 GPUs, CUDA 12.4, NCCL 2.20.5, and PyTorch 2.4.1.
- Model scripts set the random seed to `1234`, but complete determinism is not guaranteed across CUDA kernels, drivers, and hardware.
- LogPrompt results may vary with model revision, API implementation, prompt serialization, and service-side nondeterminism.
- Raw-log inputs can be substantially larger and more expensive than Drain-parsed inputs.
- The current split procedure is deterministic with respect to sorted input files, but newly generated timestamped run identifiers can alter ordering. Archive `manifest.json` with reported results.
- Use distinct `MASTER_PORT` values for concurrent experiments.
- The OOM scenario intentionally destabilizes one worker process. Run it in an isolated research environment rather than on shared production workloads.

## Cleaning Generated Artifacts

The cleanup script removes generated runs, labels, and evaluation outputs:

```bash
bash scripts/clean_data.sh
```

This operation is destructive. Archive any dataset or result files needed for a paper artifact before running it.

## Citation

Please cite the accompanying GF-Bench paper when using this artifact. The final bibliographic entry should be added here after publication.

```bibtex
% Citation to be updated after publication.
```

## Third-Party Code

The `evaluation/logbert/` directory contains adapted or vendored code used to evaluate LogBERT, DeepLog, and LogAnomaly. Its upstream license is retained in:

```text
evaluation/logbert/LICENSE
```

When redistributing or modifying that component, preserve its license and attribution requirements.

## License

A repository-wide license is not currently included. Until a root `LICENSE` file is added, the presence of source code in this repository should not be interpreted as granting general rights to use, modify, or redistribute the project beyond applicable law and the separately licensed third-party components.

## Artifact Limitations

The current release is intentionally narrow:

- two visible GPUs in the reported experiments;
- one synthetic data-parallel workload;
- two controlled failure families;
- one hardware/software configuration;
- run-level anomaly detection rather than full root-cause localization;
- in-distribution evaluation over generated workloads and fault families.

These constraints should be considered when interpreting results or comparing GF-Bench with production-scale distributed training systems.
