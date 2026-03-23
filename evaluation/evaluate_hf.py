#!/usr/bin/env python3
"""
HuggingFace-backed benchmark runner for NCCL log anomaly detection baselines.

Identical pipeline to evaluate.py, but loads log data and labels directly
from the HuggingFace dataset (bryancui/nccl-log-rca-bench) instead of the
local filesystem.

Usage
-----
    # All models (default)
    python evaluate_hf.py

    # Single model
    python evaluate_hf.py --model logbert
    python evaluate_hf.py --model deeplog
    python evaluate_hf.py --model loganomaly

    # Multiple models
    python evaluate_hf.py --model deeplog loganomaly

    # Skip preprocessing / training
    python evaluate_hf.py --model logbert --skip-preprocess --skip-train

    # Custom HF dataset
    python evaluate_hf.py --hf-dataset bryancui/nccl-log-rca-bench

    # Tune thresholds
    python evaluate_hf.py --logbert-seq-threshold 0.3 --logdeep-seq-threshold 0.4
"""

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent.resolve()

# Allow 'import data_process' and 'import logbert_nccl'
sys.path.insert(0, str(EVAL_DIR))
# Allow 'from bert_pytorch import …' and 'from logparser import …'
sys.path.insert(0, str(EVAL_DIR / "logbert"))

import data_process                                        # noqa: E402
import logbert_nccl                                        # noqa: E402
from bert_pytorch.dataset import WordVocab, LogDataset     # noqa: E402
from bert_pytorch.dataset.sample import fixed_window       # noqa: E402
from bert_pytorch.dataset.utils import seed_everything     # noqa: E402

seed_everything(seed=1234)

# ── Shared paths / options ────────────────────────────────────────────────────

OUTPUT_DIR          = Path(logbert_nccl.OUTPUT_DIR)
MODEL_DIR           = Path(logbert_nccl.MODEL_DIR)
OPTIONS             = logbert_nccl.options   # single source of truth for LogBERT hyperparams
DEEPLOG_SAVE_DIR    = OUTPUT_DIR / "deeplog"
LOGANOMALY_SAVE_DIR = OUTPUT_DIR / "loganomaly"

HF_DATASET_ID  = "bryancui/nccl-log-rca-bench"
HF_CONFIG_NAME = "nccl_log_benchmark"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (identical to evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command, streaming output; fail fast on non-zero exit."""
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(EVAL_DIR), check=True)


def load_manifest() -> dict:
    manifest_path = OUTPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Run preprocessing first."
        )
    with open(manifest_path, "r") as f:
        return json.load(f)


def compute_binary_metrics(df: pd.DataFrame) -> dict:
    tp = int(((df["true_label"] == 1) & (df["pred_label"] == 1)).sum())
    tn = int(((df["true_label"] == 0) & (df["pred_label"] == 0)).sum())
    fp = int(((df["true_label"] == 0) & (df["pred_label"] == 1)).sum())
    fn = int(((df["true_label"] == 1) & (df["pred_label"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy  = (tp + tn) / len(df) if len(df) > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HF dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_dataset(dataset_id: str) -> list:
    """
    Download (or load from cache) the HF dataset and return all records as a
    list of dicts.

    Each record contains at minimum:
      phase      – "phase1" (normal) or "phase2" (anomalous)
      run_id     – unique run identifier
      scenario   – scenario label (e.g. "normal", "delay", "oom")
      fault_type – fault type label (e.g. "no_fault", "packet_delay", "oom")
      nccl_logs  – list of per-rank raw NCCL log strings
    """
    print(f"\nLoading HuggingFace dataset: {dataset_id} (config: {HF_CONFIG_NAME})")
    ds = load_dataset(dataset_id, HF_CONFIG_NAME, split="train")
    records = list(ds)
    phase_counts = {}
    for r in records:
        phase_counts[r["phase"]] = phase_counts.get(r["phase"], 0) + 1
    print(f"  Total records loaded : {len(records)}")
    for phase, count in sorted(phase_counts.items()):
        print(f"    {phase}: {count} runs")
    return records


def load_labels_hf(records: list) -> pd.DataFrame:
    """
    Build a labels DataFrame from HF dataset records.
    phase1 → true_label = 0 (normal)
    phase2 → true_label = 1 (anomalous)
    """
    rows = []
    for rec in records:
        rows.append({
            "run_id":     rec["run_id"],
            "scenario":   rec.get("scenario",   "unknown"),
            "fault_type": rec.get("fault_type", "unknown"),
            "true_label": 0 if rec["phase"] == "phase1" else 1,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HF-aware preprocessing (replaces data_process.main())
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_hf(records: list) -> None:
    """
    Equivalent of data_process.main(), operating on in-memory log strings
    from HF dataset records instead of local log files.

    Each record's `nccl_logs` field is a list of per-rank raw log strings,
    corresponding to the per-rank nccl_logs_*.txt files in the local dataset.

    Produces exactly the same output files as data_process.main():
      output/nccl/train
      output/nccl/test_normal
      output/nccl/test_abnormal
      output/nccl/manifest.json
      output/nccl/event_map.json
    """
    from logparser import Drain  # noqa: F401 – ensures Drain is importable

    drain_input_dir  = OUTPUT_DIR / "drain_input"
    drain_output_dir = OUTPUT_DIR / "drain_output"
    drain_input_dir.mkdir(parents=True, exist_ok=True)
    drain_output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_logs_path = drain_input_dir / "nccl_all.log"

    print("=" * 60)
    print("Step 1/6  Partitioning records by phase")
    print("=" * 60)
    phase1_records = sorted(
        [r for r in records if r["phase"] == "phase1"], key=lambda r: r["run_id"]
    )
    phase2_records = sorted(
        [r for r in records if r["phase"] == "phase2"], key=lambda r: r["run_id"]
    )
    print(f"  Phase1 (normal)    : {len(phase1_records)} runs")
    print(f"  Phase2 (anomalous) : {len(phase2_records)} runs")

    print("\n" + "=" * 60)
    print("Step 2/6  Stripping headers and writing unified log file")
    print("=" * 60)
    file_manifest = []
    line_num = 0

    # Preserve the same phase ordering as data_process.collect_log_files
    # (phase1 / normal first, then phase2 / anomalous).
    with open(all_logs_path, "w", encoding="utf-8") as out_f:
        for phase_name, phase_recs in [("phase1", phase1_records), ("phase2", phase2_records)]:
            for rec in phase_recs:
                run_id    = rec["run_id"]
                nccl_logs = rec.get("nccl_logs") or []
                for log_idx, log_content in enumerate(nccl_logs):
                    start = line_num
                    for raw_line in io.StringIO(log_content):
                        content = data_process.strip_header(raw_line)
                        if data_process.is_valid_nccl_line(content):
                            out_f.write(content + "\n")
                            line_num += 1
                    file_manifest.append({
                        "start":    start,
                        "end":      line_num,
                        "phase":    phase_name,
                        "run_id":   run_id,
                        "log_file": f"nccl_logs_{log_idx}.txt",
                    })

    total_lines = file_manifest[-1]["end"] if file_manifest else 0
    print(f"  Total preprocessed NCCL lines : {total_lines}")

    print("\n" + "=" * 60)
    print("Step 3/6  Running Drain log parser")
    print("=" * 60)
    data_process.run_drain(all_logs_path, drain_output_dir)

    structured_csv = drain_output_dir / "nccl_all.log_structured.csv"
    templates_csv  = drain_output_dir / "nccl_all.log_templates.csv"

    print("\n" + "=" * 60)
    print("Step 4/6  Building event-ID mapping")
    print("=" * 60)
    event_map = data_process.build_event_map(templates_csv)
    with open(OUTPUT_DIR / "event_map.json", "w") as f:
        json.dump(event_map, f, indent=2)
    print(f"  Unique event templates : {len(event_map)}")

    print("\n" + "=" * 60)
    print("Step 5/6  Building per-file event sequences")
    print("=" * 60)
    sequences = data_process.build_sequences(structured_csv, file_manifest, event_map)
    print(f"  Total sequences : {len(sequences)}")

    print("\n" + "=" * 60)
    print("Step 6/6  Writing LogBERT input files")
    print("=" * 60)
    data_process.write_output_files(sequences, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# LogBERT – vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(options: dict) -> WordVocab:
    print("\n" + "=" * 60)
    print("LogBERT – building vocabulary")
    print("=" * 60)
    with open(options["train_vocab"], "r") as f:
        texts = f.readlines()
    vocab = WordVocab(texts, max_size=None, min_freq=1)
    print(f"  Vocabulary size: {len(vocab)}")
    vocab.save_vocab(options["vocab_path"])
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# LogBERT – training
# ─────────────────────────────────────────────────────────────────────────────

def train_logbert(options: dict) -> None:
    from bert_pytorch import Trainer  # local import to keep startup fast

    print("\n" + "=" * 60)
    print("LogBERT – training")
    print("=" * 60)
    os.makedirs(options["model_dir"], exist_ok=True)
    Trainer(options).train()


# ─────────────────────────────────────────────────────────────────────────────
# LogBERT – prediction (preserves line order)
# ─────────────────────────────────────────────────────────────────────────────

def _load_sequences_in_order(file_path: str, options: dict):
    """
    Read a LogBERT sequence file and convert each line to a (logkeys, times)
    pair using the same fixed_window logic as Predictor.generate_test.
    """
    log_seqs, time_seqs = [], []
    with open(file_path, "r") as f:
        for line in f.readlines():
            lk, tm = fixed_window(
                line,
                options["window_size"],
                adaptive_window=options["adaptive_window"],
                seq_len=options["seq_len"],
                min_len=options["min_len"],
            )
            if lk:
                log_seqs.append(lk[0])
                time_seqs.append(tm[0])
            else:
                log_seqs.append(None)
                time_seqs.append(None)
    return log_seqs, time_seqs


def predict_file(
    model: torch.nn.Module,
    vocab: WordVocab,
    options: dict,
    file_path: str,
    device: torch.device,
) -> list:
    """
    Score every sequence in *file_path* with LogBERT.

    Results are returned in the original line order (no length-based
    reordering).  Each entry is a dict
        {undetected_tokens, masked_tokens, deepSVDD_label}
    or None if the line produced no valid sequence.
    """
    log_seqs, time_seqs = _load_sequences_in_order(file_path, options)

    valid_idx  = [i for i, s in enumerate(log_seqs) if s is not None]
    valid_log  = [log_seqs[i]  for i in valid_idx]
    valid_time = [time_seqs[i] for i in valid_idx]

    if not valid_idx:
        return [None] * len(log_seqs)

    log_arr  = np.array(valid_log,  dtype=object)
    time_arr = np.array(valid_time, dtype=object)

    dataset = LogDataset(
        log_arr, time_arr, vocab,
        seq_len=options["seq_len"],
        corpus_lines=options["corpus_lines"],
        on_memory=options["on_memory"],
        predict_mode=True,
        mask_ratio=options["mask_ratio"],
    )
    loader = DataLoader(
        dataset,
        batch_size=options["batch_size"],
        num_workers=options["num_workers"],
        collate_fn=dataset.collate_fn,
    )

    valid_results = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = {k: v.to(device) for k, v in data.items()}
            result = model(data["bert_input"], data["time_input"])
            mask_lm_output = result["logkey_output"]   # [B, seq_len, vocab_size]

            for i in range(len(data["bert_label"])):
                seq_res = {
                    "undetected_tokens": 0,
                    "masked_tokens": 0,
                    "deepSVDD_label": 0,
                }

                mask_index = data["bert_label"][i] > 0
                num_masked = int(torch.sum(mask_index).item())
                seq_res["masked_tokens"] = num_masked

                if options["is_logkey"] and num_masked > 0:
                    masked_out = mask_lm_output[i][mask_index]      # [M, V]
                    masked_lbl = data["bert_label"][i][mask_index]   # [M]
                    undetected = 0
                    for j, token in enumerate(masked_lbl):
                        top_k = torch.argsort(-masked_out[j])[: options["num_candidates"]]
                        if token not in top_k:
                            undetected += 1
                    seq_res["undetected_tokens"] = undetected

                valid_results.append(seq_res)

    # Reconstruct full-length result list preserving original line order.
    all_results: list = [None] * len(log_seqs)
    for idx, res in zip(valid_idx, valid_results):
        all_results[idx] = res

    return all_results


def is_anomalous(seq_result: dict | None, options: dict, seq_threshold: float) -> bool:
    """
    Return True when a single-sequence result should be classified as anomalous.
    """
    if seq_result is None:
        return False
    if options["is_logkey"]:
        m = seq_result["masked_tokens"]
        if m > 0 and seq_result["undetected_tokens"] > m * seq_threshold:
            return True
    if options["hypersphere_loss_test"] and seq_result.get("deepSVDD_label", 0):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# LogBERT – full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_logbert(options: dict, labels_df: pd.DataFrame, seq_threshold: float) -> dict:
    """
    Load the trained model, score both test files, aggregate per-sequence
    predictions to per-run labels, and compute classification metrics.
    """
    device = torch.device(options["device"])

    print("\n" + "=" * 60)
    print("LogBERT – running predictions")
    print("=" * 60)

    model_path = options["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run training first (or use --skip-train with an existing model)."
        )

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    vocab    = WordVocab.load_vocab(options["vocab_path"])
    manifest = load_manifest()

    normal_results   = predict_file(model, vocab, options, str(OUTPUT_DIR / "test_normal"),   device)
    abnormal_results = predict_file(model, vocab, options, str(OUTPUT_DIR / "test_abnormal"), device)

    print(f"\n  Normal   sequences scored: {sum(r is not None for r in normal_results)}")
    print(f"  Abnormal sequences scored: {sum(r is not None for r in abnormal_results)}")

    def aggregate_to_runs(seq_results: list, manifest_entries: list) -> dict:
        run_preds: dict = {}
        for res, info in zip(seq_results, manifest_entries):
            rid  = info["run_id"]
            anom = is_anomalous(res, options, seq_threshold)
            run_preds[rid] = run_preds.get(rid, False) or anom
        return {k: int(v) for k, v in run_preds.items()}

    normal_preds   = aggregate_to_runs(normal_results,   manifest["test_normal"])
    abnormal_preds = aggregate_to_runs(abnormal_results, manifest["test_abnormal"])

    print("\n" + "=" * 60)
    print("LogBERT – comparing against ground-truth labels")
    print("=" * 60)

    label_map = dict(zip(labels_df["run_id"], labels_df["true_label"]))
    meta_map  = labels_df.set_index("run_id").to_dict("index")

    all_preds = {**normal_preds, **abnormal_preds}

    rows = []
    for run_id in sorted(all_preds):
        pred       = all_preds[run_id]
        true       = label_map.get(run_id, -1)
        fault_type = meta_map.get(run_id, {}).get("fault_type", "unknown")
        scenario   = meta_map.get(run_id, {}).get("scenario",   "unknown")
        rows.append({
            "run_id":     run_id,
            "scenario":   scenario,
            "fault_type": fault_type,
            "true_label": true,
            "pred_label": pred,
            "correct":    int(pred == true),
        })

    results_df = pd.DataFrame(rows)
    print("\nPer-run results:")
    print(results_df.to_string(index=False))

    metrics = compute_binary_metrics(results_df)
    metrics["seq_threshold"] = seq_threshold
    metrics["model"]         = "logbert"

    print(f"\n  Classification metrics (seq_threshold={seq_threshold:.2f})")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 score  : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")

    results_path = OUTPUT_DIR / "logbert_evaluation_results.csv"
    metrics_path = OUTPUT_DIR / "logbert_metrics.json"
    results_df.to_csv(results_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Per-run results -> {results_path}")
    print(f"  Metrics         -> {metrics_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DeepLog / LogAnomaly – evaluation from saved prediction pickles
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_logdeep_saved_results(
    model_name: str,
    save_dir: Path,
    labels_df: pd.DataFrame,
    manifest: dict,
    seq_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate DeepLog or LogAnomaly predictions from saved result pickles.

    Each sequence result dict is expected to contain:
      - logkey_anomaly:    number of anomalous log keys detected
      - predicted_logkey:  total number of log keys predicted

    A sequence is anomalous when:
        logkey_anomaly / predicted_logkey > seq_threshold

    Run-level label uses OR aggregation across all sequences for that run.
    """
    normal_path   = save_dir / "test_normal_results"
    abnormal_path = save_dir / "test_abnormal_results"

    if not normal_path.exists() or not abnormal_path.exists():
        raise FileNotFoundError(
            f"Missing prediction outputs under {save_dir}. "
            "Run model predict first."
        )

    with open(normal_path,   "rb") as f:
        normal_results = pickle.load(f)
    with open(abnormal_path, "rb") as f:
        abnormal_results = pickle.load(f)

    normal_manifest   = manifest["test_normal"]
    abnormal_manifest = manifest["test_abnormal"]

    if len(normal_results) != len(normal_manifest):
        print(
            f"Warning: {model_name} normal result count ({len(normal_results)}) "
            f"!= manifest count ({len(normal_manifest)}). Using zipped prefix."
        )
    if len(abnormal_results) != len(abnormal_manifest):
        print(
            f"Warning: {model_name} abnormal result count ({len(abnormal_results)}) "
            f"!= manifest count ({len(abnormal_manifest)}). Using zipped prefix."
        )

    def seq_is_anomaly(res: dict) -> bool:
        predicted = max(int(res.get("predicted_logkey", 0)), 1)
        errors    = int(res.get("logkey_anomaly", 0))
        return (errors / predicted) > seq_threshold

    run_preds: dict[str, bool] = {}

    for res, meta in zip(normal_results, normal_manifest):
        rid = meta["run_id"]
        run_preds[rid] = run_preds.get(rid, False) or seq_is_anomaly(res)

    for res, meta in zip(abnormal_results, abnormal_manifest):
        rid = meta["run_id"]
        run_preds[rid] = run_preds.get(rid, False) or seq_is_anomaly(res)

    label_map = dict(zip(labels_df["run_id"], labels_df["true_label"]))
    meta_map  = labels_df.set_index("run_id").to_dict("index")

    rows = []
    for run_id in sorted(run_preds):
        pred       = int(run_preds[run_id])
        true       = int(label_map.get(run_id, -1))
        scenario   = meta_map.get(run_id, {}).get("scenario",   "unknown")
        fault_type = meta_map.get(run_id, {}).get("fault_type", "unknown")
        rows.append({
            "run_id":     run_id,
            "scenario":   scenario,
            "fault_type": fault_type,
            "true_label": true,
            "pred_label": pred,
            "correct":    int(pred == true),
        })

    df      = pd.DataFrame(rows)
    metrics = compute_binary_metrics(df)
    metrics["seq_threshold"] = seq_threshold
    metrics["model"]         = model_name

    return df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Per-model runner functions
# ─────────────────────────────────────────────────────────────────────────────

def run_logbert(skip_train: bool, labels_df: pd.DataFrame, seq_threshold: float) -> dict:
    print("\n" + "=" * 70)
    print("Running LogBERT")
    print("=" * 70)

    if not skip_train:
        build_vocab(OPTIONS)
        train_logbert(OPTIONS)

    return evaluate_logbert(OPTIONS, labels_df, seq_threshold=seq_threshold)


def run_deeplog(
    skip_train: bool,
    labels_df: pd.DataFrame,
    manifest: dict,
    seq_threshold: float,
) -> dict:
    print("\n" + "=" * 70)
    print("Running DeepLog")
    print("=" * 70)

    if not skip_train:
        run_cmd([sys.executable, "deeplog_nccl.py", "vocab"])
        run_cmd([sys.executable, "deeplog_nccl.py", "train"])

    run_cmd([sys.executable, "deeplog_nccl.py", "predict"])

    df, metrics = evaluate_logdeep_saved_results(
        model_name="deeplog",
        save_dir=DEEPLOG_SAVE_DIR,
        labels_df=labels_df,
        manifest=manifest,
        seq_threshold=seq_threshold,
    )

    df.to_csv(OUTPUT_DIR / "deeplog_evaluation_results.csv", index=False)
    with open(OUTPUT_DIR / "deeplog_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(df.to_string(index=False))
    print(f"DeepLog metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def run_loganomaly(
    skip_train: bool,
    labels_df: pd.DataFrame,
    manifest: dict,
    seq_threshold: float,
) -> dict:
    print("\n" + "=" * 70)
    print("Running LogAnomaly")
    print("=" * 70)

    if not skip_train:
        run_cmd([sys.executable, "loganomaly_nccl.py", "vocab"])
        run_cmd([sys.executable, "loganomaly_nccl.py", "train"])

    run_cmd([sys.executable, "loganomaly_nccl.py", "predict"])

    df, metrics = evaluate_logdeep_saved_results(
        model_name="loganomaly",
        save_dir=LOGANOMALY_SAVE_DIR,
        labels_df=labels_df,
        manifest=manifest,
        seq_threshold=seq_threshold,
    )

    df.to_csv(OUTPUT_DIR / "loganomaly_evaluation_results.csv", index=False)
    with open(OUTPUT_DIR / "loganomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(df.to_string(index=False))
    print(f"LogAnomaly metrics: {json.dumps(metrics, indent=2)}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HuggingFace-backed NCCL log anomaly detection benchmark. "
            "Loads data from HF dataset instead of the local filesystem. "
            "Runs and evaluates LogBERT, DeepLog, and/or LogAnomaly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All models (default)
  python evaluate_hf.py

  # Single baseline
  python evaluate_hf.py --model logbert
  python evaluate_hf.py --model deeplog
  python evaluate_hf.py --model loganomaly

  # Multiple baselines
  python evaluate_hf.py --model deeplog loganomaly

  # Skip preprocessing and training
  python evaluate_hf.py --model logbert --skip-preprocess --skip-train

  # Custom HF dataset
  python evaluate_hf.py --hf-dataset bryancui/nccl-log-rca-bench
        """,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["logbert", "deeplog", "loganomaly"],
        choices=["logbert", "deeplog", "loganomaly"],
        metavar="MODEL",
        help=(
            "Which baseline(s) to evaluate: logbert, deeplog, loganomaly. "
            "Defaults to all three. Can specify multiple values."
        ),
    )
    parser.add_argument(
        "--hf-dataset",
        default=HF_DATASET_ID,
        metavar="REPO_ID",
        help=(
            f"HuggingFace dataset repository ID (default: {HF_DATASET_ID})."
        ),
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip HF data download and preprocessing (sequence files must already exist).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training and go straight to predict + evaluate.",
    )
    parser.add_argument(
        "--logbert-seq-threshold",
        type=float,
        default=0.5,
        help=(
            "LogBERT anomaly threshold: fraction of masked tokens the model "
            "fails to predict before a sequence is flagged (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--logdeep-seq-threshold",
        type=float,
        default=0.5,
        help=(
            "DeepLog / LogAnomaly anomaly threshold: ratio of anomalous log "
            "keys to total predicted log keys (default: 0.5)."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load HF dataset first (needed for both preprocessing and labels).
    records = load_hf_dataset(args.hf_dataset)

    # HF-aware preprocessing (runs once regardless of model selection).
    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("HF preprocessing – converting NCCL logs to sequences")
        print("=" * 60)
        run_preprocessing_hf(records)

    labels_df = load_labels_hf(records)
    manifest  = load_manifest()

    all_metrics = []

    if "logbert" in args.model:
        all_metrics.append(
            run_logbert(args.skip_train, labels_df, args.logbert_seq_threshold)
        )

    if "deeplog" in args.model:
        all_metrics.append(
            run_deeplog(args.skip_train, labels_df, manifest, args.logdeep_seq_threshold)
        )

    if "loganomaly" in args.model:
        all_metrics.append(
            run_loganomaly(args.skip_train, labels_df, manifest, args.logdeep_seq_threshold)
        )

    # Combined summary (only meaningful when more than one model is evaluated).
    if len(all_metrics) > 1:
        summary_cols = [
            "model", "seq_threshold",
            "TP", "TN", "FP", "FN",
            "precision", "recall", "f1", "accuracy",
        ]
        summary_df = pd.DataFrame(all_metrics)[summary_cols]

        print("\n" + "=" * 70)
        print("Combined summary")
        print("=" * 70)
        print(summary_df.to_string(index=False))

        summary_csv  = OUTPUT_DIR / "all_models_summary.csv"
        summary_json = OUTPUT_DIR / "all_models_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        with open(summary_json, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\nSaved combined summary CSV:  {summary_csv}")
        print(f"Saved combined summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
