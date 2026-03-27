#!/usr/bin/env python3
"""
Unified benchmark runner for NCCL log anomaly detection baselines.

Runs and evaluates one or more of the following models on the same preprocessed
data:
  - logbert
  - deeplog
  - loganomaly

Pipeline
--------
- Optional shared preprocessing (data_process.py, run once for all models)
- Per-model vocab / train / predict flow
- Per-run evaluation against ground-truth labels
- Per-model results saved to output/nccl/
- Optional combined summary across all selected models

Usage
-----
    # All models (default)
    python evaluate.py

    # Single model
    python evaluate.py --model logbert
    python evaluate.py --model deeplog
    python evaluate.py --model loganomaly

    # Multiple models
    python evaluate.py --model deeplog loganomaly

    # Skip preprocessing / training
    python evaluate.py --model logbert --skip-preprocess --skip-train

    # Tune thresholds
    python evaluate.py --logbert-seq-threshold 0.3 --logdeep-seq-threshold 0.4
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

DATASET_DIR        = data_process.DATASET_DIR
OUTPUT_DIR         = Path(logbert_nccl.OUTPUT_DIR)
MODEL_DIR          = Path(logbert_nccl.MODEL_DIR)
OPTIONS            = logbert_nccl.options   # single source of truth for LogBERT hyperparams
DEEPLOG_SAVE_DIR   = OUTPUT_DIR / "deeplog"
LOGANOMALY_SAVE_DIR = OUTPUT_DIR / "loganomaly"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def load_labels() -> pd.DataFrame:
    """
    Combine phase1 (normal) and phase2 (anomalous) label CSVs.
    Adds a binary `true_label` column: 0 = normal, 1 = anomalous.
    """
    phase1_path = DATASET_DIR / "labels" / "phase1_labels.csv"
    phase2_path = DATASET_DIR / "labels" / "phase2_labels.csv"
    if not phase1_path.exists():
        raise FileNotFoundError(f"Missing label file: {phase1_path}")
    if not phase2_path.exists():
        raise FileNotFoundError(f"Missing label file: {phase2_path}")
    phase1 = pd.read_csv(phase1_path)
    phase2 = pd.read_csv(phase2_path)
    phase1["true_label"] = 0
    phase2["true_label"] = 1
    return pd.concat([phase1, phase2], ignore_index=True)


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


def get_abnormal_split_keys(manifest: dict) -> list[str]:
    """
    Return manifest keys for abnormal test splits.

    Preference order:
      1) Per-fault files written by data_process.py: test_abnormal_<fault_type>
      2) Fallback legacy combined split: test_abnormal
    """
    split_keys = sorted(
        k for k in manifest.keys()
        if k.startswith("test_abnormal_") and k != "test_abnormal"
    )
    if split_keys:
        return split_keys
    return ["test_abnormal"]


def fault_type_from_split_key(split_key: str) -> str:
    if split_key == "test_abnormal":
        return "all_abnormal"
    return split_key.replace("test_abnormal_", "", 1)


def compute_per_fault_metrics(
    results_df: pd.DataFrame,
    normal_run_ids: set[str],
    model_name: str,
    seq_threshold: float,
) -> dict:
    """Compute binary metrics for each abnormal fault type vs the same normal runs."""
    abnormal_fault_types = sorted(
        ft for ft in results_df["fault_type"].dropna().astype(str).unique()
        if ft not in {"unknown", "nan"}
    )

    per_fault = {}
    for fault_type in abnormal_fault_types:
        fault_mask = results_df["fault_type"].astype(str) == fault_type
        normal_mask = results_df["run_id"].astype(str).isin(normal_run_ids)
        subset = results_df[normal_mask | fault_mask]
        m = compute_binary_metrics(subset)
        m["model"] = model_name
        m["seq_threshold"] = seq_threshold
        m["fault_type"] = fault_type
        per_fault[fault_type] = m

    return per_fault


# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    print("\n" + "=" * 60)
    print("Shared preprocessing – converting NCCL logs to sequences")
    print("=" * 60)
    data_process.main()


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

    With adaptive_window=True every line produces exactly one window, so the
    i-th result corresponds to the i-th line.  Lines that are too short yield
    None entries to preserve alignment with the manifest.
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

    Results are returned in the **original line order** (no length-based
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

    Logkey criterion: more than `seq_threshold` fraction of masked tokens were
    not among the model's top-`num_candidates` predictions.
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

def evaluate_logbert(options: dict, seq_threshold: float) -> dict:
    """
    Load the trained model, score both test files, aggregate per-sequence
    predictions to per-run labels, and compute classification metrics.

    A run is classified as anomalous if ANY of its sequences is anomalous
    (conservative OR aggregation).
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
    abnormal_split_keys = get_abnormal_split_keys(manifest)

    normal_results = predict_file(model, vocab, options, str(OUTPUT_DIR / "test_normal"), device)

    abnormal_results_by_split = {}
    for split_key in abnormal_split_keys:
        split_path = OUTPUT_DIR / split_key
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing abnormal split file: {split_path}. "
                "Run preprocessing first to regenerate split files."
            )
        abnormal_results_by_split[split_key] = predict_file(
            model, vocab, options, str(split_path), device
        )

    print(f"\n  Normal   sequences scored: {sum(r is not None for r in normal_results)}")
    for split_key in abnormal_split_keys:
        scored = sum(r is not None for r in abnormal_results_by_split[split_key])
        print(f"  Abnormal ({fault_type_from_split_key(split_key)}) sequences scored: {scored}")

    def aggregate_to_runs(seq_results: list, manifest_entries: list) -> dict:
        run_preds: dict = {}
        for res, info in zip(seq_results, manifest_entries):
            rid  = info["run_id"]
            anom = is_anomalous(res, options, seq_threshold)
            run_preds[rid] = run_preds.get(rid, False) or anom
        return {k: int(v) for k, v in run_preds.items()}

    normal_preds   = aggregate_to_runs(normal_results,   manifest["test_normal"])

    abnormal_preds = {}
    for split_key in abnormal_split_keys:
        split_preds = aggregate_to_runs(
            abnormal_results_by_split[split_key],
            manifest[split_key],
        )
        for rid, pred in split_preds.items():
            abnormal_preds[rid] = max(abnormal_preds.get(rid, 0), pred)

    print("\n" + "=" * 60)
    print("LogBERT – comparing against ground-truth labels")
    print("=" * 60)

    labels_df = load_labels()
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
    metrics["by_fault_type"] = compute_per_fault_metrics(
        results_df,
        normal_run_ids=set(normal_preds.keys()),
        model_name="logbert",
        seq_threshold=seq_threshold,
    )

    print(f"\n  Classification metrics (seq_threshold={seq_threshold:.2f})")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 score  : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")

    if metrics["by_fault_type"]:
        print("\n  Per-fault metrics")
        for fault_type, fm in metrics["by_fault_type"].items():
            print(
                f"    {fault_type}: "
                f"P={fm['precision']:.4f} R={fm['recall']:.4f} "
                f"F1={fm['f1']:.4f} Acc={fm['accuracy']:.4f} "
                f"(TP={fm['TP']} TN={fm['TN']} FP={fm['FP']} FN={fm['FN']})"
            )

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
    metrics["by_fault_type"] = compute_per_fault_metrics(
        df,
        normal_run_ids={m["run_id"] for m in normal_manifest},
        model_name=model_name,
        seq_threshold=seq_threshold,
    )

    return df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Per-model runner functions
# ─────────────────────────────────────────────────────────────────────────────

def run_logbert(skip_train: bool, seq_threshold: float) -> dict:
    print("\n" + "=" * 70)
    print("Running LogBERT")
    print("=" * 70)

    if not skip_train:
        build_vocab(OPTIONS)
        train_logbert(OPTIONS)

    return evaluate_logbert(OPTIONS, seq_threshold=seq_threshold)


def run_deeplog(skip_train: bool, labels_df: pd.DataFrame, manifest: dict, seq_threshold: float) -> dict:
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


def run_loganomaly(skip_train: bool, labels_df: pd.DataFrame, manifest: dict, seq_threshold: float) -> dict:
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
            "Unified NCCL log anomaly detection benchmark. "
            "Runs and evaluates LogBERT, DeepLog, and/or LogAnomaly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All models (default)
  python evaluate.py

  # Single baseline
  python evaluate.py --model logbert
  python evaluate.py --model deeplog
  python evaluate.py --model loganomaly

  # Multiple baselines
  python evaluate.py --model deeplog loganomaly

  # Skip preprocessing and training
  python evaluate.py --model logbert --skip-preprocess --skip-train
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
        "--skip-preprocess",
        action="store_true",
        help="Skip shared preprocessing (sequence files must already exist).",
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

    # Shared preprocessing (always runs once, regardless of model selection).
    if not args.skip_preprocess:
        run_preprocessing()

    labels_df = load_labels()
    manifest  = load_manifest()

    all_metrics = []

    if "logbert" in args.model:
        all_metrics.append(run_logbert(args.skip_train, args.logbert_seq_threshold))

    if "deeplog" in args.model:
        all_metrics.append(run_deeplog(args.skip_train, labels_df, manifest, args.logdeep_seq_threshold))

    if "loganomaly" in args.model:
        all_metrics.append(run_loganomaly(args.skip_train, labels_df, manifest, args.logdeep_seq_threshold))

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
