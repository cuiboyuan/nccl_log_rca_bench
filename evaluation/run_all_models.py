#!/usr/bin/env python3
"""
Unified benchmark runner for NCCL log anomaly detection baselines.

This script runs and evaluates all three models on the same preprocessed data:
  1) LogBERT
  2) DeepLog
  3) LogAnomaly

Pipeline
--------
- Optional preprocessing (shared once for all models)
- Per-model vocab/train/predict flow
- Per-run evaluation against ground-truth labels
- Combined summary saved to output/nccl/

Usage
-----
cd /workspace/nccl_log_rca_bench/evaluation
python run_all_models.py

Optional flags:
  --models logbert deeplog loganomaly
  --skip-preprocess
  --skip-train
  --logbert-seq-threshold 0.5
  --logdeep-seq-threshold 0.5
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Local imports (all scripts live in evaluation/)
import run_evaluation as logbert_eval
import logbert_nccl

EVAL_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = Path(logbert_nccl.OUTPUT_DIR)
DATASET_DIR = Path("/workspace/dataset_nccl_log")
DEEPLOG_SAVE_DIR = OUTPUT_DIR / "deeplog"
LOGANOMALY_SAVE_DIR = OUTPUT_DIR / "loganomaly"


def run_cmd(cmd: list[str]) -> None:
    """Run a command and stream output; fail fast on non-zero exit."""
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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_logdeep_saved_results(
    model_name: str,
    save_dir: Path,
    labels_df: pd.DataFrame,
    manifest: dict,
    seq_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate DeepLog/LogAnomaly predictions from saved result pickles.

    Each sequence result dict is expected to contain:
      - logkey_anomaly
      - predicted_logkey

    Sequence is anomalous when:
      logkey_anomaly / predicted_logkey > seq_threshold

    Run-level label uses OR aggregation across that run's two rank logs.
    """
    normal_path = save_dir / "test_normal_results"
    abnormal_path = save_dir / "test_abnormal_results"

    if not normal_path.exists() or not abnormal_path.exists():
        raise FileNotFoundError(
            f"Missing prediction outputs under {save_dir}. "
            "Run model predict first."
        )

    with open(normal_path, "rb") as f:
        normal_results = pickle.load(f)
    with open(abnormal_path, "rb") as f:
        abnormal_results = pickle.load(f)

    normal_manifest = manifest["test_normal"]
    abnormal_manifest = manifest["test_abnormal"]

    # Guard against mismatches if min_len filtering removed some rows.
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
        errors = int(res.get("logkey_anomaly", 0))
        ratio = errors / predicted
        return ratio > seq_threshold

    run_preds: dict[str, bool] = {}

    for res, meta in zip(normal_results, normal_manifest):
        rid = meta["run_id"]
        run_preds[rid] = run_preds.get(rid, False) or seq_is_anomaly(res)

    for res, meta in zip(abnormal_results, abnormal_manifest):
        rid = meta["run_id"]
        run_preds[rid] = run_preds.get(rid, False) or seq_is_anomaly(res)

    label_map = dict(zip(labels_df["run_id"], labels_df["true_label"]))
    meta_map = labels_df.set_index("run_id").to_dict("index")

    rows = []
    for run_id in sorted(run_preds):
        pred = int(run_preds[run_id])
        true = int(label_map.get(run_id, -1))
        scenario = meta_map.get(run_id, {}).get("scenario", "unknown")
        fault_type = meta_map.get(run_id, {}).get("fault_type", "unknown")
        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "fault_type": fault_type,
                "true_label": true,
                "pred_label": pred,
                "correct": int(pred == true),
            }
        )

    df = pd.DataFrame(rows)
    metrics = compute_binary_metrics(df)
    metrics["seq_threshold"] = seq_threshold
    metrics["model"] = model_name

    return df, metrics


def run_logbert(skip_train: bool, seq_threshold: float) -> dict:
    print("\n" + "=" * 70)
    print("Running LogBERT")
    print("=" * 70)

    if not skip_train:
        logbert_eval.build_vocab(logbert_eval.OPTIONS)
        logbert_eval.train_logbert(logbert_eval.OPTIONS)

    metrics = logbert_eval.evaluate(logbert_eval.OPTIONS, seq_threshold=seq_threshold)
    metrics["model"] = "logbert"
    return metrics


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LogBERT, DeepLog, and LogAnomaly in one command")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logbert", "deeplog", "loganomaly"],
        choices=["logbert", "deeplog", "loganomaly"],
        help="Subset of models to run",
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip shared preprocessing")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training and only run predict+evaluate")
    parser.add_argument(
        "--logbert-seq-threshold",
        type=float,
        default=0.5,
        help="LogBERT sequence anomaly threshold",
    )
    parser.add_argument(
        "--logdeep-seq-threshold",
        type=float,
        default=0.5,
        help="DeepLog/LogAnomaly sequence anomaly threshold (error ratio)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        print("\n" + "=" * 70)
        print("Shared preprocessing")
        print("=" * 70)
        logbert_eval.run_preprocessing()

    labels_df = logbert_eval.load_labels(DATASET_DIR)
    manifest = load_manifest()

    all_metrics = []

    if "logbert" in args.models:
        all_metrics.append(run_logbert(args.skip_train, args.logbert_seq_threshold))

    if "deeplog" in args.models:
        all_metrics.append(run_deeplog(args.skip_train, labels_df, manifest, args.logdeep_seq_threshold))

    if "loganomaly" in args.models:
        all_metrics.append(run_loganomaly(args.skip_train, labels_df, manifest, args.logdeep_seq_threshold))

    summary_df = pd.DataFrame(all_metrics)
    summary_cols = [
        "model",
        "seq_threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    summary_df = summary_df[summary_cols]

    print("\n" + "=" * 70)
    print("Combined summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    summary_csv = OUTPUT_DIR / "all_models_summary.csv"
    summary_json = OUTPUT_DIR / "all_models_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved combined summary CSV:  {summary_csv}")
    print(f"Saved combined summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
