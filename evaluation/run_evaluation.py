#!/usr/bin/env python3
"""
End-to-end LogBERT evaluation pipeline for the NCCL log dataset.

Pipeline
--------
1. Preprocess NCCL log files  → LogBERT sequence files  (data_process.py)
2. Build WordVocab from training sequences
3. Train LogBERT on phase1 (normal) sequences
4. Predict anomalies on test_normal / test_abnormal, preserving line order
5. Aggregate per-sequence predictions to per-run labels
6. Compare against ground-truth labels and report classification metrics

Usage
-----
Run the full pipeline:
    python run_evaluation.py

Skip preprocessing (if sequence files already exist):
    python run_evaluation.py --skip-preprocess

Skip training (use an already-trained model):
    python run_evaluation.py --skip-train

Tune the detection threshold:
    python run_evaluation.py --seq-threshold 0.3

Output
------
  <workspace>/output/nccl/evaluation_results.csv  – per-run predictions
  <workspace>/output/nccl/metrics.json             – aggregate metrics
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Path setup (must happen before local imports) ─────────────────────────────

EVAL_DIR = Path(__file__).parent.resolve()

# Allow 'import data_process' and 'import logbert_nccl'
sys.path.insert(0, str(EVAL_DIR))
# Allow 'from bert_pytorch import …' and 'from logparser import …'
sys.path.insert(0, str(EVAL_DIR / "logbert"))

import data_process                                  # noqa: E402
import logbert_nccl                                  # noqa: E402
from bert_pytorch.dataset import WordVocab, LogDataset  # noqa: E402
from bert_pytorch.dataset.sample import fixed_window    # noqa: E402
from bert_pytorch.dataset.utils import seed_everything  # noqa: E402

seed_everything(seed=1234)

# ── Shared paths / options ────────────────────────────────────────────────────

DATASET_DIR = Path("/workspace/dataset_nccl_log")
OUTPUT_DIR  = Path(logbert_nccl.OUTPUT_DIR)
MODEL_DIR   = Path(logbert_nccl.MODEL_DIR)
OPTIONS     = logbert_nccl.options   # single source of truth for hyperparams

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    print("\n" + "=" * 60)
    print("STEP 1 – Preprocessing NCCL logs")
    print("=" * 60)
    data_process.main()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(options: dict) -> WordVocab:
    print("\n" + "=" * 60)
    print("STEP 2 – Building vocabulary")
    print("=" * 60)
    with open(options["train_vocab"], "r") as f:
        texts = f.readlines()
    vocab = WordVocab(texts, max_size=None, min_freq=1)
    print(f"  Vocabulary size: {len(vocab)}")
    vocab.save_vocab(options["vocab_path"])
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Training
# ─────────────────────────────────────────────────────────────────────────────

def train_logbert(options: dict) -> None:
    from bert_pytorch import Trainer  # local import to keep startup fast

    print("\n" + "=" * 60)
    print("STEP 3 – Training LogBERT")
    print("=" * 60)
    os.makedirs(options["model_dir"], exist_ok=True)
    Trainer(options).train()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Custom prediction preserving line order
# ─────────────────────────────────────────────────────────────────────────────

def _load_sequences_in_order(file_path: str, options: dict):
    """
    Read a LogBERT sequence file and convert each line to a (logkeys, times)
    pair using the same fixed_window logic as Predictor.generate_test.

    With adaptive_window=True every line produces exactly one window (the
    whole session), so the i-th result corresponds to the i-th line.

    Returns two parallel lists whose entries are either numpy arrays (valid
    sequence) or None (line was too short and filtered out).
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
                # adaptive_window → exactly one window per line
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
    Run LogBERT anomaly scoring on every sequence in *file_path*.

    Results are returned in the **original line order** of the file (no
    length-based reordering).  Each entry is either a dict:
        {undetected_tokens, masked_tokens, deepSVDD_label}
    or None if the line produced no valid sequence.
    """
    log_seqs, time_seqs = _load_sequences_in_order(file_path, options)

    valid_idx   = [i for i, s in enumerate(log_seqs) if s is not None]
    valid_log   = [log_seqs[i]  for i in valid_idx]
    valid_time  = [time_seqs[i] for i in valid_idx]

    if not valid_idx:
        return [None] * len(log_seqs)

    log_arr  = np.array(valid_log,  dtype=object)
    time_arr = np.array(valid_time, dtype=object)

    dataset = LogDataset(
        log_arr,
        time_arr,
        vocab,
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

                mask_index  = data["bert_label"][i] > 0
                num_masked  = int(torch.sum(mask_index).item())
                seq_res["masked_tokens"] = num_masked

                if options["is_logkey"] and num_masked > 0:
                    masked_out = mask_lm_output[i][mask_index]     # [M, V]
                    masked_lbl = data["bert_label"][i][mask_index]  # [M]
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

    The logkey criterion: more than `seq_threshold` fraction of the masked
    tokens were not among the model's top-`num_candidates` predictions.
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
# Step 5 – Ground-truth labels
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(dataset_dir: Path) -> pd.DataFrame:
    """
    Combine phase1 (normal) and phase2 (anomalous) label CSVs.
    Adds a binary `true_label` column: 0 = normal, 1 = anomalous.
    """
    phase1 = pd.read_csv(dataset_dir / "labels" / "phase1_labels.csv")
    phase2 = pd.read_csv(dataset_dir / "labels" / "phase2_labels.csv")
    phase1["true_label"] = 0
    phase2["true_label"] = 1
    return pd.concat([phase1, phase2], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(options: dict, seq_threshold: float = 0.5) -> dict:
    """
    Load the trained model, run prediction on both test files, aggregate
    per-sequence scores to per-run labels, and compute classification metrics.

    A run is classified as anomalous if ANY of its constituent log-file
    sequences is detected as anomalous (conservative OR aggregation).
    """
    device = torch.device(options["device"])

    print("\n" + "=" * 60)
    print("STEP 4 – Running LogBERT predictions")
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

    vocab = WordVocab.load_vocab(options["vocab_path"])

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Predict on both test files.
    normal_results   = predict_file(
        model, vocab, options,
        str(OUTPUT_DIR / "test_normal"),   device
    )
    abnormal_results = predict_file(
        model, vocab, options,
        str(OUTPUT_DIR / "test_abnormal"), device
    )

    print(f"\n  Normal   sequences scored : {sum(r is not None for r in normal_results)}")
    print(f"  Abnormal sequences scored : {sum(r is not None for r in abnormal_results)}")

    # Aggregate per-sequence → per-run predictions.
    # A run is anomalous if at least one of its sequences is anomalous.
    def aggregate_to_runs(seq_results: list, manifest_entries: list) -> dict:
        run_preds: dict = {}
        for res, info in zip(seq_results, manifest_entries):
            rid = info["run_id"]
            anom = is_anomalous(res, options, seq_threshold)
            run_preds[rid] = run_preds.get(rid, False) or anom
        return {k: int(v) for k, v in run_preds.items()}

    normal_preds   = aggregate_to_runs(normal_results,   manifest["test_normal"])
    abnormal_preds = aggregate_to_runs(abnormal_results, manifest["test_abnormal"])

    print("\n" + "=" * 60)
    print("STEP 5 – Comparing against ground-truth labels")
    print("=" * 60)

    labels_df = load_labels(DATASET_DIR)
    label_map = dict(zip(labels_df["run_id"], labels_df["true_label"]))
    meta_map  = labels_df.set_index("run_id").to_dict("index")

    all_preds = {**normal_preds, **abnormal_preds}

    rows = []
    for run_id in sorted(all_preds):
        pred       = all_preds[run_id]
        true       = label_map.get(run_id, -1)
        fault_type = meta_map.get(run_id, {}).get("fault_type", "unknown")
        scenario   = meta_map.get(run_id, {}).get("scenario",   "unknown")
        rows.append(
            {
                "run_id":     run_id,
                "scenario":   scenario,
                "fault_type": fault_type,
                "true_label": true,
                "pred_label": pred,
                "correct":    int(pred == true),
            }
        )
    results_df = pd.DataFrame(rows)

    print("\nPer-run results:")
    print(results_df.to_string(index=False))

    # Compute binary classification metrics.
    TP = int(((results_df["true_label"] == 1) & (results_df["pred_label"] == 1)).sum())
    TN = int(((results_df["true_label"] == 0) & (results_df["pred_label"] == 0)).sum())
    FP = int(((results_df["true_label"] == 0) & (results_df["pred_label"] == 1)).sum())
    FN = int(((results_df["true_label"] == 1) & (results_df["pred_label"] == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy  = (TP + TN) / len(results_df) if results_df.shape[0] > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Classification metrics  (seq_threshold={seq_threshold:.2f})")
    print("=" * 60)
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 score  : {f1:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")

    # ── Save results ────────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    metrics = {
        "seq_threshold": seq_threshold,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Per-run results -> {results_path}")
    print(f"  Metrics         -> {metrics_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end LogBERT evaluation on the NCCL log dataset"
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip log preprocessing (sequence files must already exist)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (best_bert.pth must already exist in model_dir)"
    )
    parser.add_argument(
        "--seq-threshold", type=float, default=0.5,
        help=(
            "Anomaly threshold: a sequence is anomalous when the fraction of "
            "masked tokens that the model fails to predict exceeds this value "
            "(default: 0.5)"
        ),
    )
    args = parser.parse_args()

    if not args.skip_preprocess:
        run_preprocessing()

    if not args.skip_train:
        build_vocab(OPTIONS)
        train_logbert(OPTIONS)

    evaluate(OPTIONS, seq_threshold=args.seq_threshold)


if __name__ == "__main__":
    main()
