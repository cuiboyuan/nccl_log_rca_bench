#!/usr/bin/env python3
"""
AllReduce-count baseline for NCCL log anomaly detection.

For each run, counts the number of "NCCL INFO AllReduce: opCount" log lines
emitted by every rank (one log file per rank).  If all ranks produced the same
count the run is classified as normal; any imbalance flags it as anomalous.

Usage
-----
    python allreduce_count_baseline.py

Output
------
    output/nccl/allreduce_count_evaluation_results.csv
    output/nccl/allreduce_count_metrics.json
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# Shared paths are defined once in data_process.py; import them here to avoid
# duplication.

EVAL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EVAL_DIR))

from data_process import DATASET_DIR, OUTPUT_DIR  # noqa: E402

# ── Pattern ───────────────────────────────────────────────────────────────────

# Matches the canonical AllReduce call entry:
#   hostname:pid:tid [rank] NCCL INFO AllReduce: opCount N ...
_ALLREDUCE_RE = re.compile(r"\bNCCL INFO AllReduce: opCount\b")

# Extracts the rank tag "[N]" from a log-line header.
_RANK_RE = re.compile(r"\[(\d+)\]")


# ── Per-run analysis ──────────────────────────────────────────────────────────

def get_rank(log_file: Path) -> str:
    """Return the rank string for a log file (taken from the first tagged line)."""
    with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RANK_RE.search(line)
            if m:
                return m.group(1)
    # Fallback: use the filename itself as an identifier.
    return log_file.stem


def count_allreduce_calls(log_file: Path) -> int:
    """Count AllReduce INFO entries in a single log file."""
    count = 0
    with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if _ALLREDUCE_RE.search(line):
                count += 1
    return count


def classify_run(run_dir: Path) -> dict:
    """
    Analyse one run directory.

    Returns a dict with:
        rank_counts   – {rank_str: allreduce_count}
        pred_label    – 0 (normal) if all counts are equal, 1 (anomalous) otherwise
        min_count     – minimum AllReduce count across ranks
        max_count     – maximum AllReduce count across ranks
    """
    log_files = sorted(run_dir.glob("nccl_logs_*.txt"))

    rank_counts: dict[str, int] = {}
    for lf in log_files:
        rank = get_rank(lf)
        rank_counts[rank] = count_allreduce_calls(lf)

    if not rank_counts:
        # No log files found – treat as normal (cannot determine anomaly).
        return {"rank_counts": {}, "pred_label": 0, "min_count": 0, "max_count": 0}

    counts     = list(rank_counts.values())
    min_count  = min(counts)
    max_count  = max(counts)
    pred_label = 0 if min_count == max_count else 1

    return {
        "rank_counts": rank_counts,
        "pred_label":  pred_label,
        "min_count":   min_count,
        "max_count":   max_count,
    }


# ── Dataset traversal ─────────────────────────────────────────────────────────

def collect_run_dirs(dataset_dir: Path) -> list[tuple[str, Path]]:
    """
    Return [(phase, run_dir), ...] for every run directory in the dataset.
    phase is 'phase1' (normal) or 'phase2' (anomalous).
    """
    entries = []
    for subdir_name, phase in [("phase1_runs", "phase1"), ("phase2_runs", "phase2")]:
        phase_dir = dataset_dir / subdir_name
        if not phase_dir.exists():
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            if run_dir.is_dir():
                entries.append((phase, run_dir))
    return entries


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(dataset_dir: Path) -> pd.DataFrame:
    phase1 = pd.read_csv(dataset_dir / "labels" / "phase1_labels.csv")
    phase2 = pd.read_csv(dataset_dir / "labels" / "phase2_labels.csv")
    phase1["true_label"] = 0
    phase2["true_label"] = 1
    return pd.concat([phase1, phase2], ignore_index=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

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
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
    }


def compute_per_fault_metrics(df: pd.DataFrame) -> dict:
    """Compute binary metrics broken down by fault type (vs. all normal runs)."""
    normal_mask = df["true_label"] == 0
    fault_types = sorted(
        ft for ft in df["fault_type"].dropna().astype(str).unique()
        if ft not in {"no_fault", "nan"}
    )
    per_fault = {}
    for ft in fault_types:
        fault_mask = df["fault_type"].astype(str) == ft
        subset = df[normal_mask | fault_mask].copy()
        per_fault[ft] = compute_binary_metrics(subset)
    return per_fault


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("AllReduce-count baseline")
    print("=" * 60)

    run_dirs  = collect_run_dirs(DATASET_DIR)
    labels_df = load_labels(DATASET_DIR)

    label_map = dict(zip(labels_df["run_id"], labels_df["true_label"]))
    meta_map  = labels_df.set_index("run_id").to_dict("index")

    rows = []
    for phase, run_dir in run_dirs:
        run_id = run_dir.name
        result = classify_run(run_dir)

        true_label = label_map.get(run_id, -1)
        fault_type = meta_map.get(run_id, {}).get("fault_type", "unknown")
        scenario   = meta_map.get(run_id, {}).get("scenario",   "unknown")
        pred_label = result["pred_label"]

        print(
            f"  {run_id}: rank_counts={result['rank_counts']}  "
            f"pred={'ANOMALY' if pred_label else 'normal'}  "
            f"true={'ANOMALY' if true_label else 'normal'}"
        )

        rows.append({
            "run_id":      run_id,
            "phase":       phase,
            "scenario":    scenario,
            "fault_type":  fault_type,
            "rank_counts": json.dumps(result["rank_counts"]),
            "min_count":   result["min_count"],
            "max_count":   result["max_count"],
            "true_label":  true_label,
            "pred_label":  pred_label,
            "correct":     int(pred_label == true_label),
        })

    df = pd.DataFrame(rows)

    overall_metrics = compute_binary_metrics(df)
    overall_metrics["model"]         = "allreduce_count"
    overall_metrics["by_fault_type"] = compute_per_fault_metrics(df)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(df[["run_id", "fault_type", "min_count", "max_count", "true_label", "pred_label", "correct"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Overall metrics")
    print("=" * 60)
    for k, v in overall_metrics.items():
        if k != "by_fault_type":
            print(f"  {k}: {v}")

    print("\nPer-fault-type metrics:")
    for ft, m in overall_metrics["by_fault_type"].items():
        print(f"  {ft}: {m}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_csv  = OUTPUT_DIR / "allreduce_count_evaluation_results.csv"
    metrics_json = OUTPUT_DIR / "allreduce_count_metrics.json"

    df.to_csv(results_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"\nSaved results CSV:  {results_csv}")
    print(f"Saved metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()
