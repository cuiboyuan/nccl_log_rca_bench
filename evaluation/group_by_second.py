#!/usr/bin/env python3
"""
Merge per-rank NCCL log lines within a run into per-second windows.

For each run directory that contains nccl_timestamps_*.json files (produced
during workload collection), reads log lines and their Unix timestamps, groups
them into 1-second buckets (floor of timestamp), orders within each bucket by
rank index, and writes timeline.json to the run directory.

Usage:
    python group_by_second.py [--dataset-dir PATH]
"""

import argparse
import json
import re
from pathlib import Path

DATASET_DIR = (Path(__file__).parent / "../dataset").resolve()

_RANK_RE = re.compile(r"\[(\d+)\]")


def detect_rank(log_file: Path) -> int:
    """Infer the rank for this log file by scanning lines for the '[N]' bracket."""
    with open(log_file, "r", errors="replace") as f:
        for line in f:
            m = _RANK_RE.search(line)
            if m:
                return int(m.group(1))
    raise ValueError(f"Could not detect rank from {log_file}")


def load_run(run_dir: Path) -> dict:
    """
    For every nccl_timestamps_*.json file in run_dir that has a matching
    nccl_logs_*.txt file, detect the rank and load lines + timestamps.

    Returns:
        {rank (int): {"lines": list[str], "timestamps": list[float], "filename": str}}
    """
    result = {}
    for ts_file in sorted(run_dir.glob("nccl_timestamps_*.json")):
        log_name = (
            ts_file.name
            .replace("nccl_timestamps_", "nccl_logs_")
            .replace(".json", ".txt")
        )
        log_file = run_dir / log_name
        if not log_file.exists():
            continue

        rank = detect_rank(log_file)

        with open(log_file, "r", errors="replace") as f:
            lines = f.read().splitlines()

        with open(ts_file) as f:
            ts_data = json.load(f)
        timestamps = ts_data["line_timestamps"]

        # Guard against length mismatch (lines written after the final poll)
        n = min(len(lines), len(timestamps))
        result[rank] = {
            "lines": lines[:n],
            "timestamps": timestamps[:n],
            "filename": log_file.name,
        }
    return result


def build_timeline(run_data: dict) -> list:
    """
    Flatten all rank entries into (bucket, rank, line_idx, line) tuples,
    sort by (bucket, rank, line_idx), and group into per-second windows.

    Returns:
        list of {"second": int, "lines": list[str], "index": list[dict]}
    """
    flat = []
    for rank, data in run_data.items():
        for idx, (line, ts) in enumerate(zip(data["lines"], data["timestamps"])):
            flat.append((int(ts), rank, idx, line))
    flat.sort(key=lambda x: (x[0], x[1], x[2]))

    windows = []
    current_bucket = None
    current_window = None
    for bucket, rank, line_idx, line in flat:
        if bucket != current_bucket:
            if current_window is not None:
                windows.append(current_window)
            current_bucket = bucket
            current_window = {"second": bucket, "lines": [], "index": []}
        current_window["lines"].append(line)
        current_window["index"].append({"rank": rank, "line_idx": line_idx})
    if current_window is not None:
        windows.append(current_window)

    return windows


def process_run(run_dir: Path):
    run_data = load_run(run_dir)
    if not run_data:
        return

    rank_files = {str(rank): data["filename"] for rank, data in sorted(run_data.items())}
    windows = build_timeline(run_data)
    total_lines = sum(len(w["lines"]) for w in windows)

    timeline = {
        "run_id": run_dir.name,
        "rank_files": rank_files,
        "windows": windows,
    }

    out_path = run_dir / "timeline.json"
    with open(out_path, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"{run_dir.name}: {len(windows)} windows, {total_lines} lines -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help=f"Root dataset directory (default: {DATASET_DIR})"
    )
    args = parser.parse_args()

    for phase_subdir in ("phase1_runs", "phase2_runs"):
        phase_dir = args.dataset_dir / phase_subdir
        if not phase_dir.exists():
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            if run_dir.is_dir():
                process_run(run_dir)


if __name__ == "__main__":
    main()
