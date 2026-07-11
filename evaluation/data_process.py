#!/usr/bin/env python3
"""
NCCL log preprocessing for LogBERT evaluation.

Reads NCCL log files, extracts event templates using the Drain log parser, and
writes LogBERT-compatible sequence files.

Usage:
    python data_process.py [--train-ratio 0.8]

Output directory: <workspace>/output/nccl/
    train_normal              - training portion of phase1 sequences (configurable split)
    train                     - alias for train_normal (backward compatibility)
    test_normal               - test portion of phase1 sequences
    train_abnormal            - training portion of all phase2 sequences
    test_abnormal             - test portion of all phase2 sequences
    train_abnormal_<type>     - training portion of phase2 sequences per fault type
    test_abnormal_<type>      - test portion of phase2 sequences per fault type
    vocab.pkl        - built by logbert_nccl.py vocab
    event_map.json   - EventId (Drain) -> integer mapping
    manifest.json    - line-index -> {run_id, log_file} for all split files
    drain_input/     - preprocessed log lines fed to Drain
    drain_output/    - Drain's structured CSV output
"""

import argparse
import sys
import os
import re
import json
import pandas as pd
from pathlib import Path

# Make logbert (and its logparser sub-package) importable.
EVAL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EVAL_DIR / "logbert"))

from logparser import Drain  # noqa: E402 (after sys.path setup)

# ── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR = (EVAL_DIR / "../dataset").resolve()
OUTPUT_DIR = (EVAL_DIR / "../output/nccl").resolve()

# ── NCCL log-line preprocessing ──────────────────────────────────────────────

# Matches the variable "hostname:pid:tid [rank]" or "hostname:pid:tid" prefix
# that appears before "NCCL" on every log line.
_HEADER_RE = re.compile(r"^[^\s:]+:\d+:\d+\s+(?:\[\d+\]\s+)?")


# def strip_header(line: str) -> str:
#     """Return just the 'NCCL LEVEL content' part of a raw NCCL log line."""
#     line = line.strip()
#     m = _HEADER_RE.match(line)
#     return line[m.end():] if m else line


# def is_valid_nccl_line(content: str) -> bool:
#     """True when content has the expected 'NCCL <Level> <text>' shape."""
#     parts = content.split(None, 2)
#     return len(parts) >= 3 and parts[0] == "NCCL"


# ── File collection ───────────────────────────────────────────────────────────

def collect_log_files(dataset_dir: Path) -> list:
    """
    Return [(phase, run_id, Path)] for every NCCL log file in the dataset.
    phase is 'phase1' (normal) or 'phase2' (anomalous).
    """
    entries = []
    for subdir_name, phase in [("phase1_runs", "phase1"), ("phase2_runs", "phase2")]:
        phase_dir = dataset_dir / subdir_name
        if not phase_dir.exists():
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            if run_dir.is_dir():
                for log_file in sorted(run_dir.glob("nccl_logs_*.txt")):
                    entries.append((phase, run_dir.name, log_file))
    return entries


# ── Preprocessing → single file for Drain ────────────────────────────────────

def preprocess_logs(files: list, all_logs_path: Path) -> list:
    """
    Strip log-line headers and write the cleaned content to *all_logs_path*
    (one line per valid NCCL log entry).

    Returns a manifest list: one dict per source file:
        {start, end, phase, run_id, log_file}
    where start/end are the half-open row range in all_logs_path.
    """
    manifest = []
    line_num = 0
    with open(all_logs_path, "w", encoding="utf-8") as out_f:
        for phase, run_id, log_file_path in files:
            start = line_num
            with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    raw = raw_line.strip()
                    if raw:
                        out_f.write(raw + "\n")
                        line_num += 1
            manifest.append(
                {
                    "start": start,
                    "end": line_num,
                    "phase": phase,
                    "run_id": run_id,
                    "log_file": log_file_path.name,
                }
            )
    return manifest


# ── Drain parser ──────────────────────────────────────────────────────────────

def run_drain(all_logs_path: Path, drain_out_dir: Path) -> None:
    """Run the Drain log-template miner on the preprocessed log file."""
    # log_format = "<Host> NCCL <Level> <Content>"
    log_format = "<Content>"  # Drain will treat the whole line as content

    # Regexes for variable parts that Drain should treat as wildcards.
    # rex = [
    #     r"0x[0-9a-fA-F]+",       # hex memory addresses / comm IDs
    #     r"\d+\.\d+\.\d+\.\d+",   # IP addresses
    #     r"nccl-[A-Za-z0-9]+",    # NCCL shared-memory segment names
    #     r"/dev/shm/\S+",         # shared-memory paths
    #     r"\b\d{4,}\b",           # long numeric IDs (pids, byte counts, …)
    # ]
    rex = []

    parser = Drain.LogParser(
        log_format,
        indir=str(all_logs_path.parent) + "/",
        outdir=str(drain_out_dir) + "/",
        depth=4, # standard Drain config
        st=0.4,
        rex=rex,
        keep_para=False,
    )
    parser.parse(all_logs_path.name)


# ── Event-ID mapping ──────────────────────────────────────────────────────────

def build_event_map(templates_csv: Path) -> dict:
    """
    Map Drain EventId strings → positive integers.
    Templates are sorted by descending occurrence frequency so the most
    common events get the lowest (most stable) integer IDs.
    """
    df = pd.read_csv(templates_csv)
    df.sort_values(by="Occurrences", ascending=False, inplace=True)
    return {eid: idx + 1 for idx, eid in enumerate(df["EventId"])}


# ── Sequence construction ─────────────────────────────────────────────────────

def build_sequences(structured_csv: Path, manifest: list, event_map: dict) -> list:
    """
    Convert the Drain structured CSV into per-log-file event sequences.

    Returns a list of dicts:
        {phase, run_id, log_file, events: [int]}
    Events unseen in the training templates are mapped to 1 (UNK token index).
    """
    df = pd.read_csv(structured_csv)
    # Map EventId → integer; unknown → 1 (vocab UNK index)
    df["EventInt"] = df["EventId"].map(event_map).fillna(1).astype(int)

    sequences = []
    for entry in manifest:
        rows = df.iloc[entry["start"] : entry["end"]]
        events = rows["EventInt"].tolist()
        if not events:
            continue
        sequences.append(
            {
                "phase": entry["phase"],
                "run_id": entry["run_id"],
                "log_file": entry["log_file"],
                "events": events,
            }
        )
    return sequences


def load_phase2_fault_types(dataset_dir: Path) -> dict:
    """Return {run_id: fault_type} loaded from labels/phase2_labels.csv."""
    labels_path = dataset_dir / "labels" / "phase2_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing phase2 labels file: {labels_path}")

    df = pd.read_csv(labels_path, usecols=["run_id", "fault_type"])
    df = df.dropna(subset=["run_id", "fault_type"]).copy()
    df["run_id"] = df["run_id"].astype(str)
    df["fault_type"] = df["fault_type"].astype(str)
    return dict(zip(df["run_id"], df["fault_type"]))


# ── Write LogBERT input files ─────────────────────────────────────────────────

def _split(items: list, train_ratio: float) -> tuple:
    """Split *items* into (train, test) using the given ratio (deterministic)."""
    n_train = max(1, int(len(items) * train_ratio)) if items else 0
    return items[:n_train], items[n_train:]


def write_output_files(
    sequences: list,
    output_dir: Path,
    phase2_fault_type_by_run: dict,
    normal_train_ratio: float = 0.8,
    abnormal_train_ratio: float = 0.8,
) -> None:
    """
    Write:
      train_normal              - training portion of phase1 sequences
      train                     - alias for train_normal (backward compatibility)
      test_normal               - test portion of phase1 sequences
      train_abnormal            - training portion of all phase2 sequences
      test_abnormal             - test portion of all phase2 sequences
      train_abnormal_<type>     - training portion of phase2 per fault type
      test_abnormal_<type>      - test portion of phase2 per fault type
      manifest.json             - maps split name -> list of {run_id, log_file} entries
    """
    phase1_sequences = [s for s in sequences if s["phase"] == "phase1"]
    phase2_sequences = [s for s in sequences if s["phase"] == "phase2"]

    # ── Normal (phase1) split ────────────────────────────────────────────────
    train_normal_seqs, test_normal_seqs = _split(phase1_sequences, normal_train_ratio)

    # ── Abnormal (phase2) split per fault type ───────────────────────────────
    seqs_by_type: dict = {}
    for seq in phase2_sequences:
        fault_type = phase2_fault_type_by_run.get(seq["run_id"], "unknown")
        seqs_by_type.setdefault(fault_type, []).append(seq)

    train_abnormal_by_type: dict = {}
    test_abnormal_by_type: dict = {}
    for fault_type, seqs in seqs_by_type.items():
        tr, te = _split(seqs, abnormal_train_ratio)
        train_abnormal_by_type[fault_type] = tr
        test_abnormal_by_type[fault_type] = te

    # Flatten combined abnormal lists
    train_abnormal_seqs = [s for seqs in train_abnormal_by_type.values() for s in seqs]
    test_abnormal_seqs  = [s for seqs in test_abnormal_by_type.values()  for s in seqs]

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _to_lines(seqs):
        return [" ".join(map(str, s["events"])) for s in seqs]

    def _to_manifest(seqs, include_fault_type=False):
        result = []
        for s in seqs:
            entry = {"run_id": s["run_id"], "log_file": s["log_file"]}
            if include_fault_type:
                entry["fault_type"] = phase2_fault_type_by_run.get(s["run_id"], "unknown")
            result.append(entry)
        return result

    def _write(path: Path, lines: list) -> None:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── Write files ──────────────────────────────────────────────────────────
    train_normal_lines = _to_lines(train_normal_seqs)
    _write(output_dir / "train_normal", train_normal_lines)
    _write(output_dir / "train", train_normal_lines)          # backward compat

    _write(output_dir / "test_normal",   _to_lines(test_normal_seqs))
    _write(output_dir / "train_abnormal", _to_lines(train_abnormal_seqs))
    _write(output_dir / "test_abnormal",  _to_lines(test_abnormal_seqs))

    for fault_type in sorted(seqs_by_type):
        _write(output_dir / f"train_abnormal_{fault_type}", _to_lines(train_abnormal_by_type[fault_type]))
        _write(output_dir / f"test_abnormal_{fault_type}",  _to_lines(test_abnormal_by_type[fault_type]))

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "train_normal": _to_manifest(train_normal_seqs),
        "test_normal":  _to_manifest(test_normal_seqs),
        "train_abnormal": _to_manifest(train_abnormal_seqs, include_fault_type=True),
        "test_abnormal":  _to_manifest(test_abnormal_seqs,  include_fault_type=True),
    }
    for fault_type in sorted(seqs_by_type):
        manifest[f"train_abnormal_{fault_type}"] = _to_manifest(train_abnormal_by_type[fault_type])
        manifest[f"test_abnormal_{fault_type}"]  = _to_manifest(test_abnormal_by_type[fault_type])

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Train-normal seqs    : {len(train_normal_seqs)}")
    print(f"  Test-normal seqs     : {len(test_normal_seqs)}")
    print(f"  Train-abnormal seqs  : {len(train_abnormal_seqs)}")
    print(f"  Test-abnormal seqs   : {len(test_abnormal_seqs)}")
    for fault_type in sorted(seqs_by_type):
        n_tr = len(train_abnormal_by_type[fault_type])
        n_te = len(test_abnormal_by_type[fault_type])
        print(f"  Abnormal ({fault_type:20s}) : {n_tr} train / {n_te} test")
    print(f"  Output directory     : {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(normal_train_ratio: float = 0.8, abnormal_train_ratio: float = 0.8) -> None:
    drain_input_dir = OUTPUT_DIR / "drain_input"
    drain_output_dir = OUTPUT_DIR / "drain_output"
    drain_input_dir.mkdir(parents=True, exist_ok=True)
    drain_output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_logs_path = drain_input_dir / "nccl_all.log"

    print("=" * 60)
    print("Step 1/6  Collecting log files from dataset")
    print("=" * 60)
    files = collect_log_files(DATASET_DIR)
    print(f"  Found {len(files)} log files across {len(set(r for _, r, _ in files))} runs")

    print("\n" + "=" * 60)
    print("Step 2/6  Stripping headers and writing unified log file")
    print("=" * 60)
    manifest = preprocess_logs(files, all_logs_path)
    total_lines = manifest[-1]["end"] if manifest else 0
    print(f"  Total preprocessed NCCL lines : {total_lines}")

    print("\n" + "=" * 60)
    print("Step 3/6  Running Drain log parser")
    print("=" * 60)
    run_drain(all_logs_path, drain_output_dir)

    structured_csv = drain_output_dir / "nccl_all.log_structured.csv"
    templates_csv = drain_output_dir / "nccl_all.log_templates.csv"

    print("\n" + "=" * 60)
    print("Step 4/6  Building event-ID mapping")
    print("=" * 60)
    event_map = build_event_map(templates_csv)
    with open(OUTPUT_DIR / "event_map.json", "w") as f:
        json.dump(event_map, f, indent=2)
    print(f"  Unique event templates : {len(event_map)}")

    print("\n" + "=" * 60)
    print("Step 5/6  Building per-file event sequences")
    print("=" * 60)
    sequences = build_sequences(structured_csv, manifest, event_map)
    phase2_fault_type_by_run = load_phase2_fault_types(DATASET_DIR)
    print(f"  Total sequences : {len(sequences)}")

    print("\n" + "=" * 60)
    print("Step 6/6  Writing LogBERT input files")
    print("=" * 60)
    print(f"  Normal  split ratio : {normal_train_ratio:.0%} train / {1 - normal_train_ratio:.0%} test")
    print(f"  Abnormal split ratio: {abnormal_train_ratio:.0%} train / {1 - abnormal_train_ratio:.0%} test")
    write_output_files(
        sequences, OUTPUT_DIR, phase2_fault_type_by_run,
        normal_train_ratio=normal_train_ratio,
        abnormal_train_ratio=abnormal_train_ratio,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NCCL logs for anomaly detection models.")
    parser.add_argument(
        "--normal-train-ratio",
        type=float,
        default=0.8,
        metavar="RATIO",
        help="Fraction of normal (phase1) sequences used for training (default: 0.8).",
    )
    parser.add_argument(
        "--abnormal-train-ratio",
        type=float,
        default=0.8,
        metavar="RATIO",
        help="Fraction of abnormal (phase2) sequences used for training (default: 0.8).",
    )
    _args = parser.parse_args()
    for _name, _val in [("--normal-train-ratio", _args.normal_train_ratio),
                        ("--abnormal-train-ratio", _args.abnormal_train_ratio)]:
        if not (0.0 < _val < 1.0):
            parser.error(f"{_name} must be between 0 and 1 (exclusive).")
    main(normal_train_ratio=_args.normal_train_ratio, abnormal_train_ratio=_args.abnormal_train_ratio)
