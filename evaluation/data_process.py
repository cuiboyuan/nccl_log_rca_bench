#!/usr/bin/env python3
"""
NCCL log preprocessing for LogBERT evaluation.

Reads NCCL log files from /workspace/dataset_nccl_log, extracts event
templates using the Drain log parser, and writes LogBERT-compatible
sequence files (train / test_normal / test_abnormal).

Output directory: <workspace>/output/nccl/
  train            - space-separated event-ID sequences, one per line (phase1)
  test_normal      - same sequences as train (for evaluation)
  test_abnormal    - phase2 sequences (OOM + slow runs, all anomalous)
  vocab.pkl        - built by logbert_nccl.py vocab
  event_map.json   - EventId (Drain) → integer mapping
  manifest.json    - line-index → {run_id, log_file} for test files
  drain_input/     - preprocessed log lines fed to Drain
  drain_output/    - Drain's structured CSV output
"""

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

DATASET_DIR = Path("/workspace/dataset_nccl_log")
OUTPUT_DIR = (EVAL_DIR / "../output/nccl").resolve()

# ── NCCL log-line preprocessing ──────────────────────────────────────────────

# Matches the variable "hostname:pid:tid [rank]" or "hostname:pid:tid" prefix
# that appears before "NCCL" on every log line.
_HEADER_RE = re.compile(r"^[^\s:]+:\d+:\d+\s+(?:\[\d+\]\s+)?")


def strip_header(line: str) -> str:
    """Return just the 'NCCL LEVEL content' part of a raw NCCL log line."""
    line = line.strip()
    m = _HEADER_RE.match(line)
    return line[m.end():] if m else line


def is_valid_nccl_line(content: str) -> bool:
    """True when content has the expected 'NCCL <Level> <text>' shape."""
    parts = content.split(None, 2)
    return len(parts) >= 3 and parts[0] == "NCCL"


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
                    content = strip_header(raw_line)
                    if is_valid_nccl_line(content):
                        out_f.write(content + "\n")
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
    log_format = "NCCL <Level> <Content>"

    # Regexes for variable parts that Drain should treat as wildcards.
    rex = [
        r"0x[0-9a-fA-F]+",       # hex memory addresses / comm IDs
        r"\d+\.\d+\.\d+\.\d+",   # IP addresses
        r"nccl-[A-Za-z0-9]+",    # NCCL shared-memory segment names
        r"/dev/shm/\S+",         # shared-memory paths
        r"\b\d{4,}\b",           # long numeric IDs (pids, byte counts, …)
    ]

    parser = Drain.LogParser(
        log_format,
        indir=str(all_logs_path.parent) + "/",
        outdir=str(drain_out_dir) + "/",
        depth=4,
        st=0.5,
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


# ── Write LogBERT input files ─────────────────────────────────────────────────

def write_output_files(sequences: list, output_dir: Path) -> None:
    """
    Write:
      train            – one line per phase1 log file (space-separated event IDs)
      test_normal      – same as train (phase1 sequences used for normal test)
      test_abnormal    – phase2 log file sequences
      manifest.json    – maps line index → {run_id, log_file} for test files
    """
    train_lines, test_normal_lines, test_abnormal_lines = [], [], []
    test_normal_manifest, test_abnormal_manifest = [], []

    for seq in sequences:
        line = " ".join(map(str, seq["events"]))
        info = {"run_id": seq["run_id"], "log_file": seq["log_file"]}
        if seq["phase"] == "phase1":
            train_lines.append(line)
            test_normal_lines.append(line)
            test_normal_manifest.append(info)
        else:
            test_abnormal_lines.append(line)
            test_abnormal_manifest.append(info)

    def _write(path: Path, lines: list) -> None:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    _write(output_dir / "train", train_lines)
    _write(output_dir / "test_normal", test_normal_lines)
    _write(output_dir / "test_abnormal", test_abnormal_lines)

    manifest = {
        "test_normal": test_normal_manifest,
        "test_abnormal": test_abnormal_manifest,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Training sequences   : {len(train_lines)}")
    print(f"  Test-normal seqs     : {len(test_normal_lines)}")
    print(f"  Test-abnormal seqs   : {len(test_abnormal_lines)}")
    print(f"  Output directory     : {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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
    print(f"  Total sequences : {len(sequences)}")

    print("\n" + "=" * 60)
    print("Step 6/6  Writing LogBERT input files")
    print("=" * 60)
    write_output_files(sequences, OUTPUT_DIR)


if __name__ == "__main__":
    main()
