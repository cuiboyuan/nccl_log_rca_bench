#!/usr/bin/env python3
"""
LogPrompt evaluation for NCCL log anomaly detection.

Uses Drain-extracted log event templates as input to an LLM, classifying each
run as normal or anomalous.  No training is required.

Based on:
    Wang, Y. et al. "LogPrompt: Prompt Engineering Towards Zero-Shot and
    Interpretable Log Analysis."  ICPC 2024 (long paper) / ICSE 2024 (poster).
    arXiv:2308.07610  https://arxiv.org/abs/2308.07610
    Original code: https://github.com/lunyiliu/LogPrompt

Strategies
----------
CoT       Chain-of-thought: ask the LLM to reason through the run's templates
          and give a final Verdict: normal / Verdict: abnormal.
InContext Few-shot: prepend a set of labelled example runs before asking about
          the target run.  Requires --example-file.

Input files (produced by data_process.py — run it first):
  output/nccl/manifest.json
  output/nccl/event_map.json
  output/nccl/test_normal
  output/nccl/test_abnormal  (or test_abnormal_<fault_type> variants)
  output/nccl/drain_output/nccl_all.log_templates.csv
  dataset/labels/phase{1,2}_labels.csv

Outputs (written to --output-dir, default output/nccl/logprompt/):
  logprompt_evaluation_results.csv  per-run: run_id / true_label / pred_label / …
  logprompt_metrics.json            overall + per-fault precision/recall/F1
  logprompt_responses.jsonl         raw LLM responses keyed by run_id

Usage
-----
    python evaluate_logprompt.py --api-key SK-... --strategy CoT
    python evaluate_logprompt.py --api-url http://localhost:8000/v1/chat/completions \\
                                  --model llama3 --strategy CoT
    python evaluate_logprompt.py --api-key SK-... --strategy InContext \\
                                  --example-file examples.csv
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Path setup ────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EVAL_DIR))

import data_process  # noqa: E402

DATASET_DIR  = data_process.DATASET_DIR
OUTPUT_DIR   = (EVAL_DIR / "../output/nccl").resolve()
DRAIN_OUTPUT = OUTPUT_DIR / "drain_output"

DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL   = "gpt-3.5-turbo"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_int_to_template() -> dict:
    """
    Build {integer_event_id: EventTemplate} by chaining:
      event_map.json  (EventId → int)  →  reversed  →  templates CSV  (EventId → template).
    """
    event_map_path = OUTPUT_DIR / "event_map.json"
    templates_csv  = DRAIN_OUTPUT / "nccl_all.log_templates.csv"

    for p in (event_map_path, templates_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}. Run data_process.py first.")

    with open(event_map_path) as f:
        event_map = json.load(f)                      # {EventId: int}
    int_to_eid = {v: k for k, v in event_map.items()}  # {int: EventId}

    df = pd.read_csv(templates_csv)
    eid_to_template = dict(zip(df["EventId"], df["EventTemplate"]))

    return {
        i: eid_to_template.get(eid, f"<unknown:{eid}>")
        for i, eid in int_to_eid.items()
    }


def load_manifest() -> dict:
    path = OUTPUT_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest.json: {path}. Run data_process.py first.")
    with open(path) as f:
        return json.load(f)


def load_labels() -> pd.DataFrame:
    p1 = pd.read_csv(DATASET_DIR / "labels" / "phase1_labels.csv")
    p2 = pd.read_csv(DATASET_DIR / "labels" / "phase2_labels.csv")
    p1["true_label"] = 0
    p2["true_label"] = 1
    return pd.concat([p1, p2], ignore_index=True)


def _abnormal_split_keys(manifest: dict) -> list:
    """Prefer per-fault splits; fall back to combined test_abnormal."""
    per_fault = sorted(
        k for k in manifest
        if k.startswith("test_abnormal_") and k != "test_abnormal"
    )
    return per_fault if per_fault else ["test_abnormal"]


def build_run_templates(manifest: dict, int_to_template: dict) -> dict:
    """
    Return {run_id: [template, ...]} — unique templates per run in first-seen
    order, collected across all sequence lines that belong to that run.
    Includes both test and train splits so that training runs can be used as
    InContext examples.
    """
    run_templates: dict[str, list] = {}
    run_seen:      dict[str, set]  = {}

    train_abnormal_keys = sorted(
        k for k in manifest
        if k.startswith("train_abnormal_") and k != "train_abnormal"
    )
    train_abnormal_keys = train_abnormal_keys if train_abnormal_keys else ["train_abnormal"]
    split_keys = (
        ["train_normal"] + train_abnormal_keys
        + ["test_normal"] + _abnormal_split_keys(manifest)
    )

    for split_key in split_keys:
        if split_key not in manifest:
            continue
        seq_file = OUTPUT_DIR / split_key
        if not seq_file.exists():
            raise FileNotFoundError(
                f"Missing sequence file: {seq_file}. Run data_process.py first."
            )
        with open(seq_file) as f:
            lines = f.readlines()

        for line, entry in zip(lines, manifest[split_key]):
            run_id = entry["run_id"]
            if run_id not in run_templates:
                run_templates[run_id] = []
                run_seen[run_id]      = set()
            for tok in line.split():
                try:
                    tmpl = int_to_template.get(int(tok), f"<unknown:{tok}>")
                except ValueError:
                    continue
                if tmpl not in run_seen[run_id]:
                    run_templates[run_id].append(tmpl)
                    run_seen[run_id].add(tmpl)

    return run_templates


def _find_log_file(dataset_dir: Path, run_id: str, log_file: str):
    """Locate a NCCL log file by searching phase1_runs and phase2_runs."""
    for phase_dir in ("phase1_runs", "phase2_runs"):
        path = dataset_dir / phase_dir / run_id / log_file
        if path.exists():
            return path
    return None


def build_run_raw_lines(manifest: dict, dataset_dir: Path) -> dict:
    """
    Return {run_id: [line, ...]} — unique raw log lines per run in first-seen
    order, collected across all log files that belong to that run.
    Includes both test and train splits so that training runs can be used as
    InContext examples.
    """
    run_lines: dict[str, list] = {}
    run_seen:  dict[str, set]  = {}

    train_abnormal_keys = sorted(
        k for k in manifest
        if k.startswith("train_abnormal_") and k != "train_abnormal"
    )
    train_abnormal_keys = train_abnormal_keys if train_abnormal_keys else ["train_abnormal"]
    split_keys = (
        ["train_normal"] + train_abnormal_keys
        + ["test_normal"] + _abnormal_split_keys(manifest)
    )

    for split_key in split_keys:
        if split_key not in manifest:
            continue
        for entry in manifest[split_key]:
            run_id   = entry["run_id"]
            log_file = entry["log_file"]
            log_path = _find_log_file(dataset_dir, run_id, log_file)
            if log_path is None:
                print(f"  Warning: log file not found: {run_id}/{log_file}")
                continue
            if run_id not in run_lines:
                run_lines[run_id] = []
                run_seen[run_id]  = set()
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line and line not in run_seen[run_id]:
                        run_lines[run_id].append(line)
                        run_seen[run_id].add(line)

    return run_lines


# ── Prompt construction ───────────────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are an expert in distributed deep learning systems and NCCL (NVIDIA Collective "
    "Communications Library) logs. You analyse log event templates from NCCL training "
    "runs and determine whether the run is normal or anomalous."
)

DEFAULT_COT_RULES = """\
(a) Mark it normal when values (such as memory address, floating number and register value) in a log are invalid.
(b) Mark it normal when lack of information.
(c) Never consider <*> and missing values as abnormal patterns.
(d) Mark it abnormal when and only when an alert is explicitly expressed in \
textual content (such as keywords like error, warning, fail, or abort)."""

_COT_USER = """\
Classify the following NCCL distributed training run as normal or abnormal based \
on its {log_desc}. Apply these rules:
{rules}

{log_desc_cap} observed in this run:
{templates}

Concisely explain your reasoning, then give your final answer on its own line \
in exactly this format:
Verdict: normal
or
Verdict: abnormal"""

_INCONTEXT_USER = """\
Classify the given log run into 0 and 1 categories based on semantic similarity \
to the following labelled example runs (0 = normal, 1 = abnormal):
{examples}

Now classify this run. Never consider <*> and missing values as abnormal patterns.
{log_desc_cap} observed in this run:
{templates}

Organize your answer on its own line in exactly this format:
Verdict: 0
or
Verdict: 1"""


def _fmt_templates(templates: list) -> str:
    return "\n".join(f"({i + 1}) {t}" for i, t in enumerate(templates))


def build_cot_prompt(
    templates: list,
    log_desc: str = "unique log event templates",
    rules: str = DEFAULT_COT_RULES,
) -> str:
    return _COT_USER.format(
        log_desc=log_desc,
        log_desc_cap=log_desc.capitalize(),
        rules=rules,
        templates=_fmt_templates(templates),
    )


def build_incontext_prompt(templates: list, examples: list, log_desc: str = "unique log event templates") -> str:
    """
    examples: list of dicts with keys {run_id, label, templates}.
    """
    blocks = []
    for ex in examples:
        category = 1 if ex["label"] == "abnormal" else 0
        blocks.append(
            f"Example (Category: {category}):\n{_fmt_templates(ex['templates'])}"
        )
    return _INCONTEXT_USER.format(
        examples="\n\n".join(blocks),
        templates=_fmt_templates(templates),
        log_desc_cap=log_desc.capitalize(),
    )


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str = "",
    max_retries: int = 8,
) -> str:
    """POST to an OpenAI-compatible chat completions endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            if attempt == max_retries - 1:
                raise
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                # Honour Retry-After header if present, otherwise exponential back-off
                retry_after = exc.response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt * 5, 120)
                print(f"    Rate-limited (429), retrying in {wait}s…")
            else:
                wait = 2 ** attempt
                print(f"    HTTP {status} error ({exc}), retrying in {wait}s…")
            time.sleep(wait)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    API error ({exc}), retrying in {wait}s…")
            time.sleep(wait)


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_verdict(response: str) -> int:
    """
    Return 1 (abnormal), 0 (normal), or -1 (unparseable).

    CoT responses use "Verdict: normal/abnormal".
    InContext responses use "Verdict: 0/1" (matching original LogPrompt).
    Falls back to whichever signal appears latest in the text.
    """
    for line in reversed(response.strip().splitlines()):
        # InContext: Verdict: 0 or Verdict: 1
        m = re.search(r"verdict\s*[:\-]\s*([01])", line.strip(), re.IGNORECASE)
        if m:
            return int(m.group(1))
        # CoT: Verdict: normal or Verdict: abnormal
        m = re.search(r"verdict\s*[:\-]\s*(normal|abnormal)", line.strip(), re.IGNORECASE)
        if m:
            return 1 if m.group(1).lower() == "abnormal" else 0

    lower = response.lower()
    last_abn = lower.rfind("abnormal")
    last_nor = lower.rfind("normal")
    if last_abn > last_nor >= 0:
        return 1
    if last_nor >= 0:
        return 0
    return -1


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_binary_metrics(df: pd.DataFrame) -> dict:
    tp = int(((df["true_label"] == 1) & (df["pred_label"] == 1)).sum())
    tn = int(((df["true_label"] == 0) & (df["pred_label"] == 0)).sum())
    fp = int(((df["true_label"] == 0) & (df["pred_label"] == 1)).sum())
    fn = int(((df["true_label"] == 1) & (df["pred_label"] == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(df) if len(df) > 0 else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def compute_per_fault_metrics(results_df: pd.DataFrame, normal_run_ids: set) -> dict:
    fault_types = sorted(
        ft for ft in results_df["fault_type"].dropna().astype(str).unique()
        if ft not in {"unknown", "nan", "no_fault"}
    )
    per_fault = {}
    for ft in fault_types:
        mask = (
            (results_df["fault_type"].astype(str) == ft)
            | results_df["run_id"].astype(str).isin(normal_run_ids)
        )
        m = compute_binary_metrics(results_df[mask])
        m["fault_type"] = ft
        per_fault[ft] = m
    return per_fault


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LogPrompt evaluation for NCCL log anomaly detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_logprompt.py --api-key SK-... --strategy CoT
  python evaluate_logprompt.py --api-url http://localhost:8000/v1/chat/completions \\
                                --model llama3 --strategy CoT
  python evaluate_logprompt.py --api-key SK-... --strategy InContext \\
                                --example-file examples.csv
        """,
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL,
        help=f"Chat completions endpoint (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name passed to the API (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--strategy", default="CoT", choices=["CoT", "InContext"],
        help="Prompt strategy: CoT (default) or InContext",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help=(
            "System message prepended to every API call. "
            "Disabled by default (matches original LogPrompt behaviour). "
            "Pass --system-prompt ... to enable; the built-in NCCL expert description "
            f"is available in the DEFAULT_SYSTEM constant."
        ),
    )
    parser.add_argument(
        "--example-file", default="",
        help=(
            "CSV with columns run_id and label (normal/abnormal). "
            "Required for InContext strategy."
        ),
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip existence check for preprocessed files.",
    )
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
    parser.add_argument(
        "--cot-rules", default=None, metavar="RULES",
        help=(
            "Custom rules string injected into the CoT prompt (replaces the built-in "
            "a/b/c/d rules). Newlines are honoured. Mutually exclusive with "
            "--cot-rules-file."
        ),
    )
    parser.add_argument(
        "--cot-rules-file", default=None, metavar="FILE",
        help=(
            "Path to a plain-text file whose contents replace the built-in CoT rules. "
            "Mutually exclusive with --cot-rules."
        ),
    )
    parser.add_argument(
        "--log-type", default="parsed", choices=["parsed", "raw"],
        help=(
            "Input representation passed to the LLM: 'parsed' (default) uses "
            "Drain-extracted event templates; 'raw' uses the original log lines directly."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory to write results. Defaults to "
            "output/nccl/logprompt/<strategy>_<log-type>_<model>/"
        ),
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set $OPENAI_API_KEY)")
    if args.strategy == "InContext" and not args.example_file:
        parser.error("--example-file is required for InContext strategy")
    if args.cot_rules and args.cot_rules_file:
        parser.error("--cot-rules and --cot-rules-file are mutually exclusive")

    cot_rules = DEFAULT_COT_RULES
    if args.cot_rules:
        cot_rules = args.cot_rules
    elif args.cot_rules_file:
        with open(args.cot_rules_file) as _f:
            cot_rules = _f.read().strip()

    if args.output_dir is None:
        safe_model = re.sub(r"[^\w.-]", "_", args.model)
        rules_tag = "custom_cot_rules" if (args.cot_rules or args.cot_rules_file) else "default_cot_rules"
        args.output_dir = str(
            OUTPUT_DIR / "logprompt" / f"{args.strategy}_{args.log_type}_{safe_model}_{rules_tag}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        required = [OUTPUT_DIR / "manifest.json"]
        if args.log_type == "parsed":
            required += [
                OUTPUT_DIR / "event_map.json",
                DRAIN_OUTPUT / "nccl_all.log_templates.csv",
            ]
        if any(not p.exists() for p in required):
            print("\n" + "=" * 60)
            print("Preprocessing – converting NCCL logs to sequences")
            print("=" * 60)
            data_process.main(normal_train_ratio=args.normal_train_ratio,
                              abnormal_train_ratio=args.abnormal_train_ratio)

    # ── Load data ─────────────────────────────────────────────────────────────

    print("=" * 60)
    print("LogPrompt – loading data")
    print("=" * 60)

    manifest        = load_manifest()
    labels_df       = load_labels()

    label_map = dict(zip(labels_df["run_id"].astype(str), labels_df["true_label"]))
    meta_map  = labels_df.set_index("run_id").to_dict("index")

    if args.log_type == "parsed":
        int_to_template = load_int_to_template()
        run_inputs = build_run_templates(manifest, int_to_template)
    else:
        int_to_template = {}
        run_inputs = build_run_raw_lines(manifest, DATASET_DIR)

    log_desc = (
        "unique log event templates" if args.log_type == "parsed" else "raw log lines"
    )
    normal_run_ids = {e["run_id"] for e in manifest.get("test_normal", [])}
    test_run_ids = normal_run_ids | {
        e["run_id"]
        for k in manifest
        if k.startswith("test_abnormal")
        for e in manifest[k]
    }

    print(f"  Runs to evaluate        : {len(test_run_ids)}")
    if args.log_type == "parsed":
        print(f"  Unique template vocab   : {len(int_to_template)}")

    # ── InContext: build example list ─────────────────────────────────────────

    incontext_examples = []
    if args.strategy == "InContext":
        ex_df = pd.read_csv(args.example_file)
        for _, row in ex_df.iterrows():
            rid = str(row["run_id"])
            lbl = str(row["label"]).strip().lower()
            if lbl not in ("normal", "abnormal"):
                print(f"  Warning: skipping example run {rid} with unknown label '{lbl}'")
                continue
            if rid not in run_inputs:
                print(f"  Warning: example run {rid} not found in processed data, skipping")
                continue
            incontext_examples.append({
                "run_id":    rid,
                "label":     lbl,
                "templates": run_inputs[rid],
            })
        print(f"  InContext examples      : {len(incontext_examples)}")
        if not incontext_examples:
            parser.error("No valid examples found in --example-file")

    # ── Classify each run ─────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print(f"LogPrompt – {args.strategy} via {args.model}")
    print("=" * 60)

    rows = []
    unparseable_count = 0

    for idx, (run_id, log_inputs) in enumerate(
        sorted((rid, inp) for rid, inp in run_inputs.items() if rid in test_run_ids), 1
    ):
        print(
            f"  [{idx:>3}/{len(test_run_ids)}] {run_id}  "
            f"({len(log_inputs)} {log_desc})",
            end=" … ", flush=True,
        )

        if args.strategy == "CoT":
            prompt = build_cot_prompt(log_inputs, log_desc, cot_rules)
        else:
            prompt = build_incontext_prompt(log_inputs, incontext_examples, log_desc)

        response = call_llm(prompt, args.api_url, args.api_key, args.model,
                            system_prompt=args.system_prompt)
        pred     = parse_verdict(response)

        if pred == -1:
            print("WARN: unparseable → defaulting to normal")
            print(f"    Response snippet: {response[:200]!r}")
            pred = 0
            unparseable_count += 1
        else:
            print("abnormal" if pred == 1 else "normal")

        rows.append({
            "run_id":     run_id,
            "scenario":   meta_map.get(run_id, {}).get("scenario",   "unknown"),
            "fault_type": meta_map.get(run_id, {}).get("fault_type", "unknown"),
            "true_label": int(label_map.get(run_id, -1)),
            "pred_label": pred,
            "correct":    int(pred == int(label_map.get(run_id, -1))),
            "log_type":   args.log_type,
            "strategy":   args.strategy,
            "model":      args.model,
            "prompt":     prompt,
            "response":   response,
        })

    # ── Evaluate ──────────────────────────────────────────────────────────────

    results_df = pd.DataFrame(rows)
    eval_df    = results_df[results_df["true_label"] != -1].copy()

    print("\n" + "=" * 60)
    print("LogPrompt – evaluation")
    print("=" * 60)
    print(
        results_df[["run_id", "fault_type", "true_label", "pred_label", "correct"]]
        .to_string(index=False)
    )

    if unparseable_count:
        print(f"\n  Warning: {unparseable_count} run(s) had unparseable LLM responses (defaulted to normal)")

    metrics = compute_binary_metrics(eval_df)
    metrics["model"]       = "logprompt"
    metrics["strategy"]    = args.strategy
    metrics["llm_model"]   = args.model
    metrics["log_type"]    = args.log_type
    metrics["by_fault_type"] = compute_per_fault_metrics(eval_df, normal_run_ids)

    print(f"\n  TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 score  : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")

    if metrics["by_fault_type"]:
        print("\n  Per-fault metrics:")
        for ft, fm in metrics["by_fault_type"].items():
            print(
                f"    {ft}: P={fm['precision']:.4f} R={fm['recall']:.4f} "
                f"F1={fm['f1']:.4f} Acc={fm['accuracy']:.4f} "
                f"(TP={fm['TP']} TN={fm['TN']} FP={fm['FP']} FN={fm['FN']})"
            )

    # ── Write outputs ─────────────────────────────────────────────────────────

    results_csv    = output_dir / "logprompt_evaluation_results.csv"
    metrics_json   = output_dir / "logprompt_metrics.json"
    responses_jsonl = output_dir / "logprompt_responses.jsonl"

    results_df.drop(columns=["response", "prompt"]).to_csv(results_csv, index=False)

    with open(responses_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps({"run_id": row["run_id"], "prompt": row["prompt"], "response": row["response"]}) + "\n")

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Per-run results  → {results_csv}")
    print(f"  Metrics          → {metrics_json}")
    print(f"  Raw LLM responses→ {responses_jsonl}")


if __name__ == "__main__":
    main()
