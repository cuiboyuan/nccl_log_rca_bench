#!/usr/bin/env python3
"""
Calculate per-run and total LLM API cost from a logprompt_evaluation_results.csv.

Token cost breakdown
--------------------
  non_cached_input  = input_tokens - cached_input_tokens
  cost = (non_cached_input        * price_input
        + cached_input_tokens     * price_cached_input
        + output_tokens           * price_output) / 1_000_000

Prices are in USD per 1M tokens.  gpt-5.4 has two tiers based on total
input_tokens per request:
  Short context (<= 272 K): input $2.50, cached $0.25, output $15.00
  Long  context (>  272 K): input $5.00, cached $0.50, output $22.50
Cache writes are not charged (n/a).

Usage
-----
    # Use built-in prices for a known model:
    python calculate_cost.py results.csv

    # Override individual prices (USD per 1M tokens):
    python calculate_cost.py results.csv \\
        --price-input 2.50 --price-cached-input 0.25 --price-output 15.00

    # Point at a specific output directory:
    python calculate_cost.py output/nccl/logprompt/CoT_raw_gpt-5.4_default_cot_rules/logprompt_evaluation_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Built-in price table for gpt-5.4 (USD per 1M tokens) ───────────────────
# Source: https://developers.openai.com/api/docs/pricing
# Two tiers based on total input_tokens per request.
# Cache writes are not charged (n/a).

SHORT_CONTEXT_THRESHOLD = 272_000  # tokens

_GPT54_PRICES: dict[str, dict[str, float]] = {
    "short": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "long":  {"input": 5.00, "cached_input": 0.50, "output": 22.50},
}


def _lookup_prices(model: str, input_tokens: int) -> dict[str, float] | None:
    """Return prices for model given total input_tokens, or None if unknown."""
    if model.lower().startswith("gpt-5.4"):
        tier = "short" if input_tokens <= SHORT_CONTEXT_THRESHOLD else "long"
        return _GPT54_PRICES[tier]
    return None


def compute_cost(row: pd.Series, prices: dict[str, float]) -> float:
    non_cached_input = max(0, row["input_tokens"] - row["cached_input_tokens"])
    return (
        non_cached_input             * prices["input"]
        + row["cached_input_tokens"] * prices["cached_input"]
        + row["output_tokens"]       * prices["output"]
    ) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate per-run LLM cost from logprompt_evaluation_results.csv."
    )
    parser.add_argument("csv_file", help="Path to logprompt_evaluation_results.csv")
    parser.add_argument(
        "--price-input", type=float, default=None,
        metavar="USD_PER_1M",
        help="Override: price for non-cached input tokens (USD/1M).",
    )
    parser.add_argument(
        "--price-cached-input", type=float, default=None,
        metavar="USD_PER_1M",
        help="Override: price for cached input tokens (USD/1M).",
    )
    parser.add_argument(
        "--price-output", type=float, default=None,
        metavar="USD_PER_1M",
        help="Override: price for output tokens (USD/1M).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Optional path to write cost-annotated CSV (default: print to stdout).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        sys.exit(f"Error: file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"input_tokens", "cached_input_tokens", "output_tokens", "model"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing required columns: {missing}")

    # Resolve prices per row (accounts for context-length tier)
    def _prices_for_row(row: pd.Series) -> dict[str, float]:
        prices = _lookup_prices(row["model"], int(row["input_tokens"]))
        if prices is None:
            print(
                f"Warning: unknown model '{row['model']}'; using $0 for missing prices. "
                "Use --price-* flags to set prices explicitly.",
                file=sys.stderr,
            )
            prices = {"input": 0.0, "cached_input": 0.0, "output": 0.0}
        # Apply CLI overrides
        overrides = {
            "input":        args.price_input,
            "cached_input": args.price_cached_input,
            "output":       args.price_output,
        }
        return {k: (v if overrides[k] is None else overrides[k]) for k, v in prices.items()}

    # Print tier summary
    short = _GPT54_PRICES["short"]
    long  = _GPT54_PRICES["long"]
    print(f"Prices for gpt-5.4 (USD per 1M tokens, threshold={SHORT_CONTEXT_THRESHOLD:,} input tokens):")
    print(f"  {'Tier':<8}  {'input':>6}  {'cached':>6}  {'output':>6}")
    print(f"  {'short':<8}  {short['input']:>6.2f}  {short['cached_input']:>6.2f}  {short['output']:>6.2f}")
    print(f"  {'long':<8}  {long['input']:>6.2f}  {long['cached_input']:>6.2f}  {long['output']:>6.2f}")
    print()

    df["cost_usd"]     = df.apply(lambda row: compute_cost(row, _prices_for_row(row)), axis=1)
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]

    # ── Per-run breakdown ─────────────────────────────────────────────────────
    display_cols = [
        "run_id", "model", "input_tokens", "cached_input_tokens",
        "output_tokens", "total_tokens", "cost_usd",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))
    print()

    # ── Totals ────────────────────────────────────────────────────────────────
    total_input  = df["input_tokens"].sum()
    total_cached = df["cached_input_tokens"].sum()
    total_output = df["output_tokens"].sum()
    total_tokens = df["total_tokens"].sum()
    total_cost   = df["cost_usd"].sum()

    n = len(df)
    print("Totals:")
    print(f"  Runs               : {n}")
    print(f"  input_tokens       : {total_input:,}")
    print(f"  cached_input_tokens: {total_cached:,}  ({100*total_cached/total_input:.1f}% of input)")
    print(f"  output_tokens      : {total_output:,}")
    print(f"  total_tokens       : {total_tokens:,}")
    print(f"  Total cost         : ${total_cost:.6f}")
    print()
    print("Averages per run:")
    print(f"  input_tokens       : {total_input/n:,.1f}")
    print(f"  cached_input_tokens: {total_cached/n:,.1f}  ({100*total_cached/total_input:.1f}% of input)")
    print(f"  output_tokens      : {total_output/n:,.1f}")
    print(f"  total_tokens       : {total_tokens/n:,.1f}")
    print(f"  cost_usd           : ${total_cost/n:.6f}")

    if args.output:
        out_path = Path(args.output)
        df.to_csv(out_path, index=False)
        print(f"\nCost-annotated CSV written to: {out_path}")


if __name__ == "__main__":
    main()
