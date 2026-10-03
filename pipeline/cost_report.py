#!/usr/bin/env python3
"""Summarise time, tokens and estimated API cost per model for a run.

Reads every `run_metadata.json` under a results directory. Costs use the list
prices in `config.MODELS`; locally served models cost nothing.

    python -m pipeline.cost_report --results results/runs/<run id>
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_metadata(results):
    """Return one row per coded episode found under `results`.

    Args:
        results: Directory to search.

    Returns:
        DataFrame of per-episode metadata.
    """
    rows = []
    for path in Path(results).rglob("run_metadata.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "elapsed_seconds" in meta:
            rows.append(meta)
    return pd.DataFrame(rows)


def summarise(meta):
    """Return per-model means and totals.

    Args:
        meta: Output of `load_metadata`.

    Returns:
        DataFrame with one row per model.
    """
    return meta.groupby("model").agg(
        episodes=("elapsed_seconds", "size"),
        seconds_per_episode=("elapsed_seconds", "mean"),
        llm_calls_per_episode=("n_llm_calls", "mean"),
        tokens_per_episode=("total_tokens", "mean"),
        cost_per_episode_usd=("estimated_cost_usd", "mean"),
        total_hours=("elapsed_seconds", lambda s: s.sum() / 3600),
        total_cost_usd=("estimated_cost_usd", "sum"),
    ).round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="optional CSV path")
    args = parser.parse_args()

    meta = load_metadata(args.results)
    if meta.empty:
        raise SystemExit(f"No run_metadata.json with timings under {args.results}")
    table = summarise(meta)
    print(table.to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
