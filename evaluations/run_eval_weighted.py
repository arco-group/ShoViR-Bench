#!/usr/bin/env python3
"""
Weighted-average evaluation across target categories.

For oco/doco/ro experiments each sample carries a 'target_category' field.
This script:
  1. Groups predictions/references by target_category.
  2. Computes evaluation metrics independently per category.
  3. Produces a weighted average where each category is weighted by its
     sample count.
  4. Saves a CSV with one row per category plus a final 'weighted_avg' row.

Output path mirrors the run_eval.py convention:
  outputs/oco/p100/padchest-gr/foo.json
  -> results/oco/p100/padchest-gr/foo_weighted.csv

Usage:
  python evaluations/run_eval_weighted.py --filepath <path.json>
  python evaluations/run_eval_weighted.py --filepath <path.json> --scorers CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Reuse metric functions from run_eval
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import SCORER_NAME_TO_FN  # noqa: E402


DEFAULT_SCORERS = "CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L"


def _scalar(v) -> float:
    """Extract a single float from a scalar or a CheXbert-style dict."""
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    if isinstance(v, dict):
        # CheXbert returns multiple metrics; pick Macro-F1-14 as representative
        # but we will expand all keys below
        return float(next(iter(v.values())))
    raise TypeError(f"Cannot convert {type(v)} to float")


def _expand(metric_name: str, v) -> Dict[str, float]:
    """
    Turn a scorer result into a flat {column: value} dict.
    Scalars -> {metric_name: value}
    Dicts   -> {key: value, ...}  (e.g. CheXbert returns many keys)
    """
    if isinstance(v, dict):
        return {k: float(val) for k, val in v.items()}
    return {metric_name: float(v)}


def compute_metrics_for_group(
    predictions: List[str],
    references: List[str],
    scorers: List[str],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in scorers:
        fn = SCORER_NAME_TO_FN[name]
        try:
            val = fn(predictions, references, bootstrap_ci=False)
            results.update(_expand(name, val))
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")
    return results


def run(filepath: str, scorers: List[str]) -> Path:
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Group samples by target_category
    groups: Dict[str, List] = defaultdict(list)
    for sample in data:
        cat = sample.get("target_category", "unknown")
        groups[cat].append(sample)

    if not groups:
        raise ValueError("No samples found in the input file.")

    categories = sorted(groups.keys())
    print(f"Categories: {categories}")
    print(f"Sample counts: { {c: len(groups[c]) for c in categories} }")
    print(f"Scorers: {scorers}\n")

    rows: Dict[str, Dict[str, float]] = {}

    for cat in categories:
        samples = groups[cat]
        predictions = [s.get("prediction", "") or "" for s in samples]
        references  = [s.get("reference",  "") or "" for s in samples]
        n = len(samples)
        print(f"  [{cat}] n={n} ...")
        metrics = compute_metrics_for_group(predictions, references, scorers)
        rows[cat] = {"n_samples": n, **metrics}

    # Weighted average
    total_n = sum(r["n_samples"] for r in rows.values())
    metric_cols = [c for c in next(iter(rows.values())).keys() if c != "n_samples"]
    wavg: Dict[str, float] = {"n_samples": total_n}
    for col in metric_cols:
        wavg[col] = sum(
            rows[cat][col] * rows[cat]["n_samples"]
            for cat in categories
            if col in rows[cat]
        ) / total_n
    rows["weighted_avg"] = wavg

    df = pd.DataFrame(rows).T
    df.index.name = "category"
    df["n_samples"] = df["n_samples"].astype(int)

    # Build output path: outputs/... -> results/..._weighted.csv
    path_str = str(path)
    if "outputs" in path_str:
        out_str = path_str.replace("outputs", "results", 1)
        out_path = Path(out_str).with_suffix("").with_name(
            Path(out_str).stem + "_weighted.csv"
        )
    else:
        out_path = path.with_suffix("").with_name(path.stem + "_weighted.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\nSaved: {out_path}")
    print(df.to_string())
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Weighted-average evaluation across target categories.")
    parser.add_argument("--filepath", required=True, help="Path to model output JSON.")
    parser.add_argument(
        "--scorers",
        default=DEFAULT_SCORERS,
        help=f"Comma-separated list of scorers. Default: {DEFAULT_SCORERS}",
    )
    args = parser.parse_args()

    scorers = [s.strip() for s in args.scorers.split(",")]
    unknown = [s for s in scorers if s not in SCORER_NAME_TO_FN]
    if unknown:
        parser.error(f"Unknown scorers: {unknown}. Available: {list(SCORER_NAME_TO_FN.keys())}")

    run(args.filepath, scorers)


if __name__ == "__main__":
    main()
