#!/usr/bin/env python3
"""
Radiology report generation evaluation script.

It loads a JSON file containing model predictions and references, computes a set of
standard report-generation metrics, and saves the results.

Two output modes (selectable via CLI flag):
1) Per-file CSV (legacy mode):
   - If the input path contains 'outputs', replace it with 'results' and keep the subpath
     after 'outputs', changing extension to .csv.
   - Example:
       .../outputs/baseline/mimic-cxr-jpg/foo.json
       -> results/baseline/mimic-cxr-jpg/foo.csv

2) Per-experiment+dataset CSV (aggregated mode):
   - Create/update one CSV file per (experiment, dataset), where:
       experiment = folder right after 'outputs'
       dataset    = folder right after experiment
   - Example:
       .../outputs/baseline/mimic-cxr-jpg/foo.json
       -> results/baseline/mimic-cxr-jpg/results.csv
     Each model/run becomes a row. The row index is the JSON filename stem (e.g., "foo").
   - If bootstrap_ci=False, columns are metric names.
   - If bootstrap_ci=True, columns are flattened as metric_median / metric_ci_l / metric_ci_h.

GREEN metric:
- Optional (disabled by default). Enable with --compute-green.
- Adds two columns: GREEN_mean and GREEN_std (no bootstrap CI for GREEN).
- Uses GREEN model name configurable via --green-model-name.

Breakdowns:
- If enabled, saves CheXbert breakdown CSVs next to the chosen output location:
  - per-file: next to the per-file CSV
  - per-experiment: inside results/<experiment>/<dataset>/

Expected JSON format:
[
  {"prediction": "...", "reference": "..."},
  ...
]
"""
from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
import argparse
import json
import os
import random
import time

# Reduce CUDA fragmentation during long metric runs, especially RadGraph.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU

import rrg_eval.chexbert
import rrg_eval.rouge
import rrg_eval.f1radgraph
from rrg_eval.f1radgraph import F1RadGraphv2
from rrg_eval.factuality_utils import CONDITIONS

try:
    import wandb
except ImportError:
    wandb = None
import sys
sys.path.insert(0, "GREEN")
# GREEN is optional; import only if requested.

from green_score import GREEN  # type: ignore
        

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(3)
np.random.seed(3)


# -----------------------------
# Metric functions
# Each function returns:
# - bootstrap_ci=True  -> dict with keys: median, ci_l, ci_h
# - bootstrap_ci=False -> scalar float (or list for some metrics, handled downstream)
# -----------------------------
def bleu4(predictions, references, bootstrap_ci: bool = False):
    """Compute BLEU-4. With bootstrap_ci=True, returns median/CI using SacreBLEU bootstrap."""
    if bootstrap_ci:
        ret = BLEU().corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        # NOTE: ret.score is typically on 0..100 scale in SacreBLEU
        return {"median": ret.score, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    # NOTE: evaluate BLEU is typically in 0..1 scale
    return evaluate.load("bleu").compute(predictions=predictions, references=references)["bleu"]


def bleu1(predictions, references, bootstrap_ci: bool = False):
    """Compute BLEU-1. With bootstrap_ci=True, returns median/CI using SacreBLEU bootstrap."""
    if bootstrap_ci:
        ret = BLEU(max_ngram_order=1).corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    return evaluate.load("bleu").compute(predictions=predictions, references=references, max_order=1)["bleu"]


def rougel(predictions, references, bootstrap_ci: bool = False):
    """Compute ROUGE-L."""
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rougeL"])["rougeL"]
    return evaluate.load("rouge").compute(predictions=predictions, references=references)["rougeL"]


def rouge2(predictions, references, bootstrap_ci: bool = False):
    """Compute ROUGE-2."""
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rouge2"])["rouge2"]
    return evaluate.load("rouge").compute(predictions=predictions, references=references)["rouge2"]


def bertscore(predictions, references, bootstrap_ci: bool = False):
    """
    Compute BERTScore (mean F1 over samples).
    No bootstrap CI support here; bootstrap_ci is accepted for API consistency and ignored.
    """
    out = evaluate.load("bertscore").compute(predictions=predictions, references=references)
    f1_list = out["f1"]
    return float(np.mean(f1_list))


def radgraph(predictions, references, bootstrap_ci: bool = False):
    """
    Compute F1-RadGraph v2 (partial) with optional bootstrap CI.

    Depending on the rrg_eval version, F1RadGraphv2(...) may return:
      - (float_score, per_sample_scores)
      - (dict_score, per_sample_scores) where dict_score contains 'f1-radgraph'
      - dict_score directly
    This function normalizes all cases.
    """
    model = F1RadGraphv2(reward_level="partial", batch_size=256)

    def _to_float(score_obj):
        """Extract a float score from either a float or a dict."""
        if isinstance(score_obj, (int, float, np.floating)):
            return float(score_obj)
        if isinstance(score_obj, dict):
            if "f1-radgraph" in score_obj:
                return float(score_obj["f1-radgraph"])
            for k in ("f1", "score", "radgraph_f1", "radgraph"):
                if k in score_obj:
                    return float(score_obj[k])
        raise TypeError(f"Unexpected RadGraph score type: {type(score_obj)} -> {score_obj}")

    if bootstrap_ci:
        out = model(hyps=predictions, refs=references)
        reward_list = out[1]
        bs = rrg_eval.f1radgraph.bootstrap_confidence_interval(reward_list, n_resamples=500)
        return {
            "median": float(np.median(bs.bootstrap_distribution)),
            "ci_l": float(bs.confidence_interval.low),
            "ci_h": float(bs.confidence_interval.high),
        }

    out = model(hyps=predictions, refs=references)
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return _to_float(out[0])
    return _to_float(out)


def chexbert(predictions, references, bootstrap_ci: bool = False):
    """Compute CheXbert-based label agreement metrics."""
    return rrg_eval.chexbert.evaluate(predictions, references, include_original=False, bootstrap_ci=bootstrap_ci)


SCORER_NAME_TO_FN: Dict[str, Callable[..., Any]] = {
    "ROUGE-L": rougel,
    "ROUGE-2": rouge2,
    "BLEU-4": bleu4,
    "BLEU-1": bleu1,
    "BERTScore": bertscore,
    "F1-RadGraph": radgraph,
    "CheXbert": chexbert,
}

# Maps each scorer name to the representative output column(s) it produces.
# Used to detect whether a scorer's results already exist in a saved CSV.
SCORER_OUTPUT_COLUMNS: Dict[str, List[str]] = {
    "CheXbert": ["Micro-F1-14", "Macro-F1-14", "Micro-F1-5", "Macro-F1-5",
                  "Micro-F1-14+", "Macro-F1-14+", "Micro-F1-5+", "Macro-F1-5+"],
    "F1-RadGraph": ["F1-RadGraph"],
    "BLEU-1": ["BLEU-1"],
    "BLEU-4": ["BLEU-4"],
    "ROUGE-L": ["ROUGE-L"],
    "ROUGE-2": ["ROUGE-2"],
    "BERTScore": ["BERTScore"],
}


# -----------------------------
# Evaluator
# -----------------------------
class ReportGenerationEvaluator:
    def __init__(self, scorers: Optional[List[str]] = None, bootstrap_ci: bool = False):
        if scorers is None:
            scorers = ["CheXbert"]
        self.bootstrap_ci = bootstrap_ci
        self.scorers: Dict[str, Callable[..., Any]] = {}

        for scorer_name in scorers:
            if scorer_name not in SCORER_NAME_TO_FN:
                raise NotImplementedError(f"Scorer '{scorer_name}' not implemented.")
            self.scorers[scorer_name] = SCORER_NAME_TO_FN[scorer_name]

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Run all selected scorers and postprocess into a unified dict."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: predictions={len(predictions)} vs references={len(references)}"
            )

        scores: Dict[str, Any] = {}
        for scorer_name, scorer_fn in (pbar := tqdm(self.scorers.items())):
            pbar.set_description(scorer_name)
            scores[scorer_name] = scorer_fn(predictions, references, self.bootstrap_ci)

        self.postprocess_eval(scores)
        return scores

    def postprocess_eval(self, scores: Dict[str, Any]) -> None:
        """
        Normalize outputs:
        - CheXbert expands into Micro/Macro for 14/5 classes (U-zeros and U-ones),
          and keeps breakdown tables + raw metrics for optional saving.
        - F1-RadGraph is always stored under 'F1-RadGraph' key.
        """
        if self.bootstrap_ci:
            keys = ("median", "ci_l", "ci_h")
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)

                    scores["Micro-F1-14"] = {k: metrics[0]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14"] = {k: metrics[0]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5"] = {k: metrics[1]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5"] = {k: metrics[1]["macro avg"][k] for k in keys}

                    scores["Micro-F1-14+"] = {k: metrics[2]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14+"] = {k: metrics[2]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5+"] = {k: metrics[3]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5+"] = {k: metrics[3]["macro avg"][k] for k in keys}

                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]

                elif name == "F1-RadGraph":
                    scores["F1-RadGraph"] = scores.pop(name)

        else:
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)

                    scores["Micro-F1-14"] = float(metrics[0]["micro avg"]["f1-score"])
                    scores["Macro-F1-14"] = float(metrics[0]["macro avg"]["f1-score"])
                    scores["Micro-F1-5"] = float(metrics[1]["micro avg"]["f1-score"])
                    scores["Macro-F1-5"] = float(metrics[1]["macro avg"]["f1-score"])

                    scores["Micro-F1-14+"] = float(metrics[2]["micro avg"]["f1-score"])
                    scores["Macro-F1-14+"] = float(metrics[2]["macro avg"]["f1-score"])
                    scores["Micro-F1-5+"] = float(metrics[3]["micro avg"]["f1-score"])
                    scores["Macro-F1-5+"] = float(metrics[3]["macro avg"]["f1-score"])

                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]

                elif name == "F1-RadGraph":
                    val = scores.pop(name)
                    if isinstance(val, dict) and "f1-radgraph" in val:
                        scores["F1-RadGraph"] = float(val["f1-radgraph"])
                    else:
                        scores["F1-RadGraph"] = float(val)


# -----------------------------
# Helper: build a "nice" main results table
# -----------------------------
MAIN_METRICS_ORDER = [
    "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
    "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
    "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
]


def build_main_results_table(results: Dict[str, Any], bootstrap_ci: bool) -> pd.DataFrame:
    """
    Return a DataFrame formatted as:
    - bootstrap_ci=True  -> index: ['median','ci_l','ci_h'], columns: metrics
    - bootstrap_ci=False -> index: ['score'], columns: metrics
    """
    if bootstrap_ci:
        main_dict = {k: v for k, v in results.items() if k not in ("breakdown+", "breakdown-", "chexbert_metrics")}
        df = pd.DataFrame.from_dict(main_dict)
        cols = [c for c in MAIN_METRICS_ORDER if c in df.columns]
        return df[cols]

    row: Dict[str, float] = {}
    for k in MAIN_METRICS_ORDER:
        v = results.get(k, None)
        if isinstance(v, (int, float, np.floating)):
            row[k] = float(v)

    return pd.DataFrame([row], index=["score"])


def main_results_to_single_row(main_results: pd.DataFrame, bootstrap_ci: bool) -> Dict[str, float]:
    """
    Flatten main_results into a single dict suitable for one aggregated CSV row.

    - bootstrap_ci=False: {metric: value}
    - bootstrap_ci=True : {metric_median: ..., metric_ci_l: ..., metric_ci_h: ...}
    """
    row: Dict[str, float] = {}

    if not bootstrap_ci:
        idx = main_results.index[0]  # usually "score"
        for col in main_results.columns:
            row[col] = float(main_results.loc[idx, col])
        return row

    for stat in main_results.index:
        for col in main_results.columns:
            row[f"{col}_{stat}"] = float(main_results.loc[stat, col])

    return row


# -----------------------------
# Output path derivation
# -----------------------------
def derive_per_file_output_paths(input_path: str) -> Dict[str, Path]:
    """
    Per-file output mode:
    - If input contains 'outputs', replace it with 'results' and keep subpath after 'outputs'
      changing extension to .csv.
    - Otherwise: results/<input_stem>.csv
    """
    p = Path(input_path).resolve()
    parts = list(p.parts)

    if "outputs" in parts:
        idx = parts.index("outputs")
        rel_after_outputs = Path(*parts[idx + 1 :])
        base_no_ext = rel_after_outputs.with_suffix("")
        main_csv = Path("results") / base_no_ext
    else:
        main_csv = Path("results") / p.with_suffix("").name

    main_csv = main_csv.with_suffix(".csv")

    return {
        "main": main_csv,
        "breakdown_p": main_csv.with_name(main_csv.stem + "__breakdown_p.csv"),
        "breakdown_n": main_csv.with_name(main_csv.stem + "__breakdown_n.csv"),
        "chexbert_detailed": main_csv.with_name(main_csv.stem + "__chexbert_detailed.csv"),
    }


def derive_experiment_dataset_results_csv(input_path: str) -> Path:
    """
    Per-experiment+dataset aggregated mode:
    - Find 'outputs/<experiment>/<dataset>/' and write to:
        results/<experiment>/<dataset>/results.csv
    - If 'outputs' is missing, fall back to:
        results/unknown_experiment/unknown_dataset/results.csv
    """
    p = Path(input_path).resolve()
    parts = list(p.parts)

    experiment = "unknown_experiment"
    dataset = "unknown_dataset"

    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            experiment = parts[idx + 1]
        if idx + 2 < len(parts):
            dataset = parts[idx + 2]

    return Path("results") / experiment / dataset / "results.csv"


def upsert_aggregated_csv(csv_path: Path, model_name: str, row_dict: Dict[str, float]) -> None:
    """
    Create or update the aggregated CSV file.
    - Each model_name is a row index.
    - If the model already exists, overwrite its row.
    - Otherwise append a new row.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame()

    # Add any missing columns
    for k in row_dict.keys():
        if k not in df.columns:
            df[k] = np.nan

    df.loc[model_name, list(row_dict.keys())] = list(row_dict.values())

    # Sort for readability
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv(csv_path)


def breakdown_output_dir_for_mode(filepath: str, output_mode: str) -> Path:
    """
    Decide where to write breakdown CSVs depending on output mode.
    - per-file: next to the per-file CSV
    - per-experiment: inside results/<experiment>/<dataset>/
    """
    if output_mode == "per-file":
        per_file_paths = derive_per_file_output_paths(filepath)
        return per_file_paths["main"].parent

    csv_path = derive_experiment_dataset_results_csv(filepath)
    return csv_path.parent


# -----------------------------
# GREEN metric (optional)
# -----------------------------
def compute_green_metric(
    refs: List[str],
    hyps: List[str],
    model_name: str,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Compute GREEN metric and return only mean and std.
    """
    green_scorer = GREEN(model_name, output_dir=str(output_dir))
    mean, std, _green_score_list, _summary, _result_df = green_scorer(refs, hyps)
    return {"GREEN_mean": float(mean), "GREEN_std": float(std)}


# -----------------------------
# Incremental evaluation helpers
# -----------------------------
def _load_existing_row(filepath: str, output_mode: str, bootstrap_ci: bool) -> Dict[str, float]:
    """Load already-computed metric values for this model from saved CSVs.

    Lookup order for *per-experiment* mode:
      1. Per-file CSV  (results/<experiment>/<dataset>/<model>.csv)
      2. Aggregated CSV (results/<experiment>/<dataset>/results.csv)

    For *per-file* mode only the per-file CSV is checked.

    Returns a **flat** dict  {column_name: value}  using the same format as
    ``main_results_to_single_row`` (with ``_median``/``_ci_l``/``_ci_h``
    suffixes when *bootstrap_ci* is True for per-experiment rows).
    An empty dict means nothing was found.
    """
    model_name = Path(filepath).with_suffix("").name
    row: Dict[str, float] = {}

    # --- try per-file CSV first (works for both modes) ---
    per_file_csv = derive_per_file_output_paths(filepath)["main"]
    if per_file_csv.exists():
        df = pd.read_csv(per_file_csv, index_col=0)
        # Per-file DF: index = ['score'] or ['median','ci_l','ci_h']
        if output_mode == "per-experiment":
            # Flatten into the per-experiment row format
            if bootstrap_ci:
                for stat in df.index:
                    for col in df.columns:
                        val = df.loc[stat, col]
                        if not pd.isna(val):
                            row[f"{col}_{stat}"] = float(val)
            else:
                idx = df.index[0]
                for col in df.columns:
                    val = df.loc[idx, col]
                    if not pd.isna(val):
                        row[col] = float(val)
        else:
            # per-file mode: same flat format
            idx = df.index[0] if not bootstrap_ci else "median"
            if idx in df.index:
                for col in df.columns:
                    val = df.loc[idx, col]
                    if not pd.isna(val):
                        row[col] = float(val)
        if row:
            return row

    # --- for per-experiment, also try the aggregated CSV ---
    if output_mode == "per-experiment":
        agg_csv = derive_experiment_dataset_results_csv(filepath)
        if agg_csv.exists():
            df = pd.read_csv(agg_csv, index_col=0)
            if model_name in df.index:
                for col in df.columns:
                    val = df.loc[model_name, col]
                    if not pd.isna(val):
                        row[col] = float(val)

    return row


def _find_missing_scorers(
    existing_row: Dict[str, float],
    requested_scorers: List[str],
    bootstrap_ci: bool,
) -> List[str]:
    """Return the subset of *requested_scorers* whose results are NOT in *existing_row*."""
    missing: List[str] = []
    for scorer in requested_scorers:
        output_cols = SCORER_OUTPUT_COLUMNS.get(scorer, [scorer])
        # With bootstrap CI in per-experiment format, columns have _median suffix
        if bootstrap_ci:
            check_cols = [f"{c}_median" for c in output_cols]
        else:
            check_cols = output_cols
        if not all(c in existing_row for c in check_cols):
            missing.append(scorer)
    return missing


def _green_is_missing(existing_row: Dict[str, float]) -> bool:
    """Check whether GREEN values are absent from an existing row."""
    return "GREEN_mean" not in existing_row



# -----------------------------
# Main runner
# -----------------------------
def run(
    filepath: str,
    scorers: Optional[List[str]] = None,
    report_chexbert_f1: bool = False,
    bootstrap_ci: bool = False,
    run_name: str = "mimic_cxr_eval",
    save_breakdowns: bool = False,
    output_mode: str = "per-file",  # "per-file" or "per-experiment"
    compute_green: bool = False,
    green_model_name: str = "StanfordAIMI/GREEN-radllama2-7b",
    skip_existing: bool = False,
) -> None:
    model_name = Path(filepath).with_suffix("").name

    # Default scorers (GREEN is controlled separately)
    if scorers is None:
        scorers = ["CheXbert", "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"]

    # --- Determine what already exists and what is missing ---
    existing_row: Dict[str, float] = {}
    if skip_existing:
        existing_row = _load_existing_row(filepath, output_mode, bootstrap_ci)

    missing_scorers = _find_missing_scorers(existing_row, scorers, bootstrap_ci) if existing_row else scorers
    need_green = compute_green and _green_is_missing(existing_row) if existing_row else compute_green

    # If everything is already computed, skip evaluation but still populate per-experiment CSV
    if skip_existing and not missing_scorers and not need_green:
        if output_mode == "per-experiment" and existing_row:
            csv_path = derive_experiment_dataset_results_csv(filepath)
            upsert_aggregated_csv(csv_path, model_name=model_name, row_dict=existing_row)
            print(f"[SKIP] All metrics already exist for '{model_name}'. "
                  f"Populated aggregated CSV from per-file results: {csv_path}")
        else:
            print(f"[SKIP] All requested metrics already exist for '{model_name}' ({output_mode}). "
                  "Use without --skip-existing to override.")
        return

    if existing_row:
        found = [s for s in scorers if s not in missing_scorers]
        if found:
            print(f"[REUSE] Loaded existing results for: {', '.join(found)}")
        if missing_scorers:
            print(f"[COMPUTE] Missing scorers to compute: {', '.join(missing_scorers)}")
        if need_green:
            print(f"[COMPUTE] GREEN metric missing — will compute")
        elif compute_green and not need_green:
            print(f"[REUSE] GREEN already present")

    t_start = time.perf_counter()

    # --- Load predictions/references (only if we need to compute something) ---
    preds: List[str] = []
    refs: List[str] = []
    results: Dict[str, Any] = {}

    if missing_scorers or need_green:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            preds.append(item.get("prediction", ""))
            refs.append(item.get("reference", ""))
        print(f"Total reports: {len(preds)}\n")

    # --- Compute only the missing core metrics ---
    if missing_scorers:
        evaluator = ReportGenerationEvaluator(scorers=missing_scorers, bootstrap_ci=bootstrap_ci)
        results = evaluator.evaluate(preds, refs)

    # Build the main results table from newly computed metrics
    main_results = build_main_results_table(results, bootstrap_ci=bootstrap_ci) if results else pd.DataFrame()

    # Flatten newly computed results into a row dict
    new_row: Dict[str, float] = {}
    if not main_results.empty:
        new_row = main_results_to_single_row(main_results, bootstrap_ci=bootstrap_ci)

    print("========== Main Results (core) ==========")
    if not main_results.empty:
        print(main_results)
    else:
        print("(all core metrics loaded from cache)")
    print("")

    # --- Compute GREEN if missing and requested ---
    green_vals: Dict[str, float] = {}
    if need_green:
        out_dir_for_green = breakdown_output_dir_for_mode(filepath, output_mode)
        out_dir_for_green.mkdir(parents=True, exist_ok=True)
        green_vals = compute_green_metric(
            refs=refs,
            hyps=preds,
            model_name=green_model_name,
            output_dir=out_dir_for_green,
        )
        print("========== GREEN ==========")
        print(green_vals)
        print("")
    elif compute_green and not _green_is_missing(existing_row):
        # Carry forward existing GREEN values
        green_vals = {
            "GREEN_mean": existing_row["GREEN_mean"],
            "GREEN_std": existing_row.get("GREEN_std", float("nan")),
        }

    # --- Merge existing + new into a single row ---
    merged_row: Dict[str, float] = {}
    merged_row.update(existing_row)   # start with what we had
    merged_row.update(new_row)        # overwrite with freshly computed
    if green_vals:
        merged_row.update(green_vals)

    # --- Save outputs ---
    if output_mode == "per-file":
        out_paths = derive_per_file_output_paths(filepath)
        out_paths["main"].parent.mkdir(parents=True, exist_ok=True)

        # Rebuild a full main_results DF for per-file saving
        if bootstrap_ci:
            # Reconstruct from merged_row: columns without suffix -> rows median/ci_l/ci_h
            stats = ["median", "ci_l", "ci_h"]
            col_set: set = set()
            for k in merged_row:
                for s in stats:
                    if k.endswith(f"_{s}"):
                        col_set.add(k[: -(len(s) + 1)])
                        break
            # Also include non-suffixed keys (e.g. GREEN_mean, GREEN_std)
            save_dict: Dict[str, Dict[str, float]] = {}
            for col in col_set:
                save_dict[col] = {s: merged_row.get(f"{col}_{s}", float("nan")) for s in stats}
            save_df = pd.DataFrame(save_dict)
            # Add GREEN columns if present
            if green_vals:
                save_df["GREEN_mean"] = float("nan")
                save_df["GREEN_std"] = float("nan")
                if "median" in save_df.index:
                    save_df.loc["median", "GREEN_mean"] = green_vals["GREEN_mean"]
                    save_df.loc["median", "GREEN_std"] = green_vals["GREEN_std"]
            cols = [c for c in MAIN_METRICS_ORDER if c in save_df.columns]
            extra = [c for c in save_df.columns if c not in MAIN_METRICS_ORDER]
            save_df = save_df[cols + extra]
        else:
            # Simple: one row from merged_row
            save_df = pd.DataFrame([merged_row], index=["score"])
            cols = [c for c in MAIN_METRICS_ORDER if c in save_df.columns]
            extra = [c for c in save_df.columns if c not in MAIN_METRICS_ORDER]
            save_df = save_df[cols + extra]

        save_df.to_csv(out_paths["main"], index=True)
        print(f"Saved per-file main results to: {out_paths['main']}")

    elif output_mode == "per-experiment":
        csv_path = derive_experiment_dataset_results_csv(filepath)
        upsert_aggregated_csv(csv_path, model_name=model_name, row_dict=merged_row)
        print(f"Updated aggregated CSV: {csv_path} (row='{model_name}')")

    else:
        raise ValueError("Invalid output_mode. Use 'per-file' or 'per-experiment'.")

    # Optional W&B logging
    if wandb:
        wandb_results = {k: v for k, v in merged_row.items() if not pd.isna(v)}
        wandb.init(name=run_name)
        wandb.log(wandb_results)

    # Optionally save CheXbert breakdown tables (only if CheXbert was computed this run)
    if save_breakdowns and "breakdown+" in results:
        out_dir = breakdown_output_dir_for_mode(filepath, output_mode)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = model_name
        breakdown_p_path = out_dir / f"{prefix}__breakdown_p.csv"
        breakdown_n_path = out_dir / f"{prefix}__breakdown_n.csv"
        chexbert_detailed_path = out_dir / f"{prefix}__chexbert_detailed.csv"

        print("========== CheXbert F1 (uncertain as positive) ==========")
        breakdown_p = pd.DataFrame(results["breakdown+"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ["f1-score", "precision", "recall", "support"]
        ]
        print(breakdown_p)
        print("")
        breakdown_p.to_csv(breakdown_p_path)
        print(f"Saved breakdown_p to: {breakdown_p_path}")

        print("========== CheXbert F1 (uncertain as negative) ==========")
        breakdown_n = pd.DataFrame(results["breakdown-"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ["f1-score", "precision", "recall", "support"]
        ]
        print(breakdown_n)
        print("")
        breakdown_n.to_csv(breakdown_n_path)
        print(f"Saved breakdown_n to: {breakdown_n_path}")

        if report_chexbert_f1:
            print("========== CheXbert Detailed Metrics ==========")
            chexbert_df = pd.DataFrame(results["chexbert_metrics"])[sorted(CONDITIONS) + ["avg"]].T[
                ["positive f1", "negation f1", "uncertain f1", "blank f1", "weighted f1", "kappas"]
            ]
            print(chexbert_df)
            print("")
            chexbert_df.to_csv(chexbert_detailed_path)
            print(f"Saved chexbert detailed to: {chexbert_detailed_path}")

    elapsed = time.perf_counter() - t_start
    mins, secs = divmod(elapsed, 60)
    print(f"\n[TIME] Evaluation for '{model_name}' completed in {int(mins)}m {secs:.1f}s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate radiology report generation outputs.")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the JSON predictions file.")
    parser.add_argument(
        "--scorers",
        type=str,
        default="CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L",
        help="Comma-separated list of scorers (e.g. 'CheXbert,F1-RadGraph,ROUGE-L').",
    )
    parser.add_argument("--bootstrap-ci", action="store_true", help="Enable bootstrap confidence intervals.")
    parser.add_argument("--save-breakdowns", action="store_true", help="Save CheXbert breakdown CSVs.")
    parser.add_argument("--report-chexbert-f1", action="store_true", help="Save extra CheXbert detailed metrics CSV.")
    parser.add_argument("--run-name", type=str, default="mimic_cxr_eval", help="W&B run name (if wandb installed).")
    parser.add_argument(
        "--output-mode",
        type=str,
        default="per-file",
        choices=["per-file", "per-experiment"],
        help="Output mode: 'per-file' saves one CSV per JSON; 'per-experiment' appends a row to results/<experiment>/<dataset>/results.csv.",
    )

    # Override control
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip evaluation if results already exist for this model/file. "
             "Without this flag, existing results are overwritten (default behaviour).",
    )

    # GREEN options
    parser.add_argument(
        "--compute-green",
        action="store_true",
        help="Compute GREEN metric and store GREEN_mean/GREEN_std in outputs.",
    )
    parser.add_argument(
        "--green-model-name",
        type=str,
        default="StanfordAIMI/GREEN-radllama2-7b",
        help="Hugging Face model name for GREEN.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]

    run(
        filepath=args.filepath,
        scorers=scorers,
        report_chexbert_f1=args.report_chexbert_f1,
        bootstrap_ci=args.bootstrap_ci,
        run_name=args.run_name,
        save_breakdowns=args.save_breakdowns,
        output_mode=args.output_mode,
        compute_green=args.compute_green,
        green_model_name=args.green_model_name,
        skip_existing=args.skip_existing,
    )
