"""
Plot model performance vs. random occlusion (RO) percentage.

Reads per-experiment results CSVs from results/ro/pXX/results.csv and
produces one line plot per metric, with models as separate lines and
occlusion percentage on the x-axis.

Usage:
    python -m src.postprocessing.plot_ro                          # default metrics
    python -m src.postprocessing.plot_ro --metrics Micro-F1-5 F1-RadGraph
    python -m src.postprocessing.plot_ro --results-dir results/ro --out-dir figures/ro
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------
# Friendly model labels (strip HF org prefixes and metadata suffixes)
# ---------------------
MODEL_SHORT_NAMES: Dict[str, str] = {
    "google__medgemma": "MedGemma",
    "microsoft__maira-2": "MAIRA-2",
    "aehrc__cxrmate-rrg24": "CXRMate-ED",
    "nvidia__NV-Reason-CXR-3B": "NV-Reason-CXR",
    "StanfordAIMI__CheXagent": "CheXagent",
    "StanfordAIMI__CheXOne": "CheXOne",
    "X-iZhang__libra-v1.0-7b": "Libra",
    "X-iZhang__libra-llava-rad": "LLaVA-Rad",
    "Chantal__RaDialog": "RaDialog",
}


def shorten_model_name(raw: str) -> str:
    """Convert raw CSV index to a readable model name."""
    for prefix, short in MODEL_SHORT_NAMES.items():
        if raw.startswith(prefix):
            return short
    return raw


# ---------------------
# Data loading
# ---------------------
def load_ro_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Load all results/ro/pXX/results.csv into a single DataFrame.

    Returns:
        DataFrame with columns: model, percentage, <metric1>, <metric2>, ...
    """
    results_dir = Path(results_dir)
    rows: list[dict] = []

    for pdir in sorted(results_dir.iterdir()):
        if not pdir.is_dir():
            continue
        m = re.match(r"^p(\d+)$", pdir.name)
        if not m:
            continue
        pct = int(m.group(1))

        csv_path = pdir / "results.csv"
        if not csv_path.exists():
            # Try per-file CSVs under padchest-gr/
            csv_path = None
            per_file_dir = pdir / "padchest-gr"
            if per_file_dir.is_dir():
                for f in sorted(per_file_dir.glob("*.csv")):
                    df_single = pd.read_csv(f, index_col=0)
                    if df_single.empty:
                        continue
                    row = df_single.iloc[0].to_dict()
                    row["model"] = shorten_model_name(f.stem)
                    row["percentage"] = pct
                    rows.append(row)
            continue

        df = pd.read_csv(csv_path, index_col=0)
        for raw_name, series in df.iterrows():
            row = series.to_dict()
            row["model"] = shorten_model_name(str(raw_name))
            row["percentage"] = pct
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No RO results found under {results_dir}. "
            "Run evaluations first with: sbatch scripts/run_all_evals.sh --experiment ro --output-mode per-experiment"
        )

    combined = pd.DataFrame(rows)
    combined = combined.sort_values(["model", "percentage"]).reset_index(drop=True)
    return combined


# ---------------------
# Plotting
# ---------------------
METRIC_DISPLAY: Dict[str, str] = {
    "Micro-F1-5": "Micro F1-5 (top-5 diseases, U=neg)",
    "Micro-F1-5+": "Micro F1-5 (top-5 diseases, U=pos)",
    "Macro-F1-5": "Macro F1-5 (top-5 diseases, U=neg)",
    "Macro-F1-5+": "Macro F1-5 (top-5 diseases, U=pos)",
    "Micro-F1-14": "Micro F1-14 (all diseases, U=neg)",
    "Macro-F1-14": "Macro F1-14 (all diseases, U=neg)",
    "F1-RadGraph": "F1-RadGraph",
    "BLEU-4": "BLEU-4",
    "BLEU-1": "BLEU-1",
    "ROUGE-L": "ROUGE-L",
}

# One marker per model for visual distinction
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    figsize: tuple = (8, 5),
) -> Path:
    """
    Create a single line plot: x=percentage, y=metric, one line per model.
    Saves to out_dir/<metric>.pdf and returns the path.
    """
    if metric not in df.columns:
        available = [c for c in df.columns if c not in ("model", "percentage")]
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    fig, ax = plt.subplots(figsize=figsize)

    models = sorted(df["model"].unique())
    for i, model in enumerate(models):
        mdf = df[df["model"] == model].sort_values("percentage")
        ax.plot(
            mdf["percentage"],
            mdf[metric],
            marker=MARKERS[i % len(MARKERS)],
            label=model,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Random Occlusion (%)", fontsize=12)
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=12)
    ax.set_title(f"{METRIC_DISPLAY.get(metric, metric)} vs. Occlusion Level", fontsize=13)

    # Force x-ticks to the actual percentage values present
    pcts = sorted(df["percentage"].unique())
    ax.set_xticks(pcts)
    ax.set_xticklabels([f"{p}%" for p in pcts])

    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = metric.replace("+", "_pos").replace("-", "_")
    out_path = out_dir / f"{safe_name}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_all(
    df: pd.DataFrame,
    metrics: List[str],
    out_dir: Path,
) -> List[Path]:
    """Plot all requested metrics and return output paths."""
    paths = []
    for metric in metrics:
        paths.append(plot_metric(df, metric, out_dir))
    return paths


# ---------------------
# CLI
# ---------------------
DEFAULT_METRICS = [
    "Micro-F1-5",
    "Macro-F1-5",
    "F1-RadGraph",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model performance vs. RO occlusion percentage."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/ro",
        help="Directory containing pXX/ subdirectories with results.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures/ro",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Metrics to plot (default: {DEFAULT_METRICS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_ro_results(args.results_dir)

    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, "
          f"percentages: {sorted(df['percentage'].unique())}")
    print()

    plot_all(df, args.metrics, Path(args.out_dir))
