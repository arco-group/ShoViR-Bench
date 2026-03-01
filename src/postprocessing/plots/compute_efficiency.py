"""Scatter plots: compute burden vs clinical performance (baseline, PadChest-GR).

For each clinical metric, one scatter plot is produced with:
  - x-axis: model size in billions of parameters (from compute_stats JSONs)
  - y-axis: metric value (from results/baseline/padchest-gr/results.csv)
  - Optional second x-axis: inference throughput (img/s)

Reads:
  results/compute_stats/<model_key>.json   (one per model, from compute_model_stats.py)
  results/baseline/padchest-gr/results.csv

Output:
  results/plots/compute_efficiency.{pdf,png}

Usage:
    # Full run (requires compute_stats JSONs to exist):
    python -m src.postprocessing.plots.compute_efficiency

    # Quick test with synthetic dummy data (no JSONs needed):
    python -m src.postprocessing.plots.compute_efficiency --test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results"
STATS_DIR   = RESULTS_DIR / "compute_stats"
BASELINE_CSV = RESULTS_DIR / "baseline" / "padchest-gr" / "results.csv"
OUTPUT_DIR  = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Plot style (matches existing plots) ─────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Didot", "GFS Didot", "CMU Serif", "Computer Modern",
                         "Georgia", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size":        11,
    "axes.labelsize":   13,
    "axes.titlesize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

# ── Model identity ───────────────────────────────────────────────────────────
# Maps the CSV row index prefix → short display name
_ROW_PREFIX_TO_NAME: dict[str, str] = {
    "Chantal__RaDialog":      "RaDialog",
    "StanfordAIMI__CheXagent": "CheXagent-2",
    "X-iZhang__libra-llava":  "LIBRA-LLaVA",
    "X-iZhang__libra-v1":     "LIBRA-v1",
    "aehrc__cxrmate":         "CXRMate",
    "google__medgemma":       "MedGemma",
    "microsoft__maira":       "MAIRA-2",
    "nvidia__NV-Reason":      "NV-Reason-CXR",
    "StanfordAIMI__CheXOne":  "CheXOne",
}

# Fallback parameter counts (billions) from model cards — used when no JSON present
_KNOWN_PARAMS_B: dict[str, float] = {
    "RaDialog":      7.0,
    "CheXagent-2":   3.0,
    "LIBRA-LLaVA":   7.0,
    "LIBRA-v1":      7.0,
    "CXRMate":       0.35,
    "MedGemma":      4.3,
    "MAIRA-2":       8.0,
    "NV-Reason-CXR": 3.0,
    "CheXOne":       7.0,
}

# Colours assigned per model (consistent across plots)
_MODEL_COLORS: dict[str, str] = {
    "RaDialog":      "#CC6677",
    "CheXagent-2":   "#DDCC77",
    "LIBRA-LLaVA":   "#44AA99",
    "LIBRA-v1":      "#4477AA",
    "CXRMate":       "#228833",
    "MedGemma":      "#AA3377",
    "MAIRA-2":       "#66CCEE",
    "NV-Reason-CXR": "#EE7733",
    "CheXOne":       "#BBBBBB",
}

# Clinical metrics to include (column name in results.csv → display label)
CLINICAL_METRICS: dict[str, str] = {
    "F1-RadGraph":  "F1-RadGraph",
    "Micro-F1-14":  "CheXBERT Micro-F1 (14 cls)",
    "Macro-F1-14":  "CheXBERT Macro-F1 (14 cls)",
    "Micro-F1-5":   "CheXBERT Micro-F1 (5 cls)",
    "Macro-F1-5":   "CheXBERT Macro-F1 (5 cls)",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _resolve_display_name(row_index: str) -> str | None:
    for prefix, name in _ROW_PREFIX_TO_NAME.items():
        if row_index.startswith(prefix):
            return name
    return None


def load_clinical_metrics() -> pd.DataFrame:
    """Return DataFrame indexed by display name with clinical metric columns."""
    df = pd.read_csv(BASELINE_CSV, index_col=0)
    records = []
    for row_idx, row in df.iterrows():
        name = _resolve_display_name(str(row_idx))
        if name is None:
            continue
        rec = {"display_name": name}
        for col in CLINICAL_METRICS:
            if col in df.columns:
                rec[col] = float(row[col])
        records.append(rec)
    return pd.DataFrame(records).set_index("display_name")


def load_compute_stats() -> dict[str, dict]:
    """Return {display_name: stats_dict} from results/compute_stats/*.json."""
    stats: dict[str, dict] = {}
    if not STATS_DIR.exists():
        return stats
    for p in sorted(STATS_DIR.glob("*.json")):
        try:
            with p.open() as f:
                d = json.load(f)
            name = d.get("display_name") or p.stem
            stats[name] = d
        except Exception:
            continue
    return stats


def build_data_table(
    metrics_df: pd.DataFrame,
    compute_stats: dict[str, dict],
) -> pd.DataFrame:
    """Merge clinical metrics with compute stats into one DataFrame."""
    rows = []
    for name in metrics_df.index:
        row: dict = {"display_name": name}

        # Parameters
        if name in compute_stats:
            row["params_B"]    = compute_stats[name].get("n_params_total_B")
            row["throughput"]  = compute_stats[name].get("throughput_img_per_s")
            row["mean_time_s"] = compute_stats[name].get("mean_inference_time_s")
            row["gflops"]      = compute_stats[name].get("gflops")
            row["gmacs"]       = compute_stats[name].get("gmacs")
            row["flops_method"]= compute_stats[name].get("flops_method")
        else:
            row["params_B"]    = _KNOWN_PARAMS_B.get(name)
            row["throughput"]  = None
            row["mean_time_s"] = None
            row["gflops"]      = None
            row["gmacs"]       = None
            row["flops_method"]= None

        # Clinical metrics
        for col in CLINICAL_METRICS:
            if col in metrics_df.columns:
                row[col] = metrics_df.loc[name, col] if name in metrics_df.index else None

        rows.append(row)
    return pd.DataFrame(rows).set_index("display_name")


def _dummy_data() -> pd.DataFrame:
    """Return synthetic data for --test mode (no files required)."""
    rng = np.random.default_rng(0)
    models = list(_KNOWN_PARAMS_B.keys())
    records = []
    for m in models:
        p = _KNOWN_PARAMS_B[m]
        # Simulate: larger models tend to do better (with noise)
        base = 0.05 + 0.06 * np.log1p(p)
        gflops = float(p * 1e3 * rng.uniform(0.8, 1.2))   # rough: 1 GFLOPs per M params per token
        rec = {
            "display_name": m,
            "params_B":     p,
            "throughput":   float(rng.uniform(0.05, 0.5)),
            "mean_time_s":  float(rng.uniform(1.5, 25.0)),
            "gflops":       round(gflops, 1),
            "gmacs":        round(gflops / 2, 1),
            "flops_method": "dummy",
        }
        for col in CLINICAL_METRICS:
            rec[col] = float(np.clip(base + rng.normal(0, 0.03), 0.01, 0.99))
        records.append(rec)
    return pd.DataFrame(records).set_index("display_name")


# ── Plotting ─────────────────────────────────────────────────────────────────

def _annotate(ax, x, y, label, color):
    ax.scatter(x, y, color=color, s=80, zorder=5, edgecolors="white", linewidths=0.8)
    txt = ax.annotate(
        label, (x, y),
        xytext=(6, 4), textcoords="offset points",
        fontsize=8.5, color=color,
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=2, foreground="white"),
    ])



def _plot_x_vs_metrics(
    data: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    out_stem: str,
) -> None:
    """Generic scatter: x_col (x) vs each clinical metric (y)."""
    sub_data = data.dropna(subset=[x_col])
    if sub_data.empty:
        print(f"No {x_col} data available — skipping {out_stem} plot.")
        return

    metrics = [c for c in CLINICAL_METRICS if c in sub_data.columns]
    n = len(metrics)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, metric in zip(axes, metrics):
        sub = sub_data[[metric, x_col]].dropna()
        for name, row in sub.iterrows():
            color = _MODEL_COLORS.get(str(name), "#888888")
            _annotate(ax, row[x_col], row[metric], str(name), color)

        ax.set_xlabel(x_label, labelpad=6)
        ax.set_ylabel(CLINICAL_METRICS[metric], labelpad=6)
        ax.set_title(CLINICAL_METRICS[metric])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.axhline(sub[metric].mean(), color="#cccccc", lw=0.8, ls="--", zorder=1)
        ax.axvline(sub[x_col].mean(),  color="#cccccc", lw=0.8, ls="--", zorder=1)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{title}\n(PadChest-GR · Baseline)", fontsize=13, y=1.01)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = OUTPUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(data: pd.DataFrame) -> None:
    cols = (
        ["params_B", "gflops", "gmacs", "flops_method", "throughput", "mean_time_s"]
        + list(CLINICAL_METRICS.keys())
    )
    present = [c for c in cols if c in data.columns]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 180)
    print("\n── Compute × Clinical Metrics ──────────────────────────────────")
    print(data[present].to_string())
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot compute–performance trade-off (baseline, PadChest-GR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Use synthetic dummy data (no JSONs or CSV required).",
    )
    parser.add_argument(
        "--out-stem", default="compute_efficiency",
        help="Output filename stem (without extension).",
    )
    args = parser.parse_args()

    if args.test:
        print("[test mode] Using synthetic dummy data.")
        data = _dummy_data()
    else:
        print(f"Loading clinical metrics from: {BASELINE_CSV}")
        metrics_df = load_clinical_metrics()
        print(f"  {len(metrics_df)} models found in baseline CSV.")

        compute_stats = load_compute_stats()
        if compute_stats:
            print(f"  {len(compute_stats)} compute-stats JSONs found.")
        else:
            print("  No compute-stats JSONs found — using known parameter counts as fallback.")

        data = build_data_table(metrics_df, compute_stats)

    print_summary_table(data)

    # Parameters (B) vs metrics
    _plot_x_vs_metrics(
        data, "params_B", "Parameters (B)",
        "Model Size–Performance Trade-off",
        args.out_stem + "_params",
    )
    # GFLOPs vs metrics  (prefill forward pass)
    _plot_x_vs_metrics(
        data, "gflops", "GFLOPs (prefill forward pass)",
        "Compute (GFLOPs)–Performance Trade-off",
        args.out_stem + "_gflops",
    )
    # GMACs vs metrics
    _plot_x_vs_metrics(
        data, "gmacs", "GMACs (prefill forward pass)",
        "Compute (GMACs)–Performance Trade-off",
        args.out_stem + "_gmacs",
    )
    # Throughput vs metrics (generate speed)
    _plot_x_vs_metrics(
        data, "throughput", "Throughput (img / s)",
        "Throughput–Performance Trade-off",
        args.out_stem + "_throughput",
    )


if __name__ == "__main__":
    main()
