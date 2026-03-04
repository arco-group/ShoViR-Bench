"""
Bubble plot: GFLOPs (x) vs weighted-mean CheXBERT F1 at p=0 (y).

Layout: 1 row × 2 cols — left: PadChest-GR, right: MIMIC-CXR.
Bubble area ∝ n_params_total_B (from compute_stats JSONs).
Colors / markers are identical to line_plot_p.py (MODEL_PALETTE).

Data sources:
  results/compute_stats/<model>.json     → GFLOPs, n_params_total_B
  results/oco/p00/<dataset>/<model>.csv  → per-disease CheXBERT F1 at p=0

Output:
  results/plots/gflops_bubble.{pdf,png}
"""
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results"
STATS_DIR   = RESULTS_DIR / "compute_stats"
P0_DIR      = RESULTS_DIR / "oco" / "p00"
OUTPUT_DIR  = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style (matches line_plot_p.py) ───────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Didot", "GFS Didot", "CMU Serif", "Computer Modern",
                           "Georgia", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          13,
    "axes.labelsize":     15,
    "axes.titlesize":     17,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    12,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

# Colors from line_plot_p.py MODEL_PALETTE — circles only for bubble chart
MODEL_COLORS = {
    "CXRMate":       "#88CCEE",
    "MedGemma":      "#228833",
    "MAIRA-2":       "#4477AA",
    "NV-Reason-CXR": "#CCBB44",
    "CheXagent-2":   "#AA3377",
    "RaDialog":      "#EE8866",
    "LIBRA-v1":      "#BBBBBB",
    "LIBRA-LLaVA":   "#222222",
}
_FALLBACK_COLOR = "#999999"

# CSV-filename-stem → display name (same keys as line_plot_p.py NAME_MAP)
NAME_MAP = {
    # PadChest-GR (with ::seed=3)
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default::seed=3": "RaDialog",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default::seed=3": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default::seed=3":  "LIBRA-LLaVA",
    "X-iZhang__libra-v1.0-7b_libra_default::seed=3":       "LIBRA-v1",
    "aehrc__cxrmate-rrg24_cxrmateed_default::seed=3":       "CXRMate",
    "microsoft__maira-2_maira2_default::seed=3":            "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default::seed=3":  "NV-Reason-CXR",
    # MIMIC-CXR (no seed suffix)
    "aehrc__cxrmate-rrg24_cxrmateed_default":               "CXRMate",
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default": "RaDialog",
    "microsoft__maira-2_maira2_default":                    "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default":           "NV-Reason-CXR",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default":           "LIBRA-LLaVA",
    # Truncated names
    "google__medgemma-1":  "MedGemma",
    "X-iZhang__libra-v1":  "LIBRA-v1",
}

DATASETS = [
    ("mimic-cxr-jpg", "MIMIC-CXR"),
    ("padchest-gr",   "PadChest-GR"),
]

# Background colors (same as line_plot_p.py)
DS_BGCOLOR = {
    "padchest-gr":   "#E8F0FE",
    "mimic-cxr-jpg": "#E8F5E9",
}

# Weights for weighted-mean CheXBERT F1 (same source as line_plot_p.py)
_WEIGHTS_JSON = Path(__file__).resolve().parent / "weights.json"
with open(_WEIGHTS_JSON) as _f:
    _RAW_WEIGHTS = json.load(_f)

_DS_KEY = {"padchest-gr": "padchest", "mimic-cxr-jpg": "mimic"}


def _weighted_f1(row: pd.Series, dataset: str, exp_type: str = "oco") -> float:
    raw = _RAW_WEIGHTS.get(_DS_KEY.get(dataset, dataset), {}).get(exp_type, {})
    wts = {k.replace("_", " "): float(v) for k, v in raw.items()}
    vals, ws = [], []
    for col in row.index:
        w = wts.get(col)
        if w is None:
            continue
        v = pd.to_numeric(row[col], errors="coerce")
        if np.isnan(v):
            continue
        vals.append(v)
        ws.append(w)
    if not vals:
        return float(np.nanmean(pd.to_numeric(row, errors="coerce").dropna()))
    return float(np.average(vals, weights=ws))


# ── Load data ────────────────────────────────────────────────────────────────

def load_compute_stats() -> dict[str, dict]:
    """Return {display_name: stats_dict}."""
    stats: dict[str, dict] = {}
    for p in sorted(STATS_DIR.glob("*.json")):
        with p.open() as f:
            d = json.load(f)
        name = d.get("display_name") or p.stem
        stats[name] = d
    return stats


def load_p0_metrics(dataset: str) -> dict[str, float]:
    """Return {display_name: weighted_mean_f1} from the p=0 CSV files."""
    results: dict[str, float] = {}
    p0_path = P0_DIR / dataset
    for csv_path in sorted(p0_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        if "CheXbert F1 score" not in df.index:
            continue
        name = NAME_MAP.get(csv_path.stem, csv_path.stem)
        f1 = _weighted_f1(df.loc["CheXbert F1 score"], dataset, "oco")
        if name not in results:
            results[name] = f1
    return results


# ── Plot ─────────────────────────────────────────────────────────────────────

def main() -> None:
    compute_stats = load_compute_stats()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=False)
    fig.subplots_adjust(wspace=0.25, bottom=0.18)

    # Bubble size: area proportional to n_params_total_B
    SIZE_SCALE = 380.0  # multiply params_B by this to get marker area

    legend_handles = []
    legend_labels  = []
    seen_models    = set()

    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        ax.set_facecolor(DS_BGCOLOR[ds_key])

        p0_metrics   = load_p0_metrics(ds_key)
        all_models   = sorted(p0_metrics.keys())

        for model in all_models:
            y = p0_metrics[model]
            stats = compute_stats.get(model, {})
            gflops = stats.get("gflops")
            if gflops is None:
                continue
            x = gflops / 1000.0  # display in units of 10³ GFLOPs
            params_b = stats.get("n_params_total_B", 1.0)
            color = MODEL_COLORS.get(model, _FALLBACK_COLOR)

            # Bubble: area ∝ params_B
            area = params_b * SIZE_SCALE

            # Main bubble (semi-transparent)
            ax.scatter(
                x, y,
                s=area,
                color=color, marker="o",
                edgecolors="black", linewidths=0.8,
                alpha=0.55, zorder=4,
            )
            # Center dot
            ax.scatter(
                x, y,
                s=7, color="black", marker="o",
                edgecolors="none", zorder=5,
            )

            if model not in seen_models:
                proxy = plt.scatter(
                    [], [], s=90,
                    color=color, marker="o",
                    edgecolors="black", linewidths=0.8,
                    label=model,
                )
                legend_handles.append(proxy)
                legend_labels.append(model)
                seen_models.add(model)

        ymin, ymax = (0.54, 0.75) if ds_key == "padchest-gr" else (0.625, 0.775)
        ax.set_xlabel(r"GFLOPs ($\times 10^3$)", labelpad=4)
        ax.set_ylabel(r"$\mu\text{-}F1$", labelpad=4, fontsize=18)
        ax.set_xlim(-3, 25)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(round(v * 100))}%")
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        ax.tick_params(colors="#333333")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7,
                color="#888888", zorder=0)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.7,
                color="#aaaaaa", zorder=0)
        ax.set_axisbelow(True)

    # ── Legends (colour + bubble size, placed side by side) ──────────────────
    param_sizes = [0.1, 3.0, 7.0]

    leg1 = fig.legend(
        legend_handles, legend_labels,
        loc="upper left", ncol=4,
        bbox_to_anchor=(0.08, 0.09),
        frameon=True, edgecolor="#dddddd",
        facecolor="white", fancybox=False,
        columnspacing=1.0, handlelength=1.0,
        handletextpad=0.5, borderpad=0.7,
        labelspacing=0.75,
        fontsize=12,
    )
    fig.add_artist(leg1)

    # Bubble-size legend: placed to the right of the colour legend, same row.
    # Use scatter proxies so the marker area (s) matches the plot exactly, then
    # scale all three down by the same factor so the largest fits comfortably.
    # No scaling: s = p * SIZE_SCALE exactly as in the plot.
    # handlelength wide enough to contain the 7 B bubble (d ≈ 58 pt → radius ≈ 29 pt,
    # handle half-width = 5.0 * 13 / 2 = 32.5 pt > 29 pt).
    size_handles = [
        plt.scatter([], [], s=p * SIZE_SCALE,
                    color="#888888", alpha=0.55,
                    edgecolors="black", linewidths=0.8,
                    label=f"{p:g} B")
        for p in param_sizes
    ]

    leg2 = fig.legend(
        size_handles, [f"{p:g} B" for p in param_sizes],
        loc="upper left",
        bbox_to_anchor=(0.57, 0.09),
        frameon=True, edgecolor="#dddddd",
        facecolor="white", fancybox=False,
        columnspacing=1.5, handlelength=5.0,
        handletextpad=0.8, borderpad=0.9,
        labelspacing=0.5,
        fontsize=13,
        ncol=3,
    )

    fig.savefig(OUTPUT_DIR / "gflops_bubble.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUTPUT_DIR / "gflops_bubble.png", bbox_inches="tight", pad_inches=0.12)
    print(f"Saved → {OUTPUT_DIR / 'gflops_bubble.pdf'}")


if __name__ == "__main__":
    main()
