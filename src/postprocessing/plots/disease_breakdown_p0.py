"""
Grouped dot-plot: per-disease CheXBERT F1 at p=0 (baseline).

Layout: 2 rows (datasets: PadChest-GR / MIMIC-CXR).
Each panel shows one dot per model per disease, colored by model
using the same MODEL_PALETTE as line_plot_p.py.
Diseases are on the x-axis; F1 on the y-axis.

Data source:
  results/oco/p00/<dataset>/<model>.csv

Output:
  results/plots/disease_breakdown_p0.{pdf,png}
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results"
P0_DIR      = RESULTS_DIR / "oco" / "p00"
OUTPUT_DIR  = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Didot", "GFS Didot", "CMU Serif", "Computer Modern",
                           "Georgia", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          11,
    "axes.labelsize":     13,
    "axes.titlesize":     14,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
})

# Identical palette to line_plot_p.py
MODEL_PALETTE = {
    "CXRMate":       ("#EE6677", "o"),
    "MedGemma":      ("#228833", "s"),
    "MAIRA-2":       ("#4477AA", "^"),
    "NV-Reason-CXR": ("#CCBB44", "D"),
    "CheXagent-2":   ("#AA3377", "P"),
    "RaDialog":      ("#EE8866", "X"),
    "LIBRA-v1":      ("#BBBBBB", "h"),
    "LIBRA-LLaVA":   ("#222222", "d"),
}
_FALLBACK_COLOR  = "#999999"
_FALLBACK_MARKER = "o"

# CSV-stem → display name (same as line_plot_p.py)
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
    ("padchest-gr",   "PadChest-GR"),
    ("mimic-cxr-jpg", "MIMIC-CXR"),
]

DS_BGCOLOR = {
    "padchest-gr":   "#E8F0FE",
    "mimic-cxr-jpg": "#E8F5E9",
}

# Desired disease order per dataset (superset; missing ones skipped)
DISEASE_ORDER = {
    "padchest-gr": [
        "Atelectasis", "Cardiomegaly", "Fracture",
        "Lung Lesion", "Lung Opacity",
        "Pleural Effusion", "Pleural Other", "Support Devices",
    ],
    "mimic-cxr-jpg": [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices",
    ],
}


# ── Load data ────────────────────────────────────────────────────────────────

def load_p0_per_disease(dataset: str) -> dict[str, dict[str, float]]:
    """Return {model_name: {disease: f1}} from the p=0 CSVs."""
    results: dict[str, dict[str, float]] = {}
    p0_path = P0_DIR / dataset
    for csv_path in sorted(p0_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        if "CheXbert F1 score" not in df.index:
            continue
        name = NAME_MAP.get(csv_path.stem, csv_path.stem)
        if name in results:
            continue  # take first occurrence
        row = df.loc["CheXbert F1 score"]
        disease_f1 = {}
        for col in row.index:
            v = pd.to_numeric(row[col], errors="coerce")
            if not np.isnan(v):
                disease_f1[col] = float(v)
        results[name] = disease_f1
    return results


# ── Plot ─────────────────────────────────────────────────────────────────────

def main() -> None:
    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 9),
    )
    fig.subplots_adjust(hspace=0.35)

    # Jitter spread per disease group so overlapping dots are visible
    # Models within each disease are spread along x by a small offset
    DOT_SIZE    = 55
    ALPHA       = 0.88

    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        ax.set_facecolor(DS_BGCOLOR[ds_key])

        per_disease = load_p0_per_disease(ds_key)
        all_models  = sorted(per_disease.keys())

        # Determine disease columns present in data
        pref_order = DISEASE_ORDER.get(ds_key, [])
        present_diseases_set = {d for md in per_disease.values() for d in md}
        diseases = [d for d in pref_order if d in present_diseases_set]
        # Append any unexpected columns not in the preferred order
        for d in sorted(present_diseases_set):
            if d not in diseases:
                diseases.append(d)

        n_models   = len(all_models)
        n_diseases = len(diseases)
        x_ticks    = np.arange(n_diseases)

        # Spread offsets for each model within a disease tick
        jitter_half = 0.35
        offsets = np.linspace(-jitter_half, jitter_half, n_models) if n_models > 1 else [0.0]

        # Draw faint vertical stripe per disease
        for xi in x_ticks:
            ax.axvspan(xi - 0.5, xi + 0.5,
                       color=("#f0f0f0" if xi % 2 == 0 else "#e4e4e4"),
                       alpha=0.4, zorder=0)

        for m_idx, model in enumerate(all_models):
            color, mkr = MODEL_PALETTE.get(model, (_FALLBACK_COLOR, _FALLBACK_MARKER))
            disease_f1 = per_disease[model]
            xs, ys = [], []
            for d_idx, disease in enumerate(diseases):
                if disease in disease_f1:
                    xs.append(x_ticks[d_idx] + offsets[m_idx])
                    ys.append(disease_f1[disease])
            if xs:
                ax.scatter(xs, ys,
                           s=DOT_SIZE, color=color, marker=mkr,
                           edgecolors="white", linewidths=0.5,
                           alpha=ALPHA, zorder=4, label=model)

        # Per-disease mean line
        for d_idx, disease in enumerate(diseases):
            vals = [per_disease[m][disease] for m in all_models if disease in per_disease[m]]
            if vals:
                ax.plot(
                    [x_ticks[d_idx] - 0.42, x_ticks[d_idx] + 0.42],
                    [np.mean(vals)] * 2,
                    color="#555555", lw=1.1, ls="--", alpha=0.55, zorder=3,
                )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(diseases, rotation=30, ha="right", fontsize=9.5)
        ax.set_xlim(-0.6, n_diseases - 0.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("CheXBERT F1 score  ($p=0$)", labelpad=4)
        ax.set_title(ds_label, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        ax.tick_params(colors="#333333")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.35, alpha=0.55,
                color="#888888", zorder=1)
        ax.set_axisbelow(True)

    # ── Shared legend ─────────────────────────────────────────────────────────
    # Build legend from first axis (all models are the same across datasets)
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)

    # Add dashed line proxy for "per-disease mean"
    mean_proxy = mlines.Line2D([], [], color="#555555", lw=1.1, ls="--",
                               alpha=0.7, label="Per-disease mean")
    h2.append(mean_proxy); l2.append("Per-disease mean")

    fig.legend(
        h2, l2,
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.06),
        frameon=True, edgecolor="#dddddd",
        facecolor="white", fancybox=False,
        columnspacing=0.9, handlelength=1.2,
        handletextpad=0.4, borderpad=0.5,
        fontsize=10,
    )

    fig.savefig(OUTPUT_DIR / "disease_breakdown_p0.pdf",
                bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUTPUT_DIR / "disease_breakdown_p0.png",
                bbox_inches="tight", pad_inches=0.12)
    print(f"Saved → {OUTPUT_DIR / 'disease_breakdown_p0.pdf'}")


if __name__ == "__main__":
    main()
