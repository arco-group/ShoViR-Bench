import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[4] / 'results'
OUTPUT_DIR = RESULTS_DIR / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# Modern, formal sans-serif style
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

name_map = {
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default::seed=3": "RaDialog",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default::seed=3": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default::seed=3": "LIBRA-LLaVA",
    "X-iZhang__libra-v1.0-7b_libra_default::seed=3": "LIBRA-v1",
    "aehrc__cxrmate-rrg24_cxrmateed_default::seed=3": "CXRMate",
    "google__medgemma-1.5-4b-it_medgemma_default::seed=3": "MedGemma",
    "microsoft__maira-2_maira2_default::seed=3": "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default::seed=3": "NV-Reason-CXR",
}

# ── READ DATA ────────────────────────────────────────────────────────
def load_mean_f1(csv_path: Path) -> dict:
    """Load per-class F1 CSV and return model -> nanmean across conditions."""
    df = pd.read_csv(csv_path, index_col=0)
    return {
        idx: float(np.nanmean(row.values.astype(float)))
        for idx, row in df.iterrows()
    }

# Baseline = oco/p00 (unperturbed images, same CheXBERT per-class format)
baseline_data = load_mean_f1(RESULTS_DIR / 'oco' / 'p00' / 'results.csv')
oco_data      = load_mean_f1(RESULTS_DIR / 'oco'  / 'p100' / 'results.csv')
doco_data     = load_mean_f1(RESULTS_DIR / 'doco' / 'p100' / 'results.csv')
ro_data       = load_mean_f1(RESULTS_DIR / 'ro'   / 'p100' / 'results.csv')

all_keys = list(baseline_data.keys())
models   = [name_map.get(k, k) for k in all_keys]
baseline = [baseline_data[k]            for k in all_keys]
oco      = [oco_data.get(k, np.nan)     for k in all_keys]
doco     = [doco_data.get(k, np.nan)    for k in all_keys]
ro       = [ro_data.get(k, np.nan)      for k in all_keys]

# Add average
models.append("Average")
baseline.append(np.nanmean(baseline))
oco.append(np.nanmean(oco))
doco.append(np.nanmean(doco))
ro.append(np.nanmean(ro))

# ── PLOT ────────────────────────────────────────────────────────────
colors      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
experiments = ['Baseline', 'OCO', 'DOCO', 'RO']

n_total  = len(models)
split    = 5                          # models per row in row 1
row1_idx = list(range(split))
row2_idx = list(range(split, n_total))

bar_width = 0.17
fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.4))

all_vals = [baseline, oco, doco, ro]

for ax_idx, (ax, indices) in enumerate(zip(axes, [row1_idx, row2_idx])):
    n = len(indices)
    x = np.arange(n)

    for i, (exp, color) in enumerate(zip(experiments, colors)):
        vals   = [all_vals[i][j] for j in indices]
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, vals, bar_width,
               label=exp if ax_idx == 0 else None,
               color=color, edgecolor='white', linewidth=0.5,
               zorder=3, alpha=0.92)

    ax.set_xticks(x)
    labels = [models[j] for j in indices]
    ax.set_xticklabels(labels, rotation=0, ha='center', fontweight='medium')
    ax.set_ylabel('Mean CheXBERT F1')
    ax.set_ylim(0, 0.92)
    ax.set_yticks(np.arange(0, 1.0, 0.1))
    ax.grid(axis='y', linestyle='-', linewidth=0.25, alpha=0.4, color='#999999', zorder=0)
    ax.set_axisbelow(True)

    # Subtle dashed separator before Average (last entry in row 2)
    if ax_idx == 1:
        sep_x = n - 1.55
        ax.axvline(sep_x, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)

# Legend above plot
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc='upper center',
    ncol=4,
    bbox_to_anchor=(0.5, 1.025),
    frameon=False,
    columnspacing=1.8,
    handlelength=1.5,
    fontsize=9.5,
)

fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=2.2)

fig.savefig(OUTPUT_DIR / 'barplot.pdf', bbox_inches='tight', pad_inches=0.12)
fig.savefig(OUTPUT_DIR / 'barplot.png', bbox_inches='tight', pad_inches=0.12)
print(f"Saved to {OUTPUT_DIR}")
