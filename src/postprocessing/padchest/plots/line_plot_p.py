import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[4] / 'results'
OUTPUT_DIR  = RESULTS_DIR / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS   = ['oco', 'doco', 'ro']
EXP_TITLES    = {'oco': '(a) OCO', 'doco': '(b) DOCO', 'ro': '(c) RO'}
P_VALUES      = [0, 20, 40, 60, 80, 100]

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern', 'Georgia', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 8.5,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
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

# Tol's vibrant scheme (colorblind-safe)
palette = [
    '#EE6677',  # rose
    '#228833',  # green
    '#4477AA',  # blue
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#EE8866',  # orange
    '#BBBBBB',  # grey
    '#222222',  # black
]
markers    = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', 'd']
linestyles = ['-', '-', '-', '-', '-', '--', '--', '--', '--']

# ── READ DATA ────────────────────────────────────────────────────────
def load_experiment(experiment: str) -> dict:
    """Return model_short -> {p: mean_f1} for one experiment type."""
    data = defaultdict(dict)
    for p in P_VALUES:
        p_str    = f'p{p:02d}' if p < 100 else 'p100'
        csv_path = RESULTS_DIR / experiment / p_str / 'results.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, index_col=0)
        for idx, row in df.iterrows():
            short    = name_map.get(idx, idx)
            mean_f1  = float(np.nanmean(row.values.astype(float)))
            data[short][p] = mean_f1
    return data

data = {exp: load_experiment(exp) for exp in EXPERIMENTS}

# Consistent model order: sort by mean F1 across all experiments, descending
def overall_mean(model):
    vals = [f1 for exp in EXPERIMENTS
            for f1 in data[exp].get(model, {}).values()]
    return np.mean(vals) if vals else 0.0

all_models   = list(data[EXPERIMENTS[0]].keys())
model_order  = sorted(all_models, key=overall_mean, reverse=True)

# ── PLOT ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
fig.subplots_adjust(wspace=0.08)

for ax_idx, (ax, exp) in enumerate(zip(axes, EXPERIMENTS)):
    for m_idx, model in enumerate(model_order):
        model_data = data[exp].get(model, {})
        xs = [p for p in P_VALUES if p in model_data]
        ys = [model_data[p] for p in xs]

        ax.plot(xs, ys,
                color=palette[m_idx % len(palette)],
                marker=markers[m_idx % len(markers)],
                markersize=4.5,
                markeredgecolor='white',
                markeredgewidth=0.5,
                linewidth=1.4,
                linestyle=linestyles[m_idx % len(linestyles)],
                label=model if ax_idx == 0 else None,
                alpha=0.92,
                zorder=3 + m_idx * 0.1)

    ax.set_title(EXP_TITLES[exp], fontweight='bold', pad=7, fontsize=10.5)
    ax.set_xlabel('Perturbation ratio $p$ (%)', labelpad=4)
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(p) for p in P_VALUES])
    ax.set_xlim(-5, 105)

    ax.grid(True, axis='y', linestyle=':', linewidth=0.35, alpha=0.6,
            color='#888888', zorder=0)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.2, alpha=0.3,
            color='#aaaaaa', zorder=0)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#333333')

    if ax_idx == 0:
        ax.set_ylabel('Mean CheXBERT F1', labelpad=5)

# Auto y-limits with a small margin
all_vals = [f1 for exp in EXPERIMENTS
            for m in data[exp].values()
            for f1 in m.values()]
ymin = max(0.0, min(all_vals) - 0.03)
ymax = min(1.0, max(all_vals) + 0.03)
axes[0].set_ylim(ymin, ymax)

# Legend below
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=4,
           bbox_to_anchor=(0.5, -0.22),
           frameon=True,
           edgecolor='#dddddd',
           facecolor='white',
           fancybox=False,
           columnspacing=1.0,
           handlelength=2.2,
           handletextpad=0.5,
           borderpad=0.5,
           fontsize=8.2)

fig.savefig(OUTPUT_DIR / 'lineplot.pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(OUTPUT_DIR / 'lineplot.png', bbox_inches='tight', pad_inches=0.1)
print(f"Saved to {OUTPUT_DIR}")
