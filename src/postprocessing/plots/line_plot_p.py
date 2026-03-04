"""
Line plot: per-model mean CheXBERT F1 vs perturbation level p∈{0,20,40,60,80,100}.

Layout: 2 rows (datasets) × 3 columns (OCO, DOCO, RO).
Shared x-axis across all panels; shared y-axis within each row.
Each model has a fixed color and marker that is consistent across both datasets.

Output: results/plots/lineplot.{pdf,png}
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[3] / 'results'
OUTPUT_DIR  = RESULTS_DIR / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Didot', 'GFS Didot', 'CMU Serif', 'Computer Modern', 'Georgia', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
})

NAME_MAP = {
    # PadChest-GR — full names with ::seed=3
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default::seed=3": "RaDialog",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default::seed=3": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default::seed=3": "LLaVA-Rad",
    "X-iZhang__libra-v1.0-7b_libra_default::seed=3": "LIBRA-v1",
    "aehrc__cxrmate-rrg24_cxrmateed_default::seed=3": "CXRMate",
    "microsoft__maira-2_maira2_default::seed=3": "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default::seed=3": "NV-Reason-CXR",
    # MIMIC-CXR
    "aehrc__cxrmate-rrg24_cxrmateed_default": "CXRMate",
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default": "RaDialog",
    "microsoft__maira-2_maira2_default": "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default": "NV-Reason-CXR",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default": "LLaVA-Rad ",
    # Truncated filenames
    "google__medgemma-1": "MedGemma",
    "X-iZhang__libra-v1": "LIBRA-v1",
}

# Fixed color + marker + linestyle per model — identical across both datasets
MODEL_PALETTE = {
    'CXRMate':       ('#EE6677', 'o', '-'),
    'MedGemma':      ('#228833', 's', '-'),
    'MAIRA-2':       ('#4477AA', '^', '-'),
    'NV-Reason-CXR': ('#CCBB44', 'D', '-'),
    'CheXagent-2':   ('#AA3377', 'P', '-'),
    'RaDialog':      ('#EE8866', 'X', '--'),
    'LIBRA-v1':      ('#BBBBBB', 'h', '--'),
    'LLaVA-Rad ':   ('#222222', 'd', '--'),
}
_FALLBACK = ('#999999', 'o', ':')

# Pastel background per dataset
DS_BGCOLOR = {
    'padchest-gr':   '#E8F0FE',  # pastel blue
    'mimic-cxr-jpg': '#E8F5E9',  # pastel green
}
DS_LABEL = {
    'padchest-gr':   'PadChest-GR',
    'mimic-cxr-jpg': 'MIMIC-CXR',
}

SKIP_STEMS  = {'microsoft__maira-2_maira2_default::seed=3_grounded'}

# ── WEIGHTED MEAN HELPERS ────────────────────────────────────────────
_WEIGHTS_JSON = Path(__file__).resolve().parent / 'weights.json'
with open(_WEIGHTS_JSON) as _f:
    _RAW_WEIGHTS = json.load(_f)

_DS_KEY = {'padchest-gr': 'padchest', 'mimic-cxr-jpg': 'mimic'}


def _weighted_f1(row: pd.Series, dataset: str, exp_type: str) -> float:
    """
    Weighted mean of CheXBERT F1 across disease columns.
    Weights are the positive-sample counts from weights.json for the
    given dataset and experiment type.  Classes absent from the weights
    file are ignored; falls back to nanmean if no weights match.
    """
    raw = _RAW_WEIGHTS.get(_DS_KEY.get(dataset, dataset), {}).get(exp_type, {})
    wts = {k.replace('_', ' '): float(v) for k, v in raw.items()}
    vals, ws = [], []
    for col in row.index:
        w = wts.get(col)
        if w is None:
            continue
        v = pd.to_numeric(row[col], errors='coerce')
        if np.isnan(v):
            continue
        vals.append(v)
        ws.append(w)
    if not vals:
        return float(np.nanmean(row.values.astype(float)))
    return float(np.average(vals, weights=ws))

DATASETS    = [('mimic-cxr-jpg', 'MIMIC-CXR'), ('padchest-gr', 'PadChest-GR')]
EXPERIMENTS = ['ro', 'oco', 'doco']
EXP_TITLES  = {'oco': 'OCO', 'doco': 'DOCO', 'ro': 'RO'}
P_VALUES    = [0, 20, 40, 60, 80, 100]


# ── DATA LOADING ────────────────────────────────────────────────────
def load_experiment(experiment: str, dataset: str) -> dict:
    """Return {model_short: {p: mean_f1}} for one experiment + dataset."""
    data = defaultdict(dict)
    for p in P_VALUES:
        p_str   = f'p{p:02d}' if p < 100 else 'p100'
        exp_dir = RESULTS_DIR / experiment / p_str / dataset
        for csv_path in sorted(exp_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_path, index_col=0)
            except Exception:
                continue
            if 'CheXbert F1 score' not in df.index:
                continue
            if csv_path.stem in SKIP_STEMS:
                continue
            short   = NAME_MAP.get(csv_path.stem, csv_path.stem)
            mean_f1 = _weighted_f1(df.loc['CheXbert F1 score'], dataset, experiment)
            if p not in data[short]:
                data[short][p] = mean_f1
    return data


all_data = {
    ds_key: {exp: load_experiment(exp, ds_key) for exp in EXPERIMENTS}
    for ds_key, _ in DATASETS
}


# ── PLOT ─────────────────────────────────────────────────────────────
n_rows = len(DATASETS)
n_cols = len(EXPERIMENTS)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(12.0, 4.0 * n_rows),
                         sharex=True, sharey='row')
fig.subplots_adjust(wspace=0.04, hspace=0.12)

for row_idx, (ds_key, ds_label) in enumerate(DATASETS):
    ds_data = all_data[ds_key]

    all_models = list({m for exp in EXPERIMENTS for m in ds_data[exp]})
    model_order = sorted(
        all_models,
        key=lambda m: np.mean([f1 for exp in EXPERIMENTS
                                for f1 in ds_data[exp].get(m, {}).values()]) if any(
            ds_data[exp].get(m) for exp in EXPERIMENTS) else 0.0,
        reverse=True,
    )

    row_vals = [f1 for exp in EXPERIMENTS
                for m in ds_data[exp].values()
                for f1 in m.values()]
    ymin = max(0.0, min(row_vals) - 0.03)
    ymax = min(1.0, max(row_vals) + 0.03)

    for col_idx, exp in enumerate(EXPERIMENTS):
        ax = axes[row_idx, col_idx]
        ax.set_facecolor(DS_BGCOLOR[ds_key])

        for model in model_order:
            model_data = ds_data[exp].get(model, {})
            xs = [p for p in P_VALUES if p in model_data]
            ys = [model_data[p] for p in xs]
            col, mkr, ls = MODEL_PALETTE.get(model, _FALLBACK)
            ax.plot(xs, ys,
                    color=col, marker=mkr, markersize=4,
                    markeredgecolor='white', markeredgewidth=0.5,
                    linewidth=1.3, linestyle=ls,
                    label=model if col_idx == 0 else None,
                    alpha=0.92, zorder=3)

        # Column title on first row only
        if row_idx == 0:
            ax.set_title(EXP_TITLES[exp], fontweight='bold', pad=5)
        # x-label on bottom row only (top row shares x, labels auto-hidden)
        if row_idx == n_rows - 1:
            ax.set_xlabel('Perturbation ratio $p$ (%)', labelpad=3)
        # y-label on leftmost column only
        if col_idx == 0:
            ax.set_ylabel(r'$\mu\text{-}F1$', labelpad=4, fontsize=18)
        ax.set_xlim(-5, 105)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(P_VALUES)
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

# Single legend below the figure, collecting handles from both rows
all_handles, all_labels = [], []
seen = set()
for row_idx in range(n_rows):
    h, l = axes[row_idx, 0].get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in seen:
            all_handles.append(handle)
            all_labels.append(label)
            seen.add(label)

fig.legend(all_handles, all_labels,
           loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.08),
           frameon=True, edgecolor='#dddddd',
           facecolor='white', fancybox=False,
           columnspacing=0.9, handlelength=2.0,
           handletextpad=0.4, borderpad=0.5,
           fontsize=12)

fig.savefig(OUTPUT_DIR / 'lineplot.pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(OUTPUT_DIR / 'lineplot.png', bbox_inches='tight', pad_inches=0.1)
print(f"Saved → {OUTPUT_DIR / 'lineplot.pdf'}")
