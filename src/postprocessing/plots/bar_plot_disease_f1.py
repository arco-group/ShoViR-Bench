"""
Bar plot: per-disease CheXBERT F1 averaged across models, with 95% CI.

Layout: 1 row × 2 columns (PadChest-GR | MIMIC-CXR), shared y-axis.
Experiment colors are identical across both panels.
Background identifies the dataset (pastel blue = PadChest, green = MIMIC).

Output: results/plots/disease_f1.{pdf,png}
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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

ALL_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices',
]
DISEASE_LABELS = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Enlarged Cardiomediastinum': 'Enl. Card.',
    'Fracture': 'Fracture',
    'Lung Lesion': 'Lung Lesion',
    'Lung Opacity': 'Lung Opacity',
    'Pleural Effusion': 'Pleural Effusion',
    'Pleural Other': 'Pleural Other',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Support Devices': 'Support Devices',
}

SKIP_STEMS = {'microsoft__maira-2_maira2_default::seed=3_grounded'}

DATASETS = [
    ('mimic-cxr-jpg', 'MIMIC-CXR'),
    ('padchest-gr',   'PadChest-GR'),
]
EXPERIMENTS = [
    ('Baseline', 'oco',  'p00'),
    ('OCO',      'oco',  'p100'),
    ('DOCO',     'doco', 'p100'),
    ('RO',       'ro',   'p100'),
]
# Same colors for both panels
COLORS = ['#CCBB44', '#EE6677', '#4477AA', '#228833']

DS_BGCOLOR = {
    'padchest-gr':   '#E8F0FE',
    'mimic-cxr-jpg': '#E8F5E9',
}
DS_LABEL = {
    'padchest-gr':   'PadChest-GR',
    'mimic-cxr-jpg': 'MIMIC-CXR',
}


# ── DATA LOADING ────────────────────────────────────────────────────
def load_disease_f1(exp_type: str, p_str: str, dataset: str) -> pd.DataFrame:
    exp_dir = RESULTS_DIR / exp_type / p_str / dataset
    rows = []
    for csv_path in sorted(exp_dir.glob('*.csv')):
        if csv_path.stem in SKIP_STEMS:
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        if 'CheXbert F1 score' not in df.index:
            continue
        rows.append(df.loc['CheXbert F1 score'].rename(csv_path.stem))
    if not rows:
        return pd.DataFrame(columns=ALL_DISEASES)
    return pd.DataFrame(rows).reindex(columns=ALL_DISEASES)


def mean_and_ci(values: np.ndarray, confidence: float = 0.95):
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(valid[0]), np.nan
    return float(np.mean(valid)), float(stats.t.ppf((1+confidence)/2, df=n-1) * stats.sem(valid))


def build_stats(dataset: str) -> dict:
    # Always use all 13 diseases — missing ones will produce NaN bars (empty)
    diseases = ALL_DISEASES
    stats_dict = {}
    for label, exp_type, p_str in EXPERIMENTS:
        df = load_disease_f1(exp_type, p_str, dataset)
        means, cis = [], []
        for d in diseases:
            vals = df[d].values.astype(float) if d in df.columns else np.array([np.nan])
            m, h = mean_and_ci(vals)
            means.append(m); cis.append(h)
        stats_dict[label] = {'means': means, 'cis': cis}
    return {'diseases': diseases, 'stats': stats_dict,
            'n_models': len(load_disease_f1('oco', 'p00', dataset))}


dataset_stats = {ds_key: build_stats(ds_key) for ds_key, _ in DATASETS}


# ── PLOT ─────────────────────────────────────────────────────────────
n_exp = len(EXPERIMENTS)
bar_w = 0.17

fig, axes = plt.subplots(len(DATASETS), 1,
                         figsize=(9.5, 6.5),
                         sharey=True, sharex=True)
fig.subplots_adjust(hspace=0.08)

x = np.arange(len(ALL_DISEASES))  # shared x positions for all 13 diseases

for ax_idx, (ds_key, ds_label) in enumerate(DATASETS):
    ax       = axes[ax_idx]
    ds       = dataset_stats[ds_key]
    diseases = ds['diseases']  # always ALL_DISEASES
    nd       = len(diseases)

    ax.set_facecolor(DS_BGCOLOR[ds_key])

    for i, (label, _, _) in enumerate(EXPERIMENTS):
        means  = ds['stats'][label]['means']
        cis    = ds['stats'][label]['cis']
        offset = (i - (n_exp - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w,
               label=label if ax_idx == 0 else None,
               color=COLORS[i],
               edgecolor='white', linewidth=0.4, alpha=0.88, zorder=3)

    ax.set_xticks(x)
    # x-tick labels only on bottom row; rotated 30° for readability
    if ax_idx == len(DATASETS) - 1:
        ax.set_xticklabels([DISEASE_LABELS[d] for d in diseases],
                           rotation=30, ha='right', fontsize=11)
    else:
        ax.tick_params(labelbottom=False)

    ax.set_ylabel(r'F1-Score', fontsize=14)

    all_means = [v for s in ds['stats'].values() for v in s['means'] if not np.isnan(v)]
    ymax = min(1.0, (max(all_means) + 0.12) if all_means else 1.0)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 0.01, 0.1))
    ax.grid(axis='y', linestyle=':', linewidth=0.35, alpha=0.6, color='#888888', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#333333')

# Single legend below, collected from top panel
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.08),
           frameon=True, facecolor='white', edgecolor='#dddddd',
           fancybox=False, columnspacing=1.5,
           handlelength=1.3, fontsize=10)

fig.savefig(OUTPUT_DIR / 'disease_f1.pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(OUTPUT_DIR / 'disease_f1.png', bbox_inches='tight', pad_inches=0.1)
print(f"Saved → {OUTPUT_DIR / 'disease_f1.pdf'}")
