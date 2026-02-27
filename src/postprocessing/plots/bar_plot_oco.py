"""
Bar plot: per-model mean CheXBERT F1 under Baseline, OCO, DOCO, RO (p=100).

One panel per dataset (PadChest-GR top, MIMIC-CXR bottom).
Baseline = oco/p00 (same perturbation pipeline, 0% occlusion).
Data loaded from per-file CSVs (one CSV per model per experiment level).

Output: results/plots/barplot.{pdf,png}
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

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
    "X-iZhang__libra-llava-rad_llavarad_default::seed=3": "LIBRA-LLaVA",
    "X-iZhang__libra-v1.0-7b_libra_default::seed=3": "LIBRA-v1",
    "aehrc__cxrmate-rrg24_cxrmateed_default::seed=3": "CXRMate",
    "microsoft__maira-2_maira2_default::seed=3": "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default::seed=3": "NV-Reason-CXR",
    # MIMIC-CXR — names without ::seed or with different suffixes
    "aehrc__cxrmate-rrg24_cxrmateed_default": "CXRMate",
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default": "RaDialog",
    "microsoft__maira-2_maira2_default": "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default": "NV-Reason-CXR",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default": "LIBRA-LLaVA",
    # Truncated filenames (OS filename-length limit)
    "google__medgemma-1": "MedGemma",
    "X-iZhang__libra-v1": "LIBRA-v1",
}

# Pastel background per dataset (identifies dataset without text title)
DS_BGCOLOR = {
    'padchest-gr':   '#E8F0FE',  # pastel blue
    'mimic-cxr-jpg': '#E8F5E9',  # pastel green
}
DS_LABEL = {
    'padchest-gr':   'PadChest-GR',
    'mimic-cxr-jpg': 'MIMIC-CXR',
}

SKIP_STEMS = {'microsoft__maira-2_maira2_default::seed=3_grounded'}

DATASETS = [
    ('padchest-gr',   'PadChest-GR'),
    ('mimic-cxr-jpg', 'MIMIC-CXR'),
]
EXPERIMENTS = [
    ('Baseline', 'oco',  'p00'),
    ('OCO',      'oco',  'p100'),
    ('DOCO',     'doco', 'p100'),
    ('RO',       'ro',   'p100'),
]
COLORS = ['#CCBB44', '#EE6677', '#4477AA', '#228833']


# ── DATA LOADING ────────────────────────────────────────────────────
def load_mean_f1_dir(exp_type: str, p_str: str, dataset: str) -> dict[str, float]:
    """Scan per-file CSVs → {model_short: mean CheXBERT F1 across diseases}."""
    exp_dir = RESULTS_DIR / exp_type / p_str / dataset
    out = {}
    for csv_path in sorted(exp_dir.glob('*.csv')):
        if csv_path.stem in SKIP_STEMS:
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        if 'CheXbert F1 score' not in df.index:
            continue
        short = NAME_MAP.get(csv_path.stem, csv_path.stem)
        if short not in out:
            out[short] = float(np.nanmean(df.loc['CheXbert F1 score'].values.astype(float)))
    return out


# ── BUILD DATA PER DATASET ──────────────────────────────────────────
dataset_data = {}
for ds_key, ds_label in DATASETS:
    exp_results = {lbl: load_mean_f1_dir(et, ps, ds_key)
                   for lbl, et, ps in EXPERIMENTS}
    models = sorted(
        exp_results['Baseline'].keys(),
        key=lambda m: exp_results['Baseline'].get(m, 0),
        reverse=True,
    )
    models.append('Average')
    for lbl in exp_results:
        vals = [exp_results[lbl].get(m, np.nan) for m in models[:-1]]
        exp_results[lbl]['Average'] = float(np.nanmean(vals))
    dataset_data[ds_key] = {'models': models, 'exp': exp_results}

# ── GLOBAL MODEL ORDER (consistent across both panels) ──────────────
pc_models    = [m for m in dataset_data['padchest-gr']['models']   if m != 'Average']
mimic_extra  = [m for m in dataset_data['mimic-cxr-jpg']['models'] if m != 'Average' and m not in pc_models]
global_models = pc_models + mimic_extra

for ds_key, _ in DATASETS:
    for lbl in dataset_data[ds_key]['exp']:
        vals = [dataset_data[ds_key]['exp'][lbl].get(m, np.nan) for m in global_models]
        dataset_data[ds_key]['exp'][lbl]['Average'] = float(np.nanmean(vals))
    dataset_data[ds_key]['models'] = global_models + ['Average']


# ── PLOT ────────────────────────────────────────────────────────────
bar_width = 0.17
fig, axes = plt.subplots(len(DATASETS), 1,
                         figsize=(10.5, 2.8 * len(DATASETS)),
                         sharey=True)

for ax_idx, (ds_key, _) in enumerate(DATASETS):
    ax      = axes[ax_idx]
    dd      = dataset_data[ds_key]
    models  = dd['models']
    n       = len(models)
    x       = np.arange(n)

    ax.set_facecolor(DS_BGCOLOR[ds_key])

    all_vals = [[dd['exp'][lbl].get(m, np.nan) for m in models]
                for lbl, _, _ in EXPERIMENTS]

    for i, (exp_label, _, _) in enumerate(EXPERIMENTS):
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, all_vals[i], bar_width,
               label=exp_label if ax_idx == 0 else None,
               color=COLORS[i], edgecolor='white', linewidth=0.5,
               zorder=3, alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontweight='medium')
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis='y', linestyle='-', linewidth=0.25, alpha=0.4, color='#999999', zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(n - 1.55, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)

# Shared y-label on left side
fig.text(0.02, 0.5, 'Mean CheXBERT F1', va='center', rotation='vertical', fontsize=13)

fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc='upper center', ncol=4,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False, columnspacing=1.8,
    handlelength=1.5, fontsize=10,
)
fig.tight_layout(rect=[0.04, 0, 1, 0.97], h_pad=0.6)
fig.savefig(OUTPUT_DIR / 'barplot.pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(OUTPUT_DIR / 'barplot.png', bbox_inches='tight', pad_inches=0.1)
print(f"Saved → {OUTPUT_DIR / 'barplot.pdf'}")
