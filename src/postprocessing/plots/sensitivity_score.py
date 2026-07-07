"""
Adjusted delta:

    Δ(exp) = F1(RO, p=100) − F1(exp, p=100)

Positive → exp degrades performance more than random occlusion.
Shown for OCO and DOCO at full occlusion strength (p=100).

Layout: 1 row × 2 columns (PadChest-GR | MIMIC-CXR).
Each subplot: grouped bars per model, two bars (OCO / DOCO).
Shared y-axis across subplots.

Output: results/plots/sensitivity_score.{pdf,png}
        results/plots/sensitivity_scores.csv
"""
import json
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

SKIP_STEMS = {'microsoft__maira-2_maira2_default::seed=3_grounded'}

# ── WEIGHTED MEAN HELPERS ────────────────────────────────────────────
_WEIGHTS_JSON = Path(__file__).resolve().parent / 'weights.json'
with open(_WEIGHTS_JSON) as _f:
    _RAW_WEIGHTS = json.load(_f)

_DS_KEY = {'padchest-gr': 'padchest', 'mimic-cxr-jpg': 'mimic'}


def _weighted_f1(row: pd.Series, dataset: str, exp_type: str) -> float:
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
EXPERIMENTS = ['oco', 'doco', 'ro']

# Experiment colours (OCO / DOCO bars)
EXP_COLOR = {'oco': '#EE6677', 'doco': '#4477AA'}

EXP_LABEL = {
    exp: rf'$\Delta = \mu\text{{-}}F1_{{\mathrm{{RO}}}} - \mu\text{{-}}F1_{{\mathrm{{{exp.upper()}}}}}$'
    for exp in ('oco', 'doco')
}

# Dataset background
DS_BGCOLOR = {
    'padchest-gr':   '#E8F0FE',
    'mimic-cxr-jpg': '#E8F5E9',
}


# ── HELPERS ─────────────────────────────────────────────────────────
def load_mean_f1_p100(exp: str, dataset: str) -> dict[str, float]:
    """Weighted-mean CheXBERT F1 at p=100 for the given experiment."""
    exp_dir = RESULTS_DIR / exp / 'p100' / dataset
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
            out[short] = _weighted_f1(df.loc['CheXbert F1 score'], dataset, exp)
    return out


def compute_deltas(dataset: str) -> pd.DataFrame:
    """
    Δ(exp) = F1(RO, p=100) − F1(exp, p=100)

    Returns a DataFrame with columns ['oco', 'doco'] indexed by model.
    Models missing from RO are dropped.
    """
    ro   = load_mean_f1_p100('ro',   dataset)
    oco  = load_mean_f1_p100('oco',  dataset)
    doco = load_mean_f1_p100('doco', dataset)

    records = []
    for m in sorted(set(ro) & (set(oco) | set(doco))):
        f_ro = ro.get(m, np.nan)
        if np.isnan(f_ro):
            continue
        row = {'model': m}
        for exp, src in [('oco', oco), ('doco', doco)]:
            f_exp = src.get(m, np.nan)
            row[exp] = f_ro - f_exp if not np.isnan(f_exp) else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index('model')


# ── COMPUTE ──────────────────────────────────────────────────────────
deltas = {ds_key: compute_deltas(ds_key) for ds_key, _ in DATASETS}

# Print tables
for ds_key, ds_label in DATASETS:
    df = deltas[ds_key]
    print(f"\n{'='*56}\n  {ds_label} — Δ = F1(RO100) − F1(exp100)\n{'='*56}")
    for exp in ['oco', 'doco']:
        if exp in df.columns:
            print(f"\n── {exp.upper()} ──")
            print(df[exp].sort_values(ascending=False)
                         .to_string(float_format=lambda x: f'{x:+.4f}'))

# Save CSV
frames = {}
for ds_key, ds_label in DATASETS:
    for exp in ['oco', 'doco']:
        col = deltas[ds_key][exp] if exp in deltas[ds_key].columns else pd.Series(dtype=float)
        frames[f'{ds_label}_{exp.upper()}'] = col
pd.concat(frames, axis=1).to_csv(OUTPUT_DIR / 'sensitivity_scores.csv')
print(f"\nCSV → {OUTPUT_DIR / 'sensitivity_scores.csv'}")


# ── GLOBAL MODEL ORDER ────────────────────────────────────────────────
# Sort by PadChest-GR OCO Δ desc; MIMIC-only models appended
_pc  = deltas['padchest-gr']
_mim = deltas['mimic-cxr-jpg']

_pc_models  = sorted(_pc.index.tolist(),
                     key=lambda m: _pc.loc[m, 'oco'] if not np.isnan(_pc.loc[m, 'oco']) else -np.inf,
                     reverse=True)
_mimic_only = sorted([m for m in _mim.index if m not in _pc.index],
                     key=lambda m: _mim.loc[m, 'oco'] if not np.isnan(_mim.loc[m, 'oco']) else -np.inf,
                     reverse=True)
GLOBAL_MODEL_ORDER = _pc_models + _mimic_only


# ── PLOT ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(DATASETS),
                         figsize=(6.5 * len(DATASETS), 4.8),
                         sharey=True)
fig.subplots_adjust(wspace=0.06)

bar_w = 0.35

for col_idx, (ds_key, ds_label) in enumerate(DATASETS):
    ax  = axes[col_idx]
    df  = deltas[ds_key]

    # Use only models present in this dataset
    models = [m for m in GLOBAL_MODEL_ORDER if m in df.index]
    x      = np.arange(len(models))

    ax.set_facecolor(DS_BGCOLOR[ds_key])

    for i, exp in enumerate(['oco', 'doco']):
        vals   = [df.loc[m, exp] if m in df.index and not np.isnan(df.loc[m, exp]) else np.nan
                  for m in models]
        offset = (i - 0.5) * bar_w
        lbl    = EXP_LABEL[exp]
        ax.bar(x + offset, vals, bar_w,
               label=lbl if col_idx == 0 else None,
               color=EXP_COLOR[exp],
               edgecolor='white', linewidth=0.4, alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11)

    ax.grid(axis='y', linestyle=':', linewidth=0.35, alpha=0.6, color='#888888', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#333333')

axes[0].set_ylabel(
    r'$\Delta = \mu\text{-}F1_{\mathrm{RO}} - \mu\text{-}F1(\mathrm{\star})$',
    labelpad=6, fontsize=14)

# Shared legend below
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, -0.18),
           frameon=True, facecolor='white', edgecolor='#dddddd',
           columnspacing=1.4, handlelength=1.2, fontsize=14)

fig.savefig(OUTPUT_DIR / 'sensitivity_score.pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(OUTPUT_DIR / 'sensitivity_score.png', bbox_inches='tight', pad_inches=0.1)
print(f"Plot → {OUTPUT_DIR / 'sensitivity_score.pdf'}")
