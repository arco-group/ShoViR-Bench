"""
Attack Sensitivity Score — corrected for general noise sensitivity via RO.

Δ_OCO  = S_AUC(OCO)  − S_AUC(RO)   → net pathology-specific sensitivity
Δ_DOCO = S_AUC(DOCO) − S_AUC(RO)   → net co-occurrence sensitivity

S_AUC = (1/100) × ∫₀¹⁰⁰ [F(0)−F(p)] / F(0) dp   (trapezoid rule)

Layout: 2 rows (datasets) × 2 columns (raw | corrected Δ).
Shared y-axis within each column. Pastel background identifies the dataset.

Output: results/plots/sensitivity_score.{pdf,png}
        results/plots/sensitivity_scores.csv
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
    # MIMIC-CXR
    "aehrc__cxrmate-rrg24_cxrmateed_default": "CXRMate",
    "Chantal__RaDialog-interactive-radiology-report-generation_radialog_default": "RaDialog",
    "microsoft__maira-2_maira2_default": "MAIRA-2",
    "nvidia__NV-Reason-CXR-3B_nv_reason_default": "NV-Reason-CXR",
    "StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default": "CheXagent-2",
    "X-iZhang__libra-llava-rad_llavarad_default": "LIBRA-LLaVA",
    # Truncated filenames
    "google__medgemma-1": "MedGemma",
    "X-iZhang__libra-v1": "LIBRA-v1",
}

DS_BGCOLOR = {
    'padchest-gr':   '#E8F0FE',
    'mimic-cxr-jpg': '#E8F5E9',
}
DS_LABEL = {
    'padchest-gr':   'PadChest-GR',
    'mimic-cxr-jpg': 'MIMIC-CXR',
}

SKIP_STEMS  = {'microsoft__maira-2_maira2_default::seed=3_grounded'}

DATASETS    = [('padchest-gr', 'PadChest-GR'), ('mimic-cxr-jpg', 'MIMIC-CXR')]
P_VALUES    = [0, 20, 40, 60, 80, 100]
EXPERIMENTS = ['oco', 'doco', 'ro']

EXP_COLORS   = {'oco': '#EE6677', 'doco': '#4477AA', 'ro': '#228833'}
DELTA_COLORS = {'oco': '#EE6677', 'doco': '#4477AA'}


# ── HELPERS ─────────────────────────────────────────────────────────
def load_mean_f1(exp: str, p: int, dataset: str) -> dict[str, float]:
    p_str   = f'p{p:02d}' if p < 100 else 'p100'
    exp_dir = RESULTS_DIR / exp / p_str / dataset
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


def compute_raw_scores(exp: str, dataset: str) -> pd.DataFrame:
    curves: dict[str, dict[int, float]] = {}
    for p in P_VALUES:
        for model, f1 in load_mean_f1(exp, p, dataset).items():
            curves.setdefault(model, {})[p] = f1
    records = []
    for model, pf1 in curves.items():
        f0 = pf1.get(0, np.nan)
        if np.isnan(f0) or f0 == 0:
            continue
        ps    = sorted(pf1.keys())
        drops = [(f0 - pf1[p]) / f0 for p in ps]
        s_auc = float(np.trapezoid(drops, ps) / 100.0)
        f100  = pf1.get(100, np.nan)
        s_100 = float((f0 - f100) / f0) if not np.isnan(f100) else np.nan
        records.append({'model': model, 'S_AUC': s_auc, 'S_100': s_100})
    return pd.DataFrame(records).set_index('model')


def delta_scores(raw_exp: pd.DataFrame, raw_ro: pd.DataFrame) -> pd.DataFrame:
    shared = raw_exp.index.intersection(raw_ro.index)
    delta  = raw_exp.loc[shared] - raw_ro.loc[shared]
    delta.columns = ['Δ_S_AUC', 'Δ_S_100']
    return delta


# ── COMPUTE ──────────────────────────────────────────────────────────
all_raw   = {}
all_delta = {}

for ds_key, _ in DATASETS:
    raw = {exp: compute_raw_scores(exp, ds_key) for exp in EXPERIMENTS}
    all_raw.update({(ds_key, exp): raw[exp] for exp in EXPERIMENTS})
    for exp in ['oco', 'doco']:
        all_delta[(ds_key, exp)] = delta_scores(raw[exp], raw['ro'])

# Print tables
for ds_key, ds_label in DATASETS:
    print(f"\n{'='*56}\n  {ds_label} — Δ = S_AUC(exp) − S_AUC(RO)\n{'='*56}")
    for exp in ['oco', 'doco']:
        print(f"\n── Δ_{exp.upper()} ──")
        print(all_delta[(ds_key, exp)]
              .sort_values('Δ_S_AUC', ascending=False)
              .to_string(float_format=lambda x: f'{x:+.4f}'))

# Save CSV
frames = {}
for ds_key, ds_label in DATASETS:
    for exp in EXPERIMENTS:
        frames[f'{ds_label}_raw_{exp.upper()}'] = all_raw[(ds_key, exp)]
    for exp in ['oco', 'doco']:
        frames[f'{ds_label}_delta_{exp.upper()}'] = all_delta[(ds_key, exp)]
pd.concat(frames, axis=1).to_csv(OUTPUT_DIR / 'sensitivity_scores.csv')
print(f"\nCSV → {OUTPUT_DIR / 'sensitivity_scores.csv'}")


# ── GLOBAL MODEL ORDER (consistent across both rows) ─────────────────
# Primary sort: PadChest Δ_OCO desc; MIMIC-only models appended at end
_pc_delta_oco   = all_delta[('padchest-gr',   'oco')]
_mimic_delta_oco = all_delta[('mimic-cxr-jpg', 'oco')]

_pc_models = sorted(_pc_delta_oco.index.tolist(),
                    key=lambda m: _pc_delta_oco.loc[m, 'Δ_S_AUC'], reverse=True)
_mimic_only = sorted([m for m in _mimic_delta_oco.index if m not in _pc_delta_oco.index],
                     key=lambda m: _mimic_delta_oco.loc[m, 'Δ_S_AUC'], reverse=True)
GLOBAL_MODEL_ORDER = _pc_models + _mimic_only

# ── GLOBAL Y-LIMIT for the corrected-panel red zone ──────────────────
_global_ylim_bottom = min(
    [all_delta[(dk, e)].loc[m, 'Δ_S_AUC']
     for dk, _ in DATASETS
     for e in ['oco', 'doco']
     for m in all_delta[(dk, e)].index] + [0]
) - 0.02

# ── PLOT ─────────────────────────────────────────────────────────────
# 2 rows (datasets) × 2 cols (raw | corrected); single shared y-axis
fig, axes = plt.subplots(len(DATASETS), 2,
                         figsize=(13.0, 4.0 * len(DATASETS)),
                         sharey=True)
fig.subplots_adjust(wspace=0.20, hspace=0.06)

for row_idx, (ds_key, ds_label) in enumerate(DATASETS):
    ax_raw   = axes[row_idx, 0]
    ax_delta = axes[row_idx, 1]
    raw      = {exp: all_raw[(ds_key, exp)] for exp in EXPERIMENTS}
    delta    = {exp: all_delta[(ds_key, exp)] for exp in ['oco', 'doco']}

    # Use the global model order (NaN for models absent from this dataset)
    all_models = GLOBAL_MODEL_ORDER
    x     = np.arange(len(all_models))
    bar_w = 0.24

    # Set background
    ax_raw.set_facecolor(DS_BGCOLOR[ds_key])
    ax_delta.set_facecolor(DS_BGCOLOR[ds_key])

    # ── Raw panel ────────────────────────────────────────────────────
    n_exp_raw = len(EXPERIMENTS)
    for i, exp in enumerate(EXPERIMENTS):
        vals   = [raw[exp].loc[m, 'S_AUC'] if m in raw[exp].index else np.nan
                  for m in all_models]
        offset = (i - (n_exp_raw - 1) / 2) * 0.20
        ax_raw.bar(x + offset, vals, 0.20,
                   label=exp.upper() if row_idx == 0 else None,
                   color=EXP_COLORS[exp],
                   edgecolor='white', linewidth=0.4, alpha=0.88, zorder=3)

    ax_raw.axhline(0, color='#666666', linewidth=0.7, zorder=2)
    ax_raw.set_xticks(x)
    if row_idx == len(DATASETS) - 1:
        ax_raw.set_xticklabels(all_models, rotation=30, ha='right', fontsize=11)
    else:
        ax_raw.set_xticklabels([])
    if row_idx == 0:
        ax_raw.set_title('(a) Raw sensitivity scores', fontweight='bold', pad=5)
    ax_raw.set_ylabel(r'$S_\mathrm{AUC}$', labelpad=3)
    # ── Corrected panel ──────────────────────────────────────────────
    for i, (exp, lbl) in enumerate([('oco', r'$\Delta_\mathrm{OCO}$'),
                                     ('doco', r'$\Delta_\mathrm{DOCO}$')]):
        vals   = [delta[exp].loc[m, 'Δ_S_AUC'] if m in delta[exp].index else np.nan
                  for m in all_models]
        offset = (i - 0.5) * bar_w
        ax_delta.bar(x + offset, vals, bar_w,
                     label=lbl if row_idx == 0 else None,
                     color=DELTA_COLORS[exp],
                     edgecolor='white', linewidth=0.4, alpha=0.88, zorder=3)

    ax_delta.axhline(0, color='#333333', linewidth=1.0, linestyle='--', zorder=4,
                     label='RO noise floor' if row_idx == 0 else None)
    ax_delta.axhspan(_global_ylim_bottom, 0, color='#ffcccc', alpha=0.18, zorder=0)
    ax_delta.set_xticks(x)
    if row_idx == len(DATASETS) - 1:
        ax_delta.set_xticklabels(all_models, rotation=30, ha='right', fontsize=11)
    else:
        ax_delta.set_xticklabels([])
    if row_idx == 0:
        ax_delta.set_title('(b) Corrected scores (RO-adjusted)',
                           fontweight='bold', pad=5)
    ax_delta.set_ylabel(r'$\Delta S_\mathrm{AUC}$', labelpad=3)

    for ax in (ax_raw, ax_delta):
        ax.grid(axis='y', linestyle=':', linewidth=0.35, alpha=0.6,
                color='#888888', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')
        ax.tick_params(colors='#333333')

# Force y-tick labels visible on right column (sharey=True suppresses them)
for row in range(len(DATASETS)):
    axes[row, 1].yaxis.set_tick_params(labelleft=True)

# Single shared legend below
h1, l1 = axes[0, 0].get_legend_handles_labels()
h2, l2 = axes[0, 1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2,
           loc='lower center', ncol=len(l1) + len(l2),
           bbox_to_anchor=(0.5, -0.08),
           frameon=True, facecolor='white', edgecolor='#dddddd',
           columnspacing=1.2, handlelength=1.2, fontsize=10)

fig.savefig(OUTPUT_DIR / 'sensitivity_score.pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(OUTPUT_DIR / 'sensitivity_score.png', bbox_inches='tight', pad_inches=0.1)
print(f"Plot → {OUTPUT_DIR / 'sensitivity_score.pdf'}")
