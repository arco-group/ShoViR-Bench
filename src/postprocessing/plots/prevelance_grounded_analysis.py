"""
================================================================================
PREVALENCE–GROUNDING CORRELATION ANALYSIS  (MIMIC-CXR + PadChest-GR)
================================================================================

Eight panels: 2 rows (CheXAgent = least shortcut, MAIRA-2 = most shortcut)
× 4 columns (MIMIC-CXR Panel A, MIMIC-CXR Panel C,
              PadChest-GR Panel A, PadChest-GR Panel C).

  - Δ(OCO)  = Baseline − OCO_100   → direct-grounding signal
  - Δ(DOCO) = Baseline − DOCO_100  → indirect / contextual-grounding signal

PANEL A — Prevalence vs. Δ(OCO)    [H3: Prevalence → Grounding]
PANEL C — Δ(OCO) vs. Δ(DOCO)      [H2: Co-occurrence Shortcuts]

Layout: 2 rows (CheXAgent, MAIRA-2) × 4 columns (dataset × panel type).
Training-set prevalence is from MIMIC-CXR for both dataset columns (all VLMs
are MIMIC-CXR-trained).

Data are loaded per-model (no averaging) from results/oco|doco|ro/p*/*/  CSV files.

Output: results/plots/prevalence_grounded_analysis.{pdf,png}
================================================================================
"""
import sys
import json
import matplotlib
matplotlib.use('Agg')          # overridden to interactive when --interactive is passed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from matplotlib.lines import Line2D

# ── CONFIG ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[3] / 'results'
OUTPUT_DIR  = RESULTS_DIR / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Didot', 'GFS Didot', 'CMU Serif', 'Computer Modern', 'Georgia', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
})

# ── MODELS ───────────────────────────────────────────────────────────────────
# Two models to compare: (model_key, display_label)
MODELS = [
    ('chexagent', 'CheXAgent (less shortcut learning)'),
    ('maira-2',   'MAIRA-2 (most shortcut learning)'),
]

# Exact CSV stem (filename without .csv) for each model
MODEL_STEMS = {
    'chexagent': 'StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default::seed=3',
    'maira-2':   'microsoft__maira-2_maira2_default::seed=3',
}

# ── CLASSES ──────────────────────────────────────────────────────────────────
# 13 CheXbert classes present in the per-disease CSV columns.
CLASSES = [
    "Enl. Card.",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Eff.",
    "Pleural Other",
    "Fracture",
    "Support Dev.",
]

# CSV column name → short class label
CSV_TO_CLASS = {
    'Enlarged Cardiomediastinum': 'Enl. Card.',
    'Cardiomegaly':               'Cardiomegaly',
    'Lung Opacity':               'Lung Opacity',
    'Lung Lesion':                'Lung Lesion',
    'Edema':                      'Edema',
    'Consolidation':              'Consolidation',
    'Pneumonia':                  'Pneumonia',
    'Atelectasis':                'Atelectasis',
    'Pneumothorax':               'Pneumothorax',
    'Pleural Effusion':           'Pleural Eff.',
    'Pleural Other':              'Pleural Other',
    'Fracture':                   'Fracture',
    'Support Devices':            'Support Dev.',
}

# Training-set prevalence (% POSITIVE) from MIMIC-CXR training split.
PREVALENCE = {
    "Enl. Card.":    5.5,
    "Cardiomegaly":  22.2,
    "Lung Opacity":  24.9,
    "Lung Lesion":    3.8,
    "Edema":          7.5,
    "Consolidation":  2.9,
    "Pneumonia":      3.2,
    "Atelectasis":   20.9,
    "Pneumothorax":   2.4,
    "Pleural Eff.":  18.1,
    "Pleural Other":  2.1,
    "Fracture":       3.7,
    "Support Dev.":  22.2,
}

# Grounding taxonomy (manual assignment)
CATEGORY = {
    "Enl. Card.":    "ungrounded",
    "Cardiomegaly":  "direct",
    "Lung Opacity":  "mixed",
    "Lung Lesion":   "ungrounded",
    "Edema":         "contextual",
    "Consolidation": "mixed",
    "Pneumonia":     "mixed",
    "Atelectasis":   "contextual",
    "Pneumothorax":  "direct",
    "Pleural Eff.":  "direct",
    "Pleural Other": "ungrounded",
    "Fracture":      "ungrounded",
    "Support Dev.":  "direct",
}

CAT_COLORS = {
    "direct":      "#2E86AB",
    "contextual":  "#F6AE2D",
    "mixed":       "#A23B72",
    "ungrounded":  "#C73E1D",
}
CAT_MARKERS = {
    "direct":      "o",
    "contextual":  "s",
    "mixed":       "D",
    "ungrounded":  "X",
}
CAT_LABELS = {
    "direct":      "Directly grounded",
    "contextual":  "Contextually grounded",
    "mixed":       "Mixed",
    "ungrounded":  "Ungrounded",
}

DATASETS = [('mimic-cxr-jpg', 'MIMIC-CXR'), ('padchest-gr', 'PadChest-GR')]
DS_BGCOLOR = {
    'padchest-gr':   '#E8F5E9',
    'mimic-cxr-jpg': '#E8F0FE',
}

# Column configuration: (dataset_key, panel_letter)
COL_CONFIG = [
    ('mimic-cxr-jpg', 'A'),
    ('mimic-cxr-jpg', 'C'),
    ('padchest-gr',   'A'),
    ('padchest-gr',   'C'),
]

COL_TITLES = [
    '(A)  MIMIC-CXR\nPrevalence vs. Direct Grounding',
    '(C)  MIMIC-CXR\nDirect vs. Indirect Grounding',
    '(A)  PadChest-GR\nPrevalence vs. Direct Grounding',
    '(C)  PadChest-GR\nDirect vs. Indirect Grounding',
]

# JSON file that stores label positions. Edit it and re-run to reposition labels.
OFFSETS_JSON = Path(__file__).resolve().parent / 'label_offsets.json'

# ── LABEL OFFSETS ────────────────────────────────────────────────────────────
DEFAULT_OFFSET = (10, 10)

# Nested structure: ALL_OFFSETS[model_key][ds_key][panel] → {class: (dx, dy)}
ALL_OFFSETS: dict = {
    'chexagent': {
        'mimic-cxr-jpg': {'A': {}, 'C': {}},
        'padchest-gr':   {'A': {}, 'C': {}},
    },
    'maira-2': {
        'mimic-cxr-jpg': {'A': {}, 'C': {}},
        'padchest-gr':   {
            'A': {
                "Cardiomegaly":  ( 10, -14),
                "Support Dev.":  ( 10,   4),
                "Pleural Eff.":  (-55,   6),
                "Atelectasis":   ( 10, -12),
                "Lung Opacity":  ( 10,   4),
                "Pneumothorax":  ( 10,   4),
                "Edema":         ( 10,   4),
                "Enl. Card.":    (-65,   4),
                "Consolidation": ( 10,  -8),
                "Pneumonia":     (-70,   4),
                "Lung Lesion":   ( 10, -12),
                "Pleural Other": ( 10,   4),
                "Fracture":      (-65,  -8),
            },
            'C': {
                "Edema":         ( 10, -12),
                "Atelectasis":   ( 10,   4),
                "Cardiomegaly":  ( 10, -12),
                "Pleural Eff.":  ( 10, -12),
                "Support Dev.":  ( 10,   4),
                "Lung Opacity":  ( 10,   4),
                "Enl. Card.":    (-65,   4),
                "Consolidation": ( 10,  -8),
                "Pneumonia":     (-65,   4),
                "Lung Lesion":   ( 10, -12),
                "Pleural Other": ( 10,   4),
                "Fracture":      (-65,  -8),
                "Pneumothorax":  ( 10,   4),
            },
        },
    },
}

_PANEL_LABELS = {
    'A': 'Panel A — Prevalence vs. Direct Grounding (OCO)',
    'C': 'Panel C — Direct vs. Indirect Grounding',
}


# ── JSON OFFSET I/O ──────────────────────────────────────────────────────────
def _full_offsets(base_dict):
    """Return dict with every CLASSES entry filled (missing → DEFAULT_OFFSET)."""
    return {c: base_dict.get(c, DEFAULT_OFFSET) for c in CLASSES}


def save_offsets_json(all_offsets: dict):
    def _ser(d):
        return {k: list(v) for k, v in d.items()}

    data: dict = {
        '_info': (
            'Label offsets in typographic points from the data point. '
            '+x = right, -x = left, +y = up, -y = down. '
            'Edit values and re-run the script to reposition labels.'
        ),
    }
    for model_key, _ in MODELS:
        data[model_key] = {}
        for ds_key, _ in DATASETS:
            data[model_key][ds_key] = {}
            for p in ('A', 'C'):
                filled = _full_offsets(
                    all_offsets.get(model_key, {}).get(ds_key, {}).get(p, {})
                )
                data[model_key][ds_key][p] = {
                    '_panel': _PANEL_LABELS[p], **_ser(filled)
                }

    with open(OFFSETS_JSON, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f'Label offsets → {OFFSETS_JSON}')


def load_offsets_json() -> dict:
    if not OFFSETS_JSON.exists():
        return ALL_OFFSETS

    with open(OFFSETS_JSON) as fh:
        raw = json.load(fh)

    def _parse_section(section: dict) -> dict:
        return {k: tuple(v) for k, v in section.items() if not k.startswith('_')}

    result: dict = {}
    for model_key, _ in MODELS:
        result[model_key] = {}
        model_raw = raw.get(model_key, {})
        for ds_key, _ in DATASETS:
            result[model_key][ds_key] = {}
            ds_raw = model_raw.get(ds_key, {})
            for p in ('A', 'C'):
                if p in ds_raw:
                    result[model_key][ds_key][p] = _parse_section(ds_raw[p])
                else:
                    result[model_key][ds_key][p] = (
                        ALL_OFFSETS.get(model_key, {})
                        .get(ds_key, {}).get(p, {})
                    )
    return result


# ── DATA LOADING ─────────────────────────────────────────────────────────────
def _load_csv_dir(exp: str, p: int, dataset: str,
                  model_stem: str) -> dict[str, float]:
    """
    Load CheXBERT F1 per disease for a single model CSV in one experiment
    directory. Returns {short_class_name: f1} (NaN where data is missing).
    """
    p_str  = f'p{p:02d}' if p < 100 else 'p100'
    target = RESULTS_DIR / exp / p_str / dataset / f'{model_stem}.csv'

    result: dict[str, float] = {c: np.nan for c in CLASSES}
    if not target.exists():
        return result

    try:
        df = pd.read_csv(target, index_col=0)
    except Exception:
        return result

    if 'CheXbert F1 score' not in df.index:
        return result

    row = df.loc['CheXbert F1 score']
    for csv_col, short in CSV_TO_CLASS.items():
        if csv_col in row.index:
            val = pd.to_numeric(row[csv_col], errors='coerce')
            if not np.isnan(val):
                result[short] = float(val)

    return result


def load_model_dataset(dataset: str, model_key: str) -> dict[str, dict[str, float]]:
    """Return {'baseline', 'oco', 'doco', 'ro'} → {class: f1} for one model."""
    stem = MODEL_STEMS[model_key]
    return {
        'baseline': _load_csv_dir('oco',  0,   dataset, stem),
        'oco':      _load_csv_dir('oco',  100, dataset, stem),
        'doco':     _load_csv_dir('doco', 100, dataset, stem),
        'ro':       _load_csv_dir('ro',   100, dataset, stem),
    }


# ── COMPUTE ARRAYS ───────────────────────────────────────────────────────────
def compute_arrays(data: dict[str, dict[str, float]]):
    prev   = np.array([PREVALENCE[c]                    for c in CLASSES])
    base   = np.array([data['baseline'].get(c, np.nan)  for c in CLASSES])
    oco    = np.array([data['oco'].get(c,      np.nan)  for c in CLASSES])
    doco   = np.array([data['doco'].get(c,     np.nan)  for c in CLASSES])
    ro     = np.array([data['ro'].get(c,       np.nan)  for c in CLASSES])
    d_oco  = base - oco
    d_doco = base - doco
    ind_ratio = np.where(d_oco > 0.01, d_doco / d_oco, np.nan)
    return prev, base, oco, doco, ro, d_oco, d_doco, ind_ratio


# ── PLOT HELPERS ─────────────────────────────────────────────────────────────
def _annotate_point(ax, name, x, y, overrides=None):
    offset = overrides.get(name, DEFAULT_OFFSET) if overrides else DEFAULT_OFFSET
    return ax.annotate(
        name, xy=(x, y), xytext=offset, textcoords='offset points',
        fontsize=10, color='0.3',
        arrowprops=dict(
            arrowstyle='-',
            connectionstyle='arc3,rad=0',
            color='0.45',
            lw=0.8,
            shrinkA=0,
            shrinkB=5,
        ),
    )


def _add_regression(ax, x, y, color='0.5', label_pos=(0.03, 0.97),
                    mask=None, x_range=None):
    valid = ~(np.isnan(x) | np.isnan(y))
    if mask is not None:
        valid = valid & mask
    xf, yf = x[valid], y[valid]
    if len(xf) < 3:
        return None, None, None, None

    slope, intercept, r, p, _ = stats.linregress(xf, yf)
    rho, p_spear = stats.spearmanr(xf, yf)

    if x_range is None:
        x_range = (xf.min() - 0.5, xf.max() + 0.5)
    xx = np.linspace(x_range[0], x_range[1], 100)
    ax.plot(xx, slope * xx + intercept, '--', color=color, linewidth=1, alpha=0.7)

    ax.text(label_pos[0], label_pos[1],
            f'$r={r:.2f}$, $\\rho={rho:.2f}$',
            transform=ax.transAxes, fontsize=10, va='top', color='0.4')
    return r, p, rho, p_spear


def _scatter_classes(ax, x, y, overrides=None, draggables=None, panel_key=None):
    for i, c in enumerate(CLASSES):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        cat = CATEGORY[c]
        ax.scatter(x[i], y[i],
                   c=CAT_COLORS[cat], marker=CAT_MARKERS[cat],
                   s=90, edgecolors='white', linewidth=0.5, zorder=3)
        ann = _annotate_point(ax, c, x[i], y[i], overrides=overrides)
        if draggables is not None:
            draggables.append(_DraggableAnnotation(ann, panel_key, c))


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#333333')
    ax.grid(axis='both', linestyle=':', linewidth=0.3, alpha=0.5,
            color='#888888', zorder=0)
    ax.set_axisbelow(True)


# ── DRAGGABLE ANNOTATIONS ─────────────────────────────────────────────────────
class _DraggableAnnotation:
    """
    Makes a matplotlib Annotation draggable with the mouse.
    On figure close the updated offsets are saved to OFFSETS_JSON.
    """
    def __init__(self, ann, panel_key, name):
        self.ann       = ann
        self.panel_key = panel_key   # e.g. "chexagent__mimic-cxr-jpg__A"
        self.name      = name
        self._press    = None
        self._init_off = None
        canvas = ann.figure.canvas
        self._cids = [
            canvas.mpl_connect('button_press_event',   self._on_press),
            canvas.mpl_connect('button_release_event', self._on_release),
            canvas.mpl_connect('motion_notify_event',  self._on_motion),
        ]

    def _on_press(self, event):
        if event.inaxes != self.ann.axes:
            return
        contains, _ = self.ann.contains(event)
        if not contains:
            return
        self._press    = (event.x, event.y)
        self._init_off = self.ann.xyann

    def _on_motion(self, event):
        if self._press is None:
            return
        dpi   = self.ann.figure.get_dpi()
        scale = 72.0 / dpi
        dx_px = event.x - self._press[0]
        dy_px = event.y - self._press[1]
        self.ann.xyann = (
            self._init_off[0] + dx_px * scale,
            self._init_off[1] + dy_px * scale,
        )
        self.ann.figure.canvas.draw_idle()

    def _on_release(self, event):
        self._press = None

    def offset(self):
        x, y = self.ann.xyann
        return (round(x), round(y))


# ── MAIN FIGURE ───────────────────────────────────────────────────────────────
def make_figure(interactive=False):
    if interactive:
        _GUI_BACKENDS = ('Qt5Agg', 'Qt6Agg', 'TkAgg', 'wxAgg', 'GTK3Agg')
        for _backend in _GUI_BACKENDS:
            try:
                plt.switch_backend(_backend)
                print(f'Interactive backend: {_backend}')
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                f'No interactive GUI backend found. Tried: {_GUI_BACKENDS}\n'
                'Install PyQt5 (pip install PyQt5) or another GUI toolkit.'
            )

    all_offsets = load_offsets_json()
    if OFFSETS_JSON.exists():
        print(f'Using label offsets from {OFFSETS_JSON}')

    # Pre-load per-model, per-dataset arrays
    arrays: dict = {}
    for model_key, _ in MODELS:
        arrays[model_key] = {}
        for ds_key, _ in DATASETS:
            data = load_model_dataset(ds_key, model_key)
            arrays[model_key][ds_key] = compute_arrays(data)

    draggables = [] if interactive else None

    n_rows, n_cols = len(MODELS), len(COL_CONFIG)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.8 * n_cols, 5.0 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.32, hspace=0.22, top=0.90)

    for row_idx, (model_key, model_label) in enumerate(MODELS):

        # Row label rotated on the left of the first column
        axes[row_idx, 0].annotate(
            model_label,
            xy=(-0.28, 0.5), xycoords='axes fraction',
            fontsize=11, fontweight='bold', color='#222222',
            va='center', ha='center', rotation=90,
        )

        for col_idx, (ds_key, panel) in enumerate(COL_CONFIG):
            ax = axes[row_idx, col_idx]
            prev, base, oco, doco, ro, d_oco, d_doco, ind_ratio = \
                arrays[model_key][ds_key]
            bg  = DS_BGCOLOR[ds_key]
            off = all_offsets.get(model_key, {}).get(ds_key, {}).get(panel, {})
            pk  = f'{model_key}__{ds_key}__{panel}'

            ax.set_facecolor(bg)

            if panel == 'A':
                # Prevalence vs Δ(OCO)
                _scatter_classes(ax, prev, d_oco, off,
                                 draggables=draggables, panel_key=pk)
                _add_regression(ax, prev, d_oco,
                                mask=(d_oco > 0.02), x_range=(0, 27),
                                label_pos=(0.04, 0.97))
                ax.set_xlabel('Training Prevalence (%)')
                ax.set_ylabel(r'$\Delta$(OCO) — Direct Grounding')
                ax.set_xlim(-1, 27)
                valid_oco = d_oco[~np.isnan(d_oco)]
                ymax = valid_oco.max() * 1.15 if valid_oco.size > 0 else 0.1
                ax.set_ylim(-0.02, ymax)

            elif panel == 'C':
                # Δ(OCO) vs Δ(DOCO)
                _scatter_classes(ax, d_oco, d_doco, off,
                                 draggables=draggables, panel_key=pk)
                valid_oco  = d_oco[~np.isnan(d_oco)]
                valid_doco = d_doco[~np.isnan(d_doco)]
                x_max = valid_oco.max() * 1.1 if valid_oco.size > 0 else 0.1
                _add_regression(ax, d_oco, d_doco,
                                x_range=(0, x_max),
                                label_pos=(0.04, 0.97))
                ax.plot([0, x_max], [0, x_max], ':', color='0.72', linewidth=0.8)
                ax.text(x_max * 0.62, x_max * 0.68,
                        r'$\Delta$DOCO $=$ $\Delta$OCO',
                        fontsize=7, color='0.6', rotation=38)
                ax.set_xlabel(r'$\Delta$(OCO) — Direct Grounding')
                ax.set_ylabel(r'$\Delta$(DOCO) — Indirect Grounding')
                xmin = min(-0.02, valid_oco.min() * 1.15) if valid_oco.size > 0 else -0.02
                ax.set_xlim(xmin, x_max)
                ymax = valid_doco.max() * 1.35 if valid_doco.size > 0 else 0.1
                ymin = min(-0.02, valid_doco.min() * 1.15) if valid_doco.size > 0 else -0.02
                ax.set_ylim(ymin, ymax)

            _style_ax(ax)

            # Column titles on the first row only
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontweight='bold', pad=7)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], marker=CAT_MARKERS[k], color='w',
               markerfacecolor=CAT_COLORS[k], markersize=9, label=CAT_LABELS[k])
        for k in ['direct', 'contextual', 'mixed', 'ungrounded']
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=12, frameon=True, facecolor='white', edgecolor='#dddddd',
               fancybox=False, columnspacing=1.2, handlelength=1.2,
               bbox_to_anchor=(0.5, 0.99))

    # ── Interactive: drag labels, save offsets on close ───────────────────────
    if interactive:
        def _on_close(event):
            new_all: dict = {
                mk: {ds_key: {'A': {}, 'C': {}} for ds_key, _ in DATASETS}
                for mk, _ in MODELS
            }
            for d in draggables:
                # panel_key format: "model_key__ds_key__panel"
                parts = d.panel_key.split('__')
                mk, dk, pn = parts[0], parts[1], parts[2]
                new_all[mk][dk][pn][d.name] = d.offset()
            # Fill untouched panels from loaded offsets
            for mk, _ in MODELS:
                for dk, _ in DATASETS:
                    for pn in ('A', 'C'):
                        if not new_all[mk][dk][pn]:
                            new_all[mk][dk][pn] = (
                                all_offsets.get(mk, {}).get(dk, {}).get(pn, {})
                            )
            save_offsets_json(new_all)

        fig.canvas.mpl_connect('close_event', _on_close)
        print('Drag labels to reposition them. Close the window to save updated offsets.')
        plt.show()

    else:
        for ext in ('pdf', 'png'):
            out = OUTPUT_DIR / f'prevalence_grounded_analysis.{ext}'
            fig.savefig(out, bbox_inches='tight', pad_inches=0.1,
                        dpi=300 if ext == 'png' else None)
            print(f'Saved → {out}')
        save_offsets_json(all_offsets)
        plt.close()


# ── STATISTICS TABLE ──────────────────────────────────────────────────────────
def print_statistics():
    for model_key, model_label in MODELS:
        print(f'\n{"#"*72}\n  MODEL: {model_label}\n{"#"*72}')
        for ds_key, ds_label in DATASETS:
            data = load_model_dataset(ds_key, model_key)
            prev, base, oco, doco, ro, d_oco, d_doco, ind_ratio = \
                compute_arrays(data)

            print(f'\n  {"="*68}\n    {ds_label}\n  {"="*68}')

            def _corr(x, y, label):
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 3:
                    print(f'  {label}: insufficient data'); return
                _, _, r, p, _ = stats.linregress(x[valid], y[valid])
                rho, ps = stats.spearmanr(x[valid], y[valid])
                print(f'  {label}')
                print(f'    Pearson  r = {r:.3f}  (p = {p:.4f})')
                print(f'    Spearman ρ = {rho:.3f}  (p = {ps:.4f})')

            _corr(prev, d_oco,   '(A) Prevalence vs Δ(OCO)  [H3]')
            _corr(d_oco, d_doco, '(C) Δ(OCO) vs Δ(DOCO)    [H2]')

            print(f'\n  {"Class":<16} {"Prev%":>6} {"Base":>6} {"OCO":>6} '
                  f'{"DOCO":>6} {"RO":>6} {"ΔOCO":>6} {"ΔDOCO":>6} {"Ind.%":>6}')
            print('  ' + '-' * 72)
            for i, c in enumerate(CLASSES):
                ind = f'{ind_ratio[i]*100:.0f}%' if not np.isnan(ind_ratio[i]) else 'n/a'
                print(f'  {c:<16} {prev[i]:>5.1f}% '
                      f'{base[i]:>6.3f} {oco[i]:>6.3f} '
                      f'{doco[i]:>6.3f} {ro[i]:>6.3f} '
                      f'{d_oco[i]:>6.3f} {d_doco[i]:>6.3f} {ind:>6}')


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    interactive = '--interactive' in sys.argv or '-i' in sys.argv
    print_statistics()
    make_figure(interactive=interactive)
    if not interactive:
        print('\nDone.')
