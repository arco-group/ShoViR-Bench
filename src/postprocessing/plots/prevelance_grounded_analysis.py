"""
================================================================================
PREVALENCE–GROUNDING CORRELATION ANALYSIS  (MIMIC-CXR + PadChest-GR)
================================================================================

Three panels per dataset row, testing three hypotheses about how training-data
properties relate to visual grounding quality in VLMs for radiology report
generation.

  - Δ(OCO)  = Baseline − OCO_100   → direct-grounding signal
  - Δ(DOCO) = Baseline − DOCO_100  → indirect / contextual-grounding signal

PANEL A — Prevalence vs. Δ(OCO)    [H3: Prevalence → Grounding]
PANEL B — Prevalence vs. Baseline  [H1: Prior Dominance]
PANEL C — Δ(OCO) vs. Δ(DOCO)      [H2: Co-occurrence Shortcuts]

Layout: 2 rows (MIMIC-CXR, PadChest-GR) × 3 columns (Panels A/B/C).
Training-set prevalence is from MIMIC-CXR for both rows (all VLMs are
MIMIC-CXR-trained, so the same learned prior applies when tested on PadChest-GR).

Data are loaded directly from results/oco|doco|ro/p*/*/  CSV files,
averaging over all available models.

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

# ── CLASSES ──────────────────────────────────────────────────────────────────
# 13 CheXbert classes present in the per-disease CSV columns.
# "No Finding" is excluded — it is absent from per-disease result files.
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
# Applied to both dataset rows since all VLMs were trained on MIMIC-CXR.
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
DS_LABEL = {
    'padchest-gr':   'PadChest-GR',
    'mimic-cxr-jpg': 'MIMIC-CXR',
}

SKIP_STEMS = {'microsoft__maira-2_maira2_default::seed=3_grounded'}

# JSON file that stores label positions. Edit it and re-run to reposition labels.
# It is written automatically next to the PDF/PNG output on every save.
OFFSETS_JSON = Path(__file__).resolve().parent / 'label_offsets.json'

# ── LABEL OFFSETS ────────────────────────────────────────────────────────────
# Each dict maps class label → (dx, dy) in *points* from the data point.
# Positive dx = right, negative dx = left; positive dy = up, negative dy = down.
# Classes not listed use the DEFAULT_OFFSET below.
# A thin leader line is drawn automatically from the label to the dot.
# To move a label: change its (dx, dy) tuple and re-run the script.
DEFAULT_OFFSET = (10, 10)   # fallback for unlisted classes

# ── PadChest-GR offsets (row 1) ──────────────────────────────────────────────
OFFSETS_PADCHEST_A = {                  # Panel A: Prevalence vs Δ(OCO)
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
}
OFFSETS_PADCHEST_B = {                  # Panel B: Prevalence vs Baseline
    "Cardiomegaly":  ( 10, -14),
    "Support Dev.":  ( 10, -12),
    "Pleural Eff.":  (-65, -10),
    "Atelectasis":   ( 10,   4),
    "Lung Opacity":  ( 10,   4),
    "Edema":         ( 10,   4),
    "Enl. Card.":    (-65,   4),
    "Consolidation": (-70,  -8),
    "Pneumonia":     (-65,   4),
    "Lung Lesion":   ( 10, -12),
    "Pleural Other": ( 10,   4),
    "Fracture":      (-65,  -8),
    "Pneumothorax":  ( 10,   4),
}
# ── MIMIC-CXR offsets (row 0) — edit after first run ─────────────────────────
OFFSETS_MIMIC_A = {}                    # Panel A: all classes use DEFAULT_OFFSET
OFFSETS_MIMIC_B = {}                    # Panel B

# Master lookup: all_offsets[ds_key]['A'/'B'] → per-class offset dict
ALL_OFFSETS = {
    'mimic-cxr-jpg': {'A': OFFSETS_MIMIC_A,   'B': OFFSETS_MIMIC_B},
    'padchest-gr':   {'A': OFFSETS_PADCHEST_A, 'B': OFFSETS_PADCHEST_B},
}


# ── JSON OFFSET I/O ──────────────────────────────────────────────────────────
_PANEL_LABELS = {
    'A': 'Panel A — Prevalence vs. Direct Grounding (OCO)',
    'B': 'Panel B — Prevalence vs. Baseline Performance',
}


def _full_offsets(base_dict):
    """Return dict with every CLASSES entry filled (missing → DEFAULT_OFFSET)."""
    return {c: base_dict.get(c, DEFAULT_OFFSET) for c in CLASSES}


def save_offsets_json(all_offsets: dict):
    """
    Serialize all_offsets[ds_key]['A'/'B'/'C'] → JSON.
    Structure: { ds_key: { panel: { '_panel': label, class: [dx, dy], ... } } }
    """
    def _ser(d):
        return {k: list(v) for k, v in d.items()}

    data: dict = {
        '_info': (
            'Label offsets in typographic points from the data point. '
            '+x = right, -x = left, +y = up, -y = down. '
            'Edit values and re-run the script to reposition labels.'
        ),
    }
    for ds_key in ('mimic-cxr-jpg', 'padchest-gr'):
        panels = all_offsets.get(ds_key, {})
        data[ds_key] = {}
        for p in ('A', 'B'):
            filled = _full_offsets(panels.get(p, {}))
            data[ds_key][p] = {'_panel': _PANEL_LABELS[p], **_ser(filled)}

    with open(OFFSETS_JSON, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f'Label offsets → {OFFSETS_JSON}')


def load_offsets_json() -> dict:
    """
    Read OFFSETS_JSON and return all_offsets[ds_key][panel] → {class: (dx,dy)}.
    Falls back to ALL_OFFSETS if the file does not exist or a key is missing.
    Keys starting with '_' (comments) are silently ignored.
    """
    if not OFFSETS_JSON.exists():
        return ALL_OFFSETS

    with open(OFFSETS_JSON) as fh:
        raw = json.load(fh)

    def _parse_section(section: dict) -> dict:
        return {k: tuple(v) for k, v in section.items() if not k.startswith('_')}

    result: dict = {}
    for ds_key in ('mimic-cxr-jpg', 'padchest-gr'):
        result[ds_key] = {}
        ds_raw = raw.get(ds_key, {})
        for p in ('A', 'B'):
            if p in ds_raw:
                result[ds_key][p] = _parse_section(ds_raw[p])
            else:
                result[ds_key][p] = ALL_OFFSETS[ds_key][p]
    return result


# ── DATA LOADING ─────────────────────────────────────────────────────────────
def _load_csv_dir(exp: str, p: int, dataset: str) -> dict[str, float]:
    """
    Average CheXBERT F1 per disease across all model CSVs in one experiment
    directory.  Returns {short_class_name: mean_f1}.
    """
    p_str   = f'p{p:02d}' if p < 100 else 'p100'
    exp_dir = RESULTS_DIR / exp / p_str / dataset
    per_model: dict[str, list[float]] = {c: [] for c in CLASSES}

    for csv_path in sorted(exp_dir.glob('*.csv')):
        if csv_path.stem in SKIP_STEMS:
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        if 'CheXbert F1 score' not in df.index:
            continue
        row = df.loc['CheXbert F1 score']
        for csv_col, short in CSV_TO_CLASS.items():
            if csv_col in row.index:
                val = pd.to_numeric(row[csv_col], errors='coerce')
                if not np.isnan(val):
                    per_model[short].append(float(val))

    return {c: float(np.mean(v)) if v else np.nan
            for c, v in per_model.items()}


def load_dataset(dataset: str) -> dict[str, dict[str, float]]:
    """Return {'baseline', 'oco', 'doco', 'ro'} → {class: mean_f1}."""
    return {
        'baseline': _load_csv_dir('oco',  0,   dataset),   # p=0 identical across exps
        'oco':      _load_csv_dir('oco',  100, dataset),
        'doco':     _load_csv_dir('doco', 100, dataset),
        'ro':       _load_csv_dir('ro',   100, dataset),
    }


# ── COMPUTE ARRAYS ───────────────────────────────────────────────────────────
def compute_arrays(data: dict[str, dict[str, float]]):
    """Convert data dicts to aligned numpy arrays for one dataset."""
    prev  = np.array([PREVALENCE[c]          for c in CLASSES])
    base  = np.array([data['baseline'].get(c, np.nan) for c in CLASSES])
    oco   = np.array([data['oco'].get(c,      np.nan) for c in CLASSES])
    doco  = np.array([data['doco'].get(c,     np.nan) for c in CLASSES])
    ro    = np.array([data['ro'].get(c,       np.nan) for c in CLASSES])
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


# ── MAIN FIGURE ──────────────────────────────────────────────────────────────
# ── DRAGGABLE ANNOTATIONS ────────────────────────────────────────────────────
class _DraggableAnnotation:
    """
    Makes a matplotlib Annotation draggable with the mouse.
    Drag the label text; the leader line redraws automatically.
    On figure close the updated offsets are printed to stdout.
    """
    def __init__(self, ann, panel_key, name):
        self.ann       = ann
        self.panel_key = panel_key
        self.name      = name
        self._press    = None       # (event.x, event.y) at mouse-down
        self._init_off = None       # xyann at mouse-down
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
        self._init_off = self.ann.xyann          # current (dx, dy) in offset-points

    def _on_motion(self, event):
        if self._press is None:
            return
        # Convert pixel delta → offset-point delta  (72 pt / inch, dpi px / inch)
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
        """Return current offset as rounded (dx, dy) ints."""
        x, y = self.ann.xyann
        return (round(x), round(y))


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

    # Load label offsets: JSON file takes priority over hardcoded dicts
    all_offsets = load_offsets_json()
    if OFFSETS_JSON.exists():
        print(f'Using label offsets from {OFFSETS_JSON}')

    # Load real data from CSVs
    ds_data = {ds_key: load_dataset(ds_key) for ds_key, _ in DATASETS}

    # Pre-compute arrays for each dataset
    arrays = {ds_key: compute_arrays(ds_data[ds_key]) for ds_key, _ in DATASETS}

    draggables = [] if interactive else None   # collect DraggableAnnotation objects

    n_rows, n_cols = len(DATASETS), 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.0 * n_cols, 5.2 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.30, hspace=0.15, top=0.92)

    col_titles = [
        '(A)  Prevalence vs. Direct Grounding',
        '(B)  Prevalence vs. Baseline Performance',
    ]

    for row_idx, (ds_key, ds_label) in enumerate(DATASETS):
        prev, base, oco, doco, ro, d_oco, d_doco, ind_ratio = arrays[ds_key]
        bg = DS_BGCOLOR[ds_key]

        # Per-dataset, per-panel offset dicts
        off_a = all_offsets[ds_key]['A']
        off_b = all_offsets[ds_key]['B']

        # ── Panel A: Prevalence vs Δ(OCO) ──────────────────────────────────
        ax = axes[row_idx, 0]
        ax.set_facecolor(bg)
        _scatter_classes(ax, prev, d_oco, off_a,
                         draggables=draggables, panel_key=f'{ds_key}__A')
        _add_regression(ax, prev, d_oco,
                        mask=(d_oco > 0.02), x_range=(0, 27),
                        label_pos=(0.04, 0.97))
        ax.set_xlabel('Training Prevalence (%)')
        ax.set_ylabel(r'$\Delta$(OCO) — Direct Grounding')
        ax.set_xlim(-1, 27)
        ymax_A = np.nanmax(d_oco[~np.isnan(d_oco)]) * 1.15
        ax.set_ylim(-0.02, ymax_A)
        _style_ax(ax)

        # ── Panel B: Prevalence vs Baseline ─────────────────────────────────
        ax = axes[row_idx, 1]
        ax.set_facecolor(bg)
        _scatter_classes(ax, prev, base, off_b,
                         draggables=draggables, panel_key=f'{ds_key}__B')
        _add_regression(ax, prev, base, x_range=(0, 27), label_pos=(0.04, 0.97))
        ax.set_xlabel('Training Prevalence (%)')
        ax.set_ylabel('Baseline F1')
        ax.set_xlim(-1, 27)
        ymax_B = np.nanmax(base[~np.isnan(base)]) * 1.15
        ax.set_ylim(-0.02, ymax_B)
        _style_ax(ax)

        # Column titles on the first row only
        if row_idx == 0:
            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontweight='bold', pad=7)

    # ── Shared legend ────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], marker=CAT_MARKERS[k], color='w',
               markerfacecolor=CAT_COLORS[k], markersize=9, label=CAT_LABELS[k])
        for k in ['direct', 'contextual', 'mixed', 'ungrounded']
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=12, frameon=True, facecolor='white', edgecolor='#dddddd',
               fancybox=False, columnspacing=1.2, handlelength=1.2,
               bbox_to_anchor=(0.5, 0.99))

    # ── Interactive mode: drag labels, save offsets on close ─────────────────
    if interactive:
        def _on_close(event):
            # Rebuild nested all_offsets from draggable annotations
            new_all: dict = {ds_key: {'A': {}, 'B': {}}
                             for ds_key, _ in DATASETS}
            for d in draggables:
                ds_key, panel = d.panel_key.split('__')
                new_all[ds_key][panel][d.name] = d.offset()
            # Fill any panels that were not touched (keep loaded offsets)
            for ds_key in new_all:
                for panel in ('A', 'B'):
                    if not new_all[ds_key][panel]:
                        new_all[ds_key][panel] = all_offsets[ds_key][panel]
            save_offsets_json(new_all)

        fig.canvas.mpl_connect('close_event', _on_close)
        print('Drag labels to reposition them. Close the window to save updated offsets.')
        plt.show()

    # ── Save mode ────────────────────────────────────────────────────────────
    else:
        for ext in ('pdf', 'png'):
            out = OUTPUT_DIR / f'prevalence_grounded_analysis.{ext}'
            fig.savefig(out, bbox_inches='tight', pad_inches=0.1,
                        dpi=300 if ext == 'png' else None)
            print(f'Saved → {out}')
        save_offsets_json(all_offsets)   # write JSON alongside the plot
        plt.close()


# ── STATISTICS TABLE ─────────────────────────────────────────────────────────
def print_statistics():
    ds_data = {ds_key: load_dataset(ds_key) for ds_key, _ in DATASETS}

    for ds_key, ds_label in DATASETS:
        prev, base, oco, doco, ro, d_oco, d_doco, ind_ratio = \
            compute_arrays(ds_data[ds_key])

        print(f'\n{"="*72}\n  {ds_label}\n{"="*72}')

        def _corr(x, y, label):
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 3:
                print(f'  {label}: insufficient data'); return
            _, _, r, p, _ = stats.linregress(x[valid], y[valid])
            rho, ps = stats.spearmanr(x[valid], y[valid])
            print(f'  {label}')
            print(f'    Pearson  r = {r:.3f}  (p = {p:.4f})')
            print(f'    Spearman ρ = {rho:.3f}  (p = {ps:.4f})')

        _corr(prev, d_oco, '(A) Prevalence vs Δ(OCO)  [H3]')
        _corr(prev, base,  '(B) Prevalence vs Baseline [H1]')

        # Per-class table
        print(f'\n  {"Class":<16} {"Prev%":>6} {"Base":>6} {"OCO":>6} '
              f'{"DOCO":>6} {"RO":>6} {"ΔOCO":>6} {"ΔDOCO":>6} {"Ind.%":>6}')
        print('  ' + '-' * 74)
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
