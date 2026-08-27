"""
Cross-disease flip-rate and ground-truth co-occurrence plots for OCO experiments.

Two distinct statistics, plotted separately so neither is muddied by the other:

1. Co-occurrence (data statistic, model-independent): how often disease A and
   disease B are both present in the same ground-truth image. This is what makes
   "occlude A, watch B" a meaningful shortcut-learning probe in the first place.
   Source: evaluations/cooccurrence_stats.py.

2. Flip rate (model behavior): among images where B is truly present and B != A,
   the fraction of cases where the model correctly mentioned B at baseline but
   stopped mentioning it once A's region was occluded. Shown as a single panel at
   100% occlusion (the clearest, most policy-relevant condition) rather than six
   small panels, plus a separate dose-response curve for the strength trend.
   Source: evaluations/flip_rate_cross_disease.py.

Inputs (both under results/oco/flip_rate_cross_disease/<dataset>/):
    class_prevalence.csv, cooccurrence_conditional_prob.csv   (co-occurrence)
    flip_rate_cross_disease_summary.csv                        (flip rate)

Outputs (results/plots/):
    cooccurrence_<dataset>.{pdf,png}
    flip_rate_heatmap_<dataset>.{pdf,png}
    flip_rate_dose_response.{pdf,png}
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[3] / 'results'
FLIP_DIR = RESULTS_DIR / 'oco' / 'flip_rate_cross_disease'
OUTPUT_DIR = RESULTS_DIR / 'plots'
OUTPUT_DIR.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Didot', 'GFS Didot', 'CMU Serif', 'Computer Modern', 'Georgia', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
})

STRENGTHS = ['p00', 'p20', 'p40', 'p60', 'p80', 'p100']
FINAL_STRENGTH = 'p100'

DATASETS = [
    ('padchest-gr', 'PadChest-GR'),
    ('mimic-cxr-jpg', 'MIMIC-CXR'),
]

FLIP_CMAP = plt.get_cmap('OrRd')       # model behavior (flip rate)
COOC_CMAP = plt.get_cmap('Blues')      # data statistic (co-occurrence)

# Minimum support to plot/report a cell as a real estimate rather than noise.
# At n=30, a proportion near 0.5 has a ~95% CI half-width of ~18 points
# (1.96 * sqrt(0.25/30)); below that the point estimate isn't worth commenting on.
MIN_PAIR_N = 30


def prevalence_order(dataset: str) -> list:
    path = FLIP_DIR / dataset / 'class_prevalence.csv'
    prev = pd.read_csv(path, index_col=0)['prevalence']
    return list(prev.sort_values(ascending=False).index)


# Fixed class -> color assignment (never cycled/reassigned), shared by every plot
# and both datasets, so e.g. "Cardiomegaly" is always the same color everywhere.
ALL_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices',
]
_tab20 = plt.get_cmap('tab20')
CLASS_COLOR = {c: _tab20(i / 20) for i, c in enumerate(ALL_CLASSES)}


def compute_reliable_pairs(dataset: str) -> pd.DataFrame:
    """Long table of every (A,B) pair with gt_cooccurrence_n, flip_rate/flip_support_n
    at FINAL_STRENGTH, and a `reliable` flag (both counts >= MIN_PAIR_N). This is the
    single definition of "reliable" used everywhere -- heatmaps, the exported CSV,
    and the per-pair dynamics plot all filter through this."""
    counts_path = FLIP_DIR / dataset / 'cooccurrence_counts.csv'
    flip_df = load_flip_summary(dataset)
    if not counts_path.exists() or flip_df.empty:
        return pd.DataFrame()

    cooccur = pd.read_csv(counts_path, index_col=0)
    flip_sub = flip_df[flip_df['strength'] == FINAL_STRENGTH]
    flip_agg = flip_sub.groupby(['occluded_class', 'affected_class']).agg(
        flip_rate=('flip_rate', 'mean'),
        flip_support_n=('n_baseline_positive', 'sum'),
    ).reset_index()

    rows = []
    for a in cooccur.index:
        for b in cooccur.columns:
            if a == b:
                continue
            rows.append({'occluded_class': a, 'affected_class': b, 'gt_cooccurrence_n': int(cooccur.loc[a, b])})
    cooccur_long = pd.DataFrame(rows)

    merged = cooccur_long.merge(flip_agg, on=['occluded_class', 'affected_class'], how='left')
    merged['reliable'] = (
        (merged['gt_cooccurrence_n'] >= MIN_PAIR_N)
        & (merged['flip_support_n'].fillna(0) >= MIN_PAIR_N)
    )
    return merged


def draw_matrix(ax, mat: np.ndarray, rows: list, cols: list, cmap, vmax: float,
                 fmt: str = '.2f', counts: np.ndarray | None = None, fontsize: float = 9.5,
                 reliable_mask: np.ndarray | None = None) -> object:
    """counts, if given, is printed as a smaller second line under each cell's
    value -- the sample size (support) backing that cell's statistic. Cells where
    reliable_mask is False are fully blanked (no value, no text) -- this matrix
    should already be cropped to rows/cols that have at least one reliable cell."""
    plot_mat = mat.copy()
    if reliable_mask is not None:
        plot_mat[~reliable_mask] = np.nan

    masked = np.ma.masked_invalid(plot_mat)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.set_facecolor('#ececec')

    nr, nc = mat.shape
    for i in range(nr):
        for j in range(nc):
            if np.isnan(plot_mat[i, j]):
                continue
            v = mat[i, j]
            txt_color = 'white' if v > 0.6 * vmax else '#1a1a1a'
            if counts is None:
                ax.text(j, i, f"{v:{fmt}}", ha='center', va='center', fontsize=fontsize, color=txt_color)
                continue
            n_val = int(counts[i, j])
            ax.text(j, i - 0.16, f"{v:{fmt}}", ha='center', va='center', fontsize=fontsize, color=txt_color)
            sub_color = txt_color if v > 0.6 * vmax else '#555555'
            ax.text(j, i + 0.22, f"n={n_val}", ha='center', va='center',
                     fontsize=fontsize * 0.68, color=sub_color)

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=fontsize + 1)
    ax.set_yticklabels(rows, fontsize=fontsize + 1)
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    return im


# ── 1. GROUND-TRUTH CO-OCCURRENCE ──────────────────────────────────
def plot_cooccurrence(dataset: str, dataset_label: str) -> None:
    prob_path = FLIP_DIR / dataset / 'cooccurrence_conditional_prob.csv'
    if not prob_path.exists():
        print(f"[skip] no co-occurrence CSV for {dataset} (run evaluations/cooccurrence_stats.py)")
        return

    reliable = compute_reliable_pairs(dataset)
    reliable = reliable[reliable['reliable']] if not reliable.empty else reliable
    if reliable.empty:
        print(f"[skip] no reliable pairs (n>={MIN_PAIR_N}) for {dataset}")
        return
    reliable_classes = set(reliable['occluded_class']) | set(reliable['affected_class'])

    cond_prob = pd.read_csv(prob_path, index_col=0)
    counts_df = pd.read_csv(FLIP_DIR / dataset / 'cooccurrence_counts.csv', index_col=0)
    order = [c for c in prevalence_order(dataset) if c in cond_prob.index and c in reliable_classes]
    cond_prob = cond_prob.loc[order, order]
    counts_df = counts_df.loc[order, order]
    prevalence = pd.read_csv(FLIP_DIR / dataset / 'class_prevalence.csv', index_col=0)['prevalence'].loc[order]

    ridx = {c: i for i, c in enumerate(order)}
    reliable_mask = np.zeros((len(order), len(order)), dtype=bool)
    for a, b in zip(reliable['occluded_class'], reliable['affected_class']):
        if a in ridx and b in ridx:
            reliable_mask[ridx[a], ridx[b]] = True

    n = len(order)
    fig, (ax_bar, ax_mat) = plt.subplots(
        1, 2, figsize=(0.72 * n + 3.4, 0.6 * n + 2.2),
        gridspec_kw={'width_ratios': [1, 3.4], 'wspace': 0.08},
    )

    ax_bar.barh(range(n), prevalence.values[::-1], color='#4477AA', height=0.65)
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(order[::-1], fontsize=10.5)
    ax_bar.set_xlabel('Prevalence', fontsize=10)
    ax_bar.invert_xaxis()
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.tick_params(left=False)
    ax_bar.set_xlim(max(prevalence.values) * 1.15, 0)
    for i, v in enumerate(prevalence.values[::-1]):
        ax_bar.text(v * 1.05, i, f"{v:.0%}", va='center', ha='left', fontsize=8, color='#333333')

    # imshow row 0 is the top of the panel; order[0] (highest prevalence) must
    # also be the top bar, so rows go in `order` as-is (no reversal here).
    mat = cond_prob.values
    im = draw_matrix(ax_mat, mat, order, order, COOC_CMAP, vmax=1.0, counts=counts_df.values,
                      fontsize=9.5, reliable_mask=reliable_mask)
    ax_mat.set_yticklabels([])
    ax_mat.set_xlabel('Also present (B)', fontsize=11)

    cbar = fig.colorbar(im, ax=ax_mat, shrink=0.8, pad=0.02)
    cbar.set_label('P(B present | A present)', fontsize=10)

    fig.suptitle(
        f"Ground-truth disease co-occurrence — {dataset_label} (reliable pairs only)\n"
        f"rows/columns ordered by prevalence (bars, left) · n = images where both A and B are present "
        f"· only pairs with co-occurrence n ≥ {MIN_PAIR_N} and flip support n ≥ {MIN_PAIR_N} shown",
        fontsize=12, y=1.01,
    )

    out_stem = OUTPUT_DIR / f'cooccurrence_{dataset}'
    fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.15)
    fig.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Saved -> {out_stem.with_suffix('.pdf')}")


# ── 2. FLIP RATE AT FULL OCCLUSION (single, large panel) ──────────
def load_flip_summary(dataset: str) -> pd.DataFrame:
    path = FLIP_DIR / dataset / 'flip_rate_cross_disease_summary.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_flip_matrix(df: pd.DataFrame, strength: str, rows: list, cols: list) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df['strength'] == strength]
    nr, nc = len(rows), len(cols)
    rate = np.full((nr, nc), np.nan)
    count = np.zeros((nr, nc))
    ridx = {c: i for i, c in enumerate(rows)}
    cidx = {c: i for i, c in enumerate(cols)}
    for (a, b), g in sub.groupby(['occluded_class', 'affected_class']):
        if a not in ridx or b not in cidx:
            continue
        i, j = ridx[a], cidx[b]
        rate[i, j] = g['flip_rate'].mean()
        count[i, j] = g['n_baseline_positive'].sum()
    return rate, count


def plot_flip_heatmap(dataset: str, dataset_label: str) -> None:
    reliable = compute_reliable_pairs(dataset)
    reliable = reliable[reliable['reliable']] if not reliable.empty else reliable
    if reliable.empty:
        print(f"[skip] no reliable pairs (n>={MIN_PAIR_N}) for {dataset}")
        return
    df = load_flip_summary(dataset)

    order = prevalence_order(dataset)
    rows = [c for c in order if c in set(reliable['occluded_class'])]
    cols = [c for c in order if c in set(reliable['affected_class'])]
    nr, nc = len(rows), len(cols)

    rate, count = build_flip_matrix(df, FINAL_STRENGTH, rows, cols)
    ridx = {c: i for i, c in enumerate(rows)}
    cidx = {c: i for i, c in enumerate(cols)}
    reliable_mask = np.zeros((nr, nc), dtype=bool)
    for a, b in zip(reliable['occluded_class'], reliable['affected_class']):
        if a in ridx and b in cidx:
            reliable_mask[ridx[a], cidx[b]] = True

    fig, ax = plt.subplots(figsize=(0.68 * nc + 2.6, 0.62 * nr + 2.2))
    im = draw_matrix(ax, rate, rows, cols, FLIP_CMAP, vmax=1.0, counts=count, fontsize=10,
                      reliable_mask=reliable_mask)
    ax.set_xlabel('Affected class B (mention flips away)', fontsize=11)
    ax.set_ylabel('Occluded class A', fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Flip rate (baseline-positive mentions of B lost)', fontsize=10)

    fig.suptitle(
        f"Cross-disease flip rate at 100% occlusion — {dataset_label} (reliable pairs only)\n"
        f"mean over models, ordered by class prevalence · n = baseline-positive samples backing the cell "
        f"· only pairs with co-occurrence n ≥ {MIN_PAIR_N} and flip support n ≥ {MIN_PAIR_N} shown",
        fontsize=12, y=1.02,
    )

    out_stem = OUTPUT_DIR / f'flip_rate_heatmap_{dataset}'
    fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.15)
    fig.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Saved -> {out_stem.with_suffix('.pdf')}")


def export_support_table(dataset: str) -> None:
    """CSV of only the reliable (A,B) pairs (both gt_cooccurrence_n and
    flip_support_n >= MIN_PAIR_N), sorted by co-occurrence count."""
    merged = compute_reliable_pairs(dataset)
    if merged.empty:
        return
    reliable = merged[merged['reliable']].drop(columns='reliable')
    reliable = reliable.sort_values('gt_cooccurrence_n', ascending=False)
    out_path = FLIP_DIR / dataset / 'pair_support_table.csv'
    reliable.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(reliable)} reliable pairs)")


# ── 3. PER-PAIR OCCLUSION DYNAMICS (small multiples, one panel per A) ─
def pair_trend(df: pd.DataFrame, a: str, b: str) -> tuple[list, list]:
    """Mean flip rate over models at each occlusion strength for one (A,B) pair."""
    xs, means = [], []
    for strength in STRENGTHS:
        sub = df[
            (df['strength'] == strength) & (df['occluded_class'] == a) & (df['affected_class'] == b)
            & (df['n_baseline_positive'] > 0)
        ]
        vals = sub['flip_rate'].dropna().values
        if len(vals) == 0:
            continue
        xs.append(int(strength[1:]))
        means.append(float(np.mean(vals)))
    return xs, means


def plot_occlusion_dynamics(dataset: str, dataset_label: str) -> None:
    """For every reliable (A,B) pair, how the flip rate rises with occlusion
    strength -- i.e. the full dose-response curve behind each cell of the
    single-strength heatmap, faceted by occluded class A (one panel per A,
    one colored line per affected class B)."""
    reliable = compute_reliable_pairs(dataset)
    reliable = reliable[reliable['reliable']] if not reliable.empty else reliable
    if reliable.empty:
        print(f"[skip] no reliable pairs (n>={MIN_PAIR_N}) for {dataset}")
        return
    df = load_flip_summary(dataset)

    order = prevalence_order(dataset)
    a_list = [c for c in order if c in set(reliable['occluded_class'])]
    pairs_by_a = {a: sorted(reliable.loc[reliable['occluded_class'] == a, 'affected_class'],
                             key=lambda b: order.index(b) if b in order else 999)
                  for a in a_list}

    ncols = min(3, len(a_list))
    nrows = int(np.ceil(len(a_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows),
                              sharex=True, sharey=True, squeeze=False)

    used_classes: set = set()
    for k, a in enumerate(a_list):
        ax = axes[k // ncols, k % ncols]
        ax.set_facecolor('#f7f7f7')
        for b in pairs_by_a[a]:
            xs, means = pair_trend(df, a, b)
            if not xs:
                continue
            ax.plot(xs, means, marker='o', markersize=4, linewidth=1.8,
                     color=CLASS_COLOR.get(b, '#333333'), zorder=3)
            used_classes.add(b)
        ax.set_title(f"A = {a}", fontsize=10.5)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.7, color='#888888', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused trailing panels, but track the last *active* row per column so
    # x-axis labels land on the true bottom of each column (which isn't always
    # the literal last grid row, since the last row can be partially filled).
    last_row_of_col = {c: 0 for c in range(ncols)}
    for k in range(nrows * ncols):
        r, c = k // ncols, k % ncols
        if k < len(a_list):
            last_row_of_col[c] = r
        else:
            axes[r, c].axis('off')

    for ax in axes[:, 0]:
        ax.set_ylabel('Flip rate', fontsize=10)
    for c in range(ncols):
        axes[last_row_of_col[c], c].set_xlabel('Occlusion strength of A (%)', fontsize=10)
        axes[last_row_of_col[c], c].tick_params(labelbottom=True)

    # One shared legend outside the grid, fixed color per class (never re-cycled),
    # ordered consistently with the ALL_CLASSES / CLASS_COLOR assignment.
    legend_classes = [c for c in ALL_CLASSES if c in used_classes]
    handles = [plt.Line2D([0], [0], color=CLASS_COLOR[c], marker='o', markersize=5, linewidth=1.8)
               for c in legend_classes]
    fig.legend(handles, legend_classes, title='Affected class B', loc='center left',
               bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9.5, title_fontsize=10,
               handlelength=1.6, labelspacing=0.6)

    fig.suptitle(
        f"Occlusion dynamics of reliable cross-disease pairs — {dataset_label}\n"
        f"each panel = one occluded class A, each line = one co-occurring affected class B "
        f"(n ≥ {MIN_PAIR_N} both ways)",
        fontsize=12.5, y=1.0 + 0.012 * nrows,
    )

    out_stem = OUTPUT_DIR / f'flip_rate_dynamics_{dataset}'
    fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.15)
    fig.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"Saved -> {out_stem.with_suffix('.pdf')}")


# ── 4. DOSE RESPONSE (unchanged design, already read well) ────────
DS_COLOR = {'padchest-gr': '#EE6677', 'mimic-cxr-jpg': '#4477AA'}


def dose_response_series(df: pd.DataFrame) -> tuple[list, list, list]:
    from scipy import stats as scipy_stats

    xs, means, cis = [], [], []
    for strength in STRENGTHS:
        sub = df[(df['strength'] == strength) & (df['n_baseline_positive'] > 0)]
        vals = sub['flip_rate'].dropna().values
        if len(vals) == 0:
            continue
        xs.append(int(strength[1:]))
        m = float(np.mean(vals))
        h = float(scipy_stats.t.ppf(0.975, df=len(vals) - 1) * scipy_stats.sem(vals)) if len(vals) > 1 else np.nan
        means.append(m)
        cis.append(h)
    return xs, means, cis


def plot_dose_response() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_facecolor('#f7f7f7')

    any_data = False
    for ds_key, ds_label in DATASETS:
        df = load_flip_summary(ds_key)
        if df.empty:
            continue
        xs, means, cis = dose_response_series(df)
        if not xs:
            continue
        any_data = True
        color = DS_COLOR[ds_key]
        means = np.array(means)
        cis = np.array(cis)
        ax.plot(xs, means, marker='o', color=color, linewidth=2, markersize=5, label=ds_label, zorder=3)
        lower = means - np.nan_to_num(cis)
        upper = means + np.nan_to_num(cis)
        ax.fill_between(xs, lower, upper, color=color, alpha=0.18, zorder=2)

    if not any_data:
        print("[skip] no data for dose-response plot")
        plt.close(fig)
        return

    ax.set_xlabel('Occlusion strength of co-occurring class A (%)', fontsize=12)
    ax.set_ylabel('Mean cross-disease flip rate', fontsize=12)
    ax.set_title('Occluding one disease region flips mentions of\nother, co-occurring diseases', fontsize=12)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.7, color='#888888', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#dddddd', fontsize=10)

    out_stem = OUTPUT_DIR / 'flip_rate_dose_response'
    fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.1)
    fig.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved -> {out_stem.with_suffix('.pdf')}")


if __name__ == '__main__':
    for ds_key, ds_label in DATASETS:
        plot_cooccurrence(ds_key, ds_label)
        plot_flip_heatmap(ds_key, ds_label)
        plot_occlusion_dynamics(ds_key, ds_label)
        export_support_table(ds_key)
    plot_dose_response()
