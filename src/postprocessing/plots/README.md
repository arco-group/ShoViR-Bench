# Postprocessing & Plotting

Scripts for visualising evaluation results. All scripts live in `src/postprocessing/` and are runnable as Python modules from the project root.

---

## `bar_plot_oco.py` -- Baseline vs OCO / DOCO / RO Bar Chart

Compares model performance at maximum occlusion (`p=100`) across the three experiment types (OCO, DOCO, RO) against the unperturbed baseline (`p=0`). Metric is the mean CheXBERT F1 averaged over all present condition classes (NaN-safe).

### Quick start

```bash
python src/postprocessing/padchest/plots/bar_plot_oco.py
```

### Input

Reads directly from the project `results/` directory (no arguments needed):

| Path | Used as |
|------|---------|
| `results/oco/p00/results.csv` | Baseline (unperturbed) |
| `results/oco/p100/results.csv` | OCO at full occlusion |
| `results/doco/p100/results.csv` | DOCO at full occlusion |
| `results/ro/p100/results.csv` | RO at full occlusion |

### Output

Saved to `results/plots/`:
- `barplot.pdf`
- `barplot.png`

---

## `line_plot_p.py` -- Performance vs Perturbation Ratio

Shows how mean CheXBERT F1 evolves as the perturbation ratio `p` increases from 0 % to 100 %, with one subplot per experiment type (OCO, DOCO, RO) and one line per model.

### Quick start

```bash
python src/postprocessing/padchest/plots/line_plot_p.py
```

### Input

Reads directly from the project `results/` directory (no arguments needed):

| Path pattern | Description |
|---|---|
| `results/{oco,doco,ro}/p{00,20,40,60,80,100}/results.csv` | Per-class CheXBERT F1 at each perturbation level |

### Output

Saved to `results/plots/`:
- `lineplot.pdf`
- `lineplot.png`

---

## `plot_ro.py` -- Random Occlusion Performance Curves

Reads per-experiment results CSVs from `results/ro/pXX/results.csv` and produces one line plot per metric, with models as separate lines and occlusion percentage on the x-axis.

### Quick start

```bash
# Default metrics (Micro-F1-5, Macro-F1-5, F1-RadGraph)
python -m src.postprocessing.plot_ro

# Custom metrics
python -m src.postprocessing.plot_ro --metrics Micro-F1-5 F1-RadGraph BLEU-4

# Custom input/output directories
python -m src.postprocessing.plot_ro --results-dir results/ro --out-dir figures/ro
```

### CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--results-dir` | `results/ro` | Directory containing `pXX/` subdirectories with `results.csv` files |
| `--out-dir` | `figures/ro` | Output directory for PDF plots |
| `--metrics` | `Micro-F1-5 Macro-F1-5 F1-RadGraph` | Space-separated list of metrics to plot |

### Available metrics

Any column present in the results CSVs can be plotted. Common choices:

| Metric | Description |
|--------|-------------|
| `Micro-F1-5` | Micro F1 over top-5 diseases (uncertain=negative) |
| `Micro-F1-5+` | Micro F1 over top-5 diseases (uncertain=positive) |
| `Macro-F1-5` | Macro F1 over top-5 diseases (uncertain=negative) |
| `Micro-F1-14` | Micro F1 over all 14 conditions (uncertain=negative) |
| `Macro-F1-14` | Macro F1 over all 14 conditions (uncertain=negative) |
| `F1-RadGraph` | RadGraph-based factual F1 |
| `BLEU-1` | BLEU-1 |
| `BLEU-4` | BLEU-4 |
| `ROUGE-L` | ROUGE-L |

### Output

One PDF per metric saved to `<out-dir>/<metric>.pdf` (e.g. `figures/ro/Micro_F1_5.pdf`).

### Key functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_ro_results` | `(results_dir) -> DataFrame` | Load all `pXX/results.csv` files into a single DataFrame with columns `model`, `percentage`, and one column per metric |
| `plot_metric` | `(df, metric, out_dir) -> Path` | Create and save a single line plot for one metric |
| `plot_all` | `(df, metrics, out_dir) -> List[Path]` | Plot all requested metrics, returns list of output paths |
| `shorten_model_name` | `(raw) -> str` | Convert raw CSV index (e.g. `google__medgemma`) to readable label (e.g. `MedGemma`) |

### Expected directory structure

```
results/ro/
  p20/
    results.csv      # or padchest-gr/*.csv (per-file mode)
  p40/
    results.csv
  p60/
    ...
```
