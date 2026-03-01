# Compute Efficiency Analysis

Measures **FLOPs**, **MACs**, **parameter count**, and **inference throughput** for each model,
then plots the trade-off against clinical performance metrics (baseline, PadChest-GR).

## Overview

The analysis runs in two stages:

```
Stage 1 — per-model stats (one SLURM job per model, inside the correct venv)
  src/postprocessing/compute_model_stats.py
  → results/compute_stats/<model_key>.json

Stage 2 — plots (any venv with matplotlib + pandas)
  src/postprocessing/plots/compute_efficiency.py
  → results/plots/compute_efficiency_params.{pdf,png}
  → results/plots/compute_efficiency_gflops.{pdf,png}
  → results/plots/compute_efficiency_gmacs.{pdf,png}
  → results/plots/compute_efficiency_throughput.{pdf,png}
```

---

## Quick test (no model loading, no SLURM)

Verify everything works with synthetic dummy data:

```bash
source .venv_eval/bin/activate
python -m src.postprocessing.plots.compute_efficiency --test
```

Plots are written to `results/plots/compute_efficiency_*.{pdf,png}`.

---

## Stage 1 — Collect per-model stats

Each model must be profiled **inside its own venv** because of dependency conflicts.
The SLURM script `run_compute_stats.sh` handles this automatically.

### Submit a single model

```bash
MODEL=medgemma bash scripts/compute_stats/run_compute_stats.sh
```

### Submit all models (loop)

```bash
for m in medgemma maira-2 cxrmateed chexagent libra llavarad radialog nv-reason-cxr-3b chexone; do
    MODEL=$m bash scripts/compute_stats/run_compute_stats.sh
done
```

### Model → venv mapping

| Model key          | Venv                 | GPU          |
|--------------------|----------------------|--------------|
| `medgemma`         | `.venv_RRG`          | A40 (30 min) |
| `maira-2`          | `.venv_RRG`          | A40 (30 min) |
| `cxrmateed`        | `.venv_RRG`          | A40 (30 min) |
| `nv-reason-cxr-3b` | `.venv_nv`           | A40 (30 min) |
| `chexone`          | `.venv_nv`           | A40 (30 min) |
| `libra`            | `.SC_Libra_venv`     | A40 (30 min) |
| `llavarad`         | `.SC_Libra_venv`     | A40 (30 min) |
| `chexagent`        | `.venv_chexagent`    | A40 (30 min) |
| `radialog`         | `.radialog_venv`     | A40 (30 min) |

### Run manually (no SLURM)

```bash
# Activate the model's own venv first, e.g.:
source .venv_RRG/bin/activate
export HF_HOME="${PWD}/.models_cache"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python -m src.postprocessing.compute_model_stats \
    --model medgemma \
    --n-images 5 \          # synthetic forward passes for timing
    --device cuda:0 \
    --dtype bfloat16

# Parameters + FLOPs only (no generate() timing, CPU is enough):
python -m src.postprocessing.compute_model_stats \
    --model cxrmateed --n-images 0 --device cpu
```

### CLI options

| Flag           | Default     | Description                                          |
|----------------|-------------|------------------------------------------------------|
| `--model`      | *(required)*| Model key (see table above)                          |
| `--n-images`   | `5`         | Synthetic images for `generate()` timing. `0` = skip timing, FLOPs + params only |
| `--device`     | `cuda:0`    | Torch device                                         |
| `--dtype`      | `bfloat16`  | Torch dtype (`float16`, `bfloat16`, `float32`)       |
| `--cache-dir`  | `.models_cache` | HuggingFace weights cache                        |
| `--output-dir` | `results/compute_stats/` | Where to write the JSON              |

### Output JSON format

`results/compute_stats/<model_key>.json`:

```json
{
  "model_key":          "medgemma",
  "model_id":           "google/medgemma-1.5-4b-it",
  "display_name":       "MedGemma",
  "n_params_total":     4300000000,
  "n_params_trainable": 4300000000,
  "n_params_total_B":   4.3,
  "device":             "cuda:0",
  "dtype":              "bfloat16",
  "gflops":             1234.5,
  "gmacs":              617.2,
  "flops_method":       "FlopCounterMode",
  "n_images_tested":    5,
  "mean_inference_time_s": 3.12,
  "std_inference_time_s":  0.08,
  "throughput_img_per_s":  0.32
}
```

#### FLOPs methods (in priority order)

| `flops_method`        | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `FlopCounterMode`     | PyTorch built-in (≥ 2.0). Exact counts for matmul/conv ops. |
| `calflops`            | `pip install calflops`. Comprehensive HF transformer support.|
| `2NL_approx(L=<len>)` | Fallback: `2 × N_params × seq_len` (Chinchilla approximation). Always works. |

> **Note:** FLOPs are measured for the **prefill forward pass** (`model(**inputs)`),
> not the full autoregressive generation loop. MACs = FLOPs / 2.

---

## Stage 2 — Generate plots

Once at least some JSONs exist in `results/compute_stats/`, run:

```bash
source .venv_eval/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python -m src.postprocessing.plots.compute_efficiency
```

For models without a JSON, the plot falls back to known parameter counts from model cards
(no FLOPs/MACs plotted for those models).

### Output plots

| File                                   | x-axis               | y-axis (each panel)       |
|----------------------------------------|----------------------|---------------------------|
| `compute_efficiency_params.{pdf,png}`  | Parameters (B)       | Clinical metric           |
| `compute_efficiency_gflops.{pdf,png}`  | GFLOPs (prefill)     | Clinical metric           |
| `compute_efficiency_gmacs.{pdf,png}`   | GMACs (prefill)      | Clinical metric           |
| `compute_efficiency_throughput.{pdf,png}` | Throughput (img/s) | Clinical metric           |

Each plot has **5 panels** (one per clinical metric):
`F1-RadGraph`, `CheXBERT Micro-F1 (14 cls)`, `CheXBERT Macro-F1 (14 cls)`,
`CheXBERT Micro-F1 (5 cls)`, `CheXBERT Macro-F1 (5 cls)`.

### CLI options

| Flag         | Default               | Description                             |
|--------------|-----------------------|-----------------------------------------|
| `--test`     | off                   | Use synthetic dummy data (no files needed) |
| `--out-stem` | `compute_efficiency`  | Output filename stem (without extension) |

---

## Monitor jobs

```bash
# Check running jobs
squeue -u $USER

# Follow logs
tail -f logs/compute_stats_<model>_<jobid>.out

# Check which JSONs have been produced
ls -lh results/compute_stats/
```
