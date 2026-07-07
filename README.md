# Shortcut Learning in Radiology Report Generation

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![Platform](https://img.shields.io/badge/platform-SLURM%20%2F%20NAISS%20Alvis-lightgrey)
![License](https://img.shields.io/badge/license-Unspecified-lightgrey)

A benchmark framework for evaluating **vision-language models (VLMs)** on **chest X-ray radiology report generation**, built around the **PadChest-GR** and **MIMIC-CXR** datasets. It supports 9 models and a suite of occlusion experiments designed to probe shortcut learning — i.e., whether a model's report quality depends on regions of the image it shouldn't need.

[Overview](#overview) · [Repository Structure](#repository-structure) · [Setup](#setup) · [Supported Models](#supported-models) · [Usage](#usage) · [Experiments](#experiments) · [Evaluation Metrics](#evaluation-metrics) · [Output Format](#output-format) · [Troubleshooting](#troubleshooting) · [Citation](#citation)

* * *

## Overview

Radiology report generation models can appear accurate while actually relying on spurious correlations (shortcuts) rather than the clinical findings a radiologist would use. This framework operationalizes that question by:

1. Running a fixed set of VLMs over the **same** images and prompts (`baseline`),
2. Perturbing the input images in controlled ways — occluding object classes, occluding random regions, or injecting noise — while holding the model and prompt fixed,
3. Scoring every condition with the same suite of report-generation metrics,

so that changes in score can be attributed to the perturbation rather than to model or prompting differences.

* * *

## Repository Structure

```
src/benchmark/               # Core benchmark: CLI, model registry, inference pipeline
├── cli.py                   # Entry point: python -m src.benchmark.cli
├── hf_runner.py              # HuggingFace inference orchestration
├── config.py                 # Configuration dataclass
├── prompts.py                 # Model-specific prompt templates
├── preprocess.py              # Image preprocessing & occlusion strategies
├── models/                    # One implementation file per model family
├── datasets/                  # PadChest-GR / MIMIC-CXR dataset loaders
└── io.py                      # I/O utilities

src/analysis/                 # Dataset analysis & visualization scripts
evaluations/                  # Legacy in-repo evaluation pipeline (being replaced by radscore, see below)
scripts/                      # SLURM batch scripts, one folder per experiment family
├── baseline/                  # Full image, no modification
├── oco/, ro/                   # (Random) Object Class Occlusion sweeps
├── doco/                       # Directed occlusion variants
└── all_noise_mean/              # Noise-injection experiments
data/                         # Dataset directory (images + JSON annotations) — gitignored
outputs/                      # Inference results, JSONL per model/experiment — gitignored
results/                      # Evaluation results (CSV/Excel) — gitignored
logs/                         # SLURM job logs — gitignored
```

**Git submodules:**
- `Libra/` — LLaVA-based architecture backing Libra & LLaVA-Rad
- `RaDialog-interactive-radiology-report-generation/` — RaDialog model
- `GREEN/` — Stanford AIMI evaluation metric (legacy; superseded by [radscore](https://github.com/fruffini/radscore))

Initialize with:
```bash
git submodule update --init --recursive
```

* * *

## Setup

Requires **Python 3.11.5**. On the NAISS Alvis cluster:

```bash
module load Python/3.11.5-GCCcore-13.2.0
```

Because model families pull in conflicting dependencies, each family gets its own virtual environment:

| Venv | Models | Requirements file |
|------|--------|--------------------|
| `.venv_RRG` | MedGemma, MAIRA-2, CXRMateED | `requirements_rrg.txt` |
| `.venv_nv` | NV-Reason-CXR, CheXOne | `requirements_reasoning.txt` |
| `.SC_Libra_venv` | Libra, LLaVA-Rad | `requirements_llava-libra.txt` |
| `.venv_chexagent` | CheXagent | `requirements_chexagent.txt` |
| `.radialog_venv` | RaDialog | `requirements_radialog.txt` |
| `.venv_eval` | Metrics (legacy path — see [Evaluation Metrics](#evaluation-metrics)) | `requirements_eval.txt` |

Example for one family:
```bash
python3.11 -m venv .venv_RRG
source .venv_RRG/bin/activate
pip install -r requirements_rrg.txt
```

Required environment variables:
```bash
export HF_TOKEN="<huggingface_token>"
export HF_HOME="${PWD}/.models_cache"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

* * *

## Supported Models

All models run in **bfloat16** precision.

| Model | Key | Virtual env |
|-------|-----|-------------|
| MedGemma | `medgemma` | `.venv_RRG` |
| MAIRA-2 | `maira-2` | `.venv_RRG` |
| CXRMateED | `cxrmateed` | `.venv_RRG` |
| NV-Reason-CXR | `nv-reason-cxr-3b` | `.venv_nv` |
| CheXOne | `chexone` | `.venv_nv` |
| Libra | `libra` | `.SC_Libra_venv` |
| LLaVA-Rad | `llavarad` | `.SC_Libra_venv` |
| CheXagent | `chexagent` | `.venv_chexagent` |
| RaDialog | `radialog` | `.radialog_venv` |

* * *

## Usage

### Run a single model

```bash
python -m src.benchmark.cli \
    --model medgemma \
    --data-json data/padchest-gr/chexpert-by-label/verified_samples.json \
    --data "data/padchest-gr/BIMCV-Padchest-GR/PadChest_GR_images" \
    --experiment baseline \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 10
```

Key flags:
- `--model` — model key from the registry (required)
- `--data-json` / `--data` — dataset annotation JSON and image directory (required)
- `--experiment` — experiment name, e.g. `baseline`, `oco_p50`, `roco_p75`, `ro_p50` (required)
- `--output-dir` — results directory (default: `outputs`)
- `--single-prompt-baseline` — for `--experiment baseline`, share one prompt across all models (writes to `outputs/baseline_SP/`)
- `--seed` — random seed for OCO/ROCO region selection (default: `3`)

### Submit SLURM jobs

```bash
./scripts/baseline/submit_all.sh          # all models, baseline experiment
sbatch scripts/baseline/padchest-gr/run_medgemma.sh   # a single model
```

Monitor with `squeue -u $USER` and `tail -f logs/**/*.out`.

* * *

## Experiments

| Experiment | Description |
|------------|--------------|
| `baseline` | Full image, no modification |
| `oco` | Object Class Occlusion — occludes a specific finding-relevant class (`oco_p25` … `oco_p100`) |
| `roco` / `ro` | Random(-ized) Object Class Occlusion at matching strength levels |
| noise | Random or correlated noise injection |

The occlusion strength (e.g. `p50`) controls how much of the target region is masked, letting you trace how report quality degrades as evidence is removed — and whether that degradation tracks the *right* evidence.

* * *

## Evaluation Metrics

Report-generation metrics (BLEU, ROUGE, BERTScore, F1-RadGraph, CheXbert F1, GREEN) are computed with **[radscore](https://github.com/fruffini/radscore)**, a standalone package maintained alongside this project. It replaces the metric code that used to live in `evaluations/` in this repository.

### Install

```bash
git clone --recurse-submodules https://github.com/fruffini/radscore.git
cd radscore
python3.11 -m venv radscore-env
source radscore-env/bin/activate
python -m pip install -U pip
python -m pip install -e .

# Optional: GREEN metric (LLM-based clinical error grading)
git submodule update --init --recursive
python -m pip install -e third_party/GREEN --ignore-requires-python
```

### Run

```bash
# Default scorers, one score per file
radscore --filepath outputs/baseline/padchest-gr/medgemma.jsonl --output-mode per-file

# Pick specific scorers and get bootstrap confidence intervals
radscore --filepath outputs/baseline/padchest-gr/medgemma.jsonl \
         --scorers CheXbert,F1-RadGraph,ROUGE-L \
         --bootstrap-ci \
         --output-mode per-experiment

# Include GREEN
radscore --filepath outputs/baseline/padchest-gr/medgemma.jsonl --compute-green
```

`radscore` expects a JSON list with `prediction` / `reference` fields (plus an optional 14-element CheXbert `label` vector and `target_category` for per-category breakdowns — see the [radscore README](https://github.com/fruffini/radscore) for the exact schema and the Python API).

> The legacy `evaluations/` pipeline in this repo is kept only for reference during the migration and will be removed from version control once `radscore` fully covers our reporting needs.

* * *

## Output Format

Inference results are written as JSONL to `outputs/<experiment>/<dataset>/<model>.jsonl`, one line per sample:

```json
{
  "model": "chexone",
  "model_id": "StanfordAIMI/CheXOne",
  "prompt": "Analyze this chest X-ray...",
  "image_path": "data/padchest-gr/.../image.png",
  "generated_text": "The chest X-ray shows...",
  "metadata": {}
}
```

* * *

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA out of memory | Lower `--num-images`, keep `--dtype bfloat16`, request a bigger-memory GPU in the SLURM script |
| Import errors | Confirm the correct venv is active and matches the model family table above |
| Model download fails | Set `HF_TOKEN`, check connectivity, verify `.models_cache` permissions |
| Submodule not found | `git submodule update --init --recursive`; check with `git submodule status` |

* * *

## Citation

```bibtex
@software{shortcut_rrg_benchmark,
  title  = {Shortcut Learning in Radiology Report Generation},
  author = {Ruffini, Filippo},
  url    = {https://github.com/fruffini/Shortcut-Learning-RRG}
}
```

## License

License to be specified.

## Contact

Filippo Ruffini — filippo.ruffini@unicampus.it
