# CLAUDE.md — Project Context

## Project Overview

Benchmark framework for evaluating **vision-language models (VLMs)** on **chest X-ray radiology report generation** using the **PadChest-GR** dataset. The framework supports 9 models, multiple occlusion experiments (to study shortcut learning), and comprehensive medical NLP evaluation metrics.

## Tech Stack

- **Python 3.11.5** (module: `Python/3.11.5-GCCcore-13.2.0` on NAISS Alvis cluster)
- **PyTorch 2.1+** with **bfloat16** precision
- **HuggingFace Transformers 4.40+**
- **SLURM** for job scheduling (NAISS project: `NAISS2025-5-662`)

## Repository Layout

```
src/benchmark/           # Core benchmark: CLI, model registry, inference pipeline
  models/                # 9 model implementations (one file per model family)
  cli.py                 # Entry point: python -m src.benchmark.cli
  hf_runner.py           # HuggingFace inference orchestration
  config.py              # Configuration dataclass
  prompts.py             # Model-specific prompt templates
  preprocess.py          # Image preprocessing & occlusion strategies
  datasets/              # PadChest-GR dataset loader
  io.py                  # I/O utilities
src/analysis/            # Dataset analysis & visualization scripts
evaluations/             # Evaluation pipeline
  run_eval.py            # Main evaluation script (metrics: BLEU, ROUGE, RadGraph-F1, CheXBERT)
  green_evaluation.py    # GREEN metric wrapper
  rrg_eval/              # Custom metric implementations
scripts/baseline/        # SLURM batch scripts (one per model + submit_all.sh)
  padchest-gr/           # PadChest-GR specific SLURM scripts
data/padchest-gr/        # Dataset directory (images + JSON annotations)
outputs/                 # Inference results (JSONL per model/experiment)
results/                 # Evaluation results (CSV/Excel)
logs/                    # SLURM job logs
```

## Git Submodules

- `Libra/` — LLaVA-based architecture for Libra & LLaVA-Rad models
- `RaDialog-interactive-radiology-report-generation/` — RaDialog model
- `GREEN/` — Stanford AIMI evaluation metrics

Initialize with: `git submodule update --init --recursive`

## Virtual Environments (Multiple due to dependency conflicts)

| Venv | Models | Requirements File |
|------|--------|-------------------|
| `.venv_RRG` | MedGemma, MAIRA-2, CXRMateED | `requirements_rrg.txt` |
| `.venv_nv` | NV-Reason-CXR, CheXOne | `requirements_reasoning.txt` |
| `.SC_Libra_venv` | Libra, LLaVA-Rad | `requirements_llava-libra.txt` |
| `.venv_chexagent` | CheXagent | `requirements_chexagent.txt` |
| `.radialog_venv` | RaDialog | `requirements_radialog.txt` |
| `.venv_eval` | Evaluation metrics | `requirements_eval.txt` |

## Key Commands

```bash
# Run model inference
python -m src.benchmark.cli --model <key> --data-json <json> --data <img_dir> --experiment baseline --device cuda:0 --dtype bfloat16

# Run evaluation
python evaluations/run_eval.py --filepath <output.json> --output-mode per-experiment

# Submit all SLURM jobs
./scripts/baseline/submit_all.sh

# Submit single model
sbatch scripts/baseline/padchest-gr/run_medgemma.sh
```

## Supported Models (keys for --model flag)

`medgemma`, `maira-2`, `cxrmateed`, `nv-reason-cxr-3b`, `chexone`, `libra`, `llavarad`, `chexagent`, `radialog`

## Experiment Types

- **baseline** — Full image, no modification
- **oco** — Object Class Occlusion (oco_p25, oco_p50, oco_p75, oco_p100)
- **roco** — Random Object Class Occlusion (same strength levels)
- **noise** — Random/correlated noise injection

## Environment Variables (required)

```bash
export HF_TOKEN="<huggingface_token>"
export HF_HOME="${PWD}/.models_cache"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

## Branch Structure

- **main** — Stable release
- **padchest-filippo** — Active development (current), PadChest-GR + evaluation pipeline
- **chexlocalize** — Evaluation infrastructure (GREEN, metrics)
- **padchest-gr** — Dataset organization

## Code Conventions

- Models are registered in `src/benchmark/models/__init__.py` via a `MODEL_REGISTRY` dict
- Each model file exports a `generate()` function following a consistent interface
- Output format is JSONL with fields: `model`, `model_id`, `prompt`, `image_path`, `generated_text`, `metadata`
- SLURM scripts follow the naming pattern `run_<model>.sh`
- Random seed is fixed to 3 for reproducibility (`numpy`/`random`)

## Important Notes

- Always activate the correct venv for each model family before running
- The `.models_cache/` directory caches HuggingFace model weights — do not delete
- `data/`, `outputs/`, `results/`, `logs/`, and venv directories are gitignored
- Image preprocessing uses CXR-specific min-max normalization (not ImageNet stats)
