# SHOVIR

<div align="center">

<p align="center">
  <a href="figures/ro/F1_RadGraph.pdf"><img src="figures/ro/F1_RadGraph.pdf" alt="F1-RadGraph vs. Random Occlusion Level across 8 VLMs" width="820"/></a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2606.30201-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.30201)
[![Paper PDF](https://img.shields.io/badge/Read%20the-Paper-1f6feb?style=for-the-badge&logo=readthedocs&logoColor=white)](https://arxiv.org/pdf/2606.30201)
[![Python](https://img.shields.io/badge/Python-3.11.5-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![SLURM](https://img.shields.io/badge/Scheduler-SLURM-6aa84f?style=for-the-badge)](https://slurm.schedmd.com/)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey?style=for-the-badge)](#license)

**A benchmark for evaluating whether vision-language models genuinely ground their diagnostic statements in visual evidence during chest X-ray report generation — rather than exploiting learned shortcuts that standard metrics reward but radiologists would not trust.**

**Filippo Ruffini** · Marco Salmé · Rosa Sicilia · Valerio Guarrasi · **Paolo Soda**

[📄 Paper](https://arxiv.org/abs/2606.30201) ·
[🧩 Overview](#overview) ·
[⚙️ Setup](#environment-setup) ·
[🚀 Usage](#usage) ·
[📊 Results](#results) ·
[📚 Citation](#citation)

</div>

---

## Overview

This repository accompanies the paper *"SHOVIR: A Benchmark for Evaluating Vision Shortcut Learning in Radiology Report Generation"* ([arXiv:2606.30201](https://arxiv.org/abs/2606.30201)).

Standard radiology report generation (RRG) metrics (BLEU, ROUGE, RadGraph-F1, CheXBERT) reward textual and clinical-entity overlap with a reference report, but say nothing about *where in the image* a model looked before writing a finding. A model can produce a fluent, clinically plausible report while never actually attending to the pathology it claims to describe — a **vision shortcut**. SHOVIR isolates this failure mode by extending two chest X-ray datasets (**PadChest-GR**, **MIMIC-CXR**) with spatial disease annotations and applying targeted image perturbations that separate two shortcut behaviors:

- **Direct shortcuts** — a finding persists in the generated report even after the visual evidence for it has been removed from the image (occluding the target pathology's own region should suppress the finding; if it doesn't, the model wasn't looking at it).
- **Contextual shortcuts** — detection of a finding fails after occluding *co-occurring, related* pathologies, even though the target region itself is left intact (the model was relying on correlated context rather than the region in question).

Across **eight state-of-the-art VLMs** (extended here to **11** in this repository) — CXRMate-ED, CheXagent, LLaVA-Rad, Libra, MAIRA-2, MedGemma, NV-Reason-CXR, RaDialog, plus Gemini and GPT-5.4 baselines — we find substantial variation in shortcut behavior across architectures and datasets, and that report fluency/quality does **not** imply strong spatial grounding: some of the highest-scoring models on standard metrics show the flattest response to occlusion, i.e. their findings are largely occlusion-invariant.

> **Dataset note.** PadChest-GR and MIMIC-CXR are governed by their own data-use agreements and are **not redistributed** with this repository. `data/`, `outputs/`, `results/`, `logs/`, and all venv directories are gitignored. The repository contains the full inference, occlusion, and evaluation pipeline, along with placeholder paths that downstream users can point at their own copy of either dataset.

---

## Repository layout

```
src/benchmark/           # Core benchmark: CLI, model registry, inference pipeline
  models/                # 11 model implementations (one file per model family)
  cli.py                 # Entry point: python -m src.benchmark.cli
  hf_runner.py           # HuggingFace inference orchestration
  config.py              # Configuration dataclass
  prompts.py             # Model-specific prompt templates
  preprocess.py          # Image preprocessing & occlusion strategies (oco/doco/roco/ro/noise)
  datasets/              # PadChest-GR / MIMIC-CXR dataset loaders
  io.py                  # I/O utilities
src/analysis/            # Dataset analysis & visualization scripts
evaluations/             # Evaluation pipeline
  run_eval.py             # Main evaluation script (BLEU, ROUGE, RadGraph-F1, CheXBERT)
  run_eval_weighted.py    # Weighted/aggregated evaluation across experiments
  green_evaluation.py     # GREEN metric wrapper
  rrg_eval/               # Custom metric implementations (incl. F1-RadGraph)
scripts/                 # SLURM batch scripts, one subtree per experiment family
  baseline/                # Full-image, no occlusion
  oco/, doco/, ro/         # Object-class / co-occurrence / random occlusion sweeps
  all_noise_mean/          # Correlated-noise perturbation sweep
figures/ro/                # Random-occlusion result figures (see Results)
Libra/                   # Git submodule: LLaVA-based backbone for Libra & LLaVA-Rad
RaDialog-interactive-radiology-report-generation/  # Git submodule: RaDialog model
GREEN/                   # Git submodule: Stanford AIMI GREEN evaluation metric
data/                    # Dataset directory (gitignored)
outputs/                 # Inference results, JSONL per model/experiment (gitignored)
results/                 # Evaluation results, CSV/Excel (gitignored)
logs/                    # SLURM job logs (gitignored)
```

Initialize submodules with:
```bash
git submodule update --init --recursive
```

---

## Environment Setup

Requires **Python 3.11.5**. On the NAISS Alvis cluster:

```bash
module load Python/3.11.5-GCCcore-13.2.0
```

Multiple virtual environments are used to avoid dependency conflicts between model families:

| Venv | Models | Requirements file |
|------|--------|--------------------|
| `.venv_RRG` | MedGemma, MAIRA-2, CXRMateED | `requirements_rrg.txt` |
| `.venv_nv` | NV-Reason-CXR, CheXOne | `requirements_reasoning.txt` |
| `.SC_Libra_venv` | Libra, LLaVA-Rad | `requirements_llava-libra.txt` |
| `.venv_chexagent` | CheXagent | `requirements_chexagent.txt` |
| `.radialog_venv` | RaDialog | `requirements_radialog.txt` |
| `.venv_eval` | Evaluation metrics | `requirements_eval.txt` |

```bash
python3.11 -m venv .venv_RRG
source .venv_RRG/bin/activate
pip install --upgrade pip
pip install -r requirements_rrg.txt
```

Gemini and GPT-5.4 are called via API and do not need a local model venv, but do need their respective API keys exported.

### Required environment variables

```bash
export HF_TOKEN="<huggingface_token>"
export HF_HOME="${PWD}/.models_cache"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

---

## Supported Models

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
| Gemini | `gemini` | API |
| GPT-5.4 | `gpt54` | API |

## Experiment types

| Key | Meaning |
|-----|---------|
| `baseline` | Full image, no modification |
| `oco_pN` | Object Class Occlusion — occlude the target pathology's own region at strength `N` (`p25`/`p50`/`p75`/`p100`) → probes **direct shortcuts** |
| `doco_pN` | Co-occurrence Object Class Occlusion — occlude regions of pathologies correlated with the target, target region left intact → probes **contextual shortcuts** |
| `roco_pN` | Random Object Class Occlusion — occlude random (non-corresponding) regions at the same strength levels, as a control |
| `ro_pN` | Random Occlusion — occlude random image patches irrespective of disease regions |
| `all_noise` / `all_noise_mean` | Correlated / mean-noise injection over the full image |

---

## Usage

### Run model inference

```bash
python -m src.benchmark.cli \
    --model medgemma \
    --data-json data/padchest-gr/chexpert-by-label/verified_samples.json \
    --data "data/padchest-gr/BIMCV-Padchest-GR/PadChest_GR_images" \
    --experiment baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 100
```

Key flags:
- `--model` — model key from the registry (required)
- `--data-json` / `--data` — dataset annotation JSON and image directory (required)
- `--experiment` — experiment name, e.g. `baseline`, `oco_p50`, `doco_p50`, `roco_p75`, `ro_p50` (required)
- `--output-dir` — results directory (default: `outputs`)
- `--single-prompt-baseline` — for `--experiment baseline`, share one prompt across models (writes to `outputs/baseline_SP/`)
- `--seed` — random seed for occlusion region selection (default: `3`)

### Run evaluation

```bash
python evaluations/run_eval.py --filepath <output.json> --output-mode per-experiment
```

### Submit SLURM batch jobs

```bash
# Submit all models for a given experiment family / dataset
./scripts/baseline/padchest-gr/submit_all.sh
./scripts/ro/padchest-gr/submit_all.sh
./scripts/oco/padchest-gr/submit_all.sh
./scripts/doco/padchest-gr/submit_all.sh

# Or submit an individual model
sbatch scripts/baseline/padchest-gr/run_medgemma.sh
```

### Monitoring

```bash
squeue -u $USER
tail -f logs/baseline/*.out
scancel -u $USER
```

Output format: JSONL, one line per sample, at `outputs/<experiment>/<dataset>/<model>.jsonl`:

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

---

## Results

The figures below (`figures/ro/`) sweep **Random Occlusion (ro)** strength from 0% to 100% and track three metrics across the eight core models evaluated in the paper. A model whose curve stays essentially **flat** as occlusion increases is not deriving its score from the occluded visual content — a signature of shortcut behavior; a model whose curve **degrades** with occlusion is, at least to that extent, grounded in the image.

<p align="center">
  <a href="figures/ro/F1_RadGraph.pdf">F1-RadGraph vs. Occlusion Level (PDF)</a> ·
  <a href="figures/ro/Macro_F1_5.pdf">Macro F1-5, top-5 diseases (PDF)</a> ·
  <a href="figures/ro/Micro_F1_5.pdf">Micro F1-5, top-5 diseases (PDF)</a>
</p>

**F1-RadGraph vs. Occlusion.** CXRMate-ED (~0.153 → ~0.135) and MAIRA-2 (~0.129 → ~0.107) are the two models whose entity-level report quality actually declines as the image is progressively occluded, indicating the strongest reliance on visual content among the eight. CheXagent, LLaVA-Rad, Libra, RaDialog, MedGemma, and NV-Reason-CXR all sit on essentially flat, low-slope curves (roughly 0.03–0.06) from 0% to 100% occlusion — their RadGraph score barely moves even with the image fully occluded, consistent with heavy reliance on learned priors rather than pixel evidence.

**Macro/Micro F1-5 (top-5 diseases, U=neg) vs. Occlusion.** Disease-detection performance is noisier but tells a similar story: CXRMate-ED and CheXagent hold the highest F1 across the whole sweep with only mild degradation, while MAIRA-2's Micro F1-5 drops the most sharply of any model between 0% and 80% occlusion (~0.552 → ~0.497), again marking it as comparatively grounded. Several models (MedGemma, NV-Reason-CXR, RaDialog) cluster at the lower end of both metrics across all occlusion levels, showing little sensitivity to the amount of image actually visible — i.e., high report fluency does not translate into spatial grounding for these models, the central finding of the paper.

Full experiment sweeps (`oco`, `doco`, `roco`, noise) and per-dataset breakdowns (PadChest-GR, MIMIC-CXR) are reported in the paper; this repository reproduces them end-to-end via `scripts/<experiment>/<dataset>/submit_all.sh` and `evaluations/run_eval.py`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA out of memory | Lower `--num-images`, keep `--dtype bfloat16`, request a bigger-memory GPU in the SLURM script |
| Import errors | Confirm the correct venv is active and matches the model family table above |
| Model download fails | Set `HF_TOKEN`, check connectivity, verify `.models_cache` permissions |
| Submodule not found | `git submodule update --init --recursive`; check with `git submodule status` |

---

## Data availability

PadChest-GR and MIMIC-CXR are subject to their own data-use agreements (PhysioNet credentialed access for MIMIC-CXR) and are **not distributed** with this repository. This release contains the full inference, occlusion, and evaluation pipeline, along with placeholder paths that can be pointed at a local copy of either dataset once access has been obtained.

---

## Citation

If you use this benchmark, please cite:

```bibtex
@article{ruffini2026shovir,
  title   = {SHOVIR: A Benchmark for Evaluating Vision Shortcut Learning in Radiology Report Generation},
  author  = {Ruffini, Filippo and Salm{\'e}, Marco and Sicilia, Rosa and Guarrasi, Valerio and Soda, Paolo},
  journal = {arXiv preprint arXiv:2606.30201},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.30201}
}
```

---

## License

[Specify license here]

## Contact

For questions or issues, please contact the corresponding author or open an issue on GitHub.
