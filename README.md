# Radiology Benchmark

Comprehensive benchmark framework for evaluating vision-language models on chest X-ray analysis tasks using the PadChest-GR dataset.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Supported Models](#supported-models)
- [Usage](#usage)
- [Running Experiments](#running-experiments)
- [Output Format](#output-format)

## Quick Start

Get started in 3 steps:

```bash
# 1. Initialize submodules
git submodule update --init --recursive

# 2. Load Python and create environment (example: .venv_RRG)
module load Python/3.11.5-GCCcore-13.2.0
python3.11 -m venv .venv_RRG
source .venv_RRG/bin/activate
pip install -r requirements.txt

# 3. Run a model
python -m src.benchmark.cli \
    --model medgemma \
    --data-json data/padchest-gr/chexpert-by-label/verified_samples.json \
    --data "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images" \
    --experiment baseline \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 10
```

For SLURM batch jobs:
```bash
# Submit all models
./scripts/baseline/submit_all.sh

# Or submit individual model
sbatch scripts/baseline/run_medgemma.sh
```

## Project Structure

```
├── src/benchmark/
│   ├── models/          # Model implementations and registry
│   ├── prompts.py       # Model-specific prompts
│   ├── cli.py          # Command-line interface
│   └── hf_runner.py    # HuggingFace pipeline runner
├── scripts/baseline/    # SLURM batch scripts
├── Libra/              # Git submodule for Libra & LLaVA-Rad models
├── RaDialog-interactive-radiology-report-generation/  # Git submodule for RaDialog
├── data/               # Dataset directory
├── outputs/            # Experiment results
│   └── baseline/
│       └── padchest-gr/  # PadChest-GR baseline results
└── logs/              # Job logs
```

## Environment Setup

This project requires **Python 3.11.5** and uses separate virtual environments for different model families due to dependency conflicts.

### Prerequisites

On NAISS Alvis cluster:
```bash
module load Python/3.11.5-GCCcore-13.2.0
```

### Initialize Submodules

This repository includes submodules for Libra and RaDialog. Initialize them first:

```bash
# Initialize and update all submodules
git submodule update --init --recursive

# Or if cloning for the first time
git clone --recurse-submodules <repository-url>
```

**Submodules:**
- `Libra/` - LLaVA-based architecture for Libra and LLaVA-Rad models
- `RaDialog-interactive-radiology-report-generation/` - RaDialog model implementation

### Virtual Environments Overview

| Environment | Models | Backend |
|------------|--------|---------|
| `.radialog_venv` | RaDialog | Custom |
| `.venv_RRG` | CXRMateED, MedGemma, MAIRA-2 | Standard HF |
| `.venv_nv` | NV-Reason-CXR, CheXOne | Qwen2.5-VL |
| `.SC_Libra_venv` | Libra, LLaVA-Rad | LLaVA-based |
| `.venv_chexagent` | CheXagent | LLaVA-Next |

### 1. Setup `.venv_RRG` (CXRMateED, MedGemma, MAIRA-2)

```bash
python3.11 -m venv .venv_RRG
source .venv_RRG/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements** (`requirements.txt`):
- `torch>=2.1.0`
- `transformers>=4.44.0`
- `Pillow>=10.0.0`
- `accelerate>=0.24.0`
- Additional HuggingFace dependencies

### 2. Setup `.venv_nv` (NV-Reason-CXR, CheXOne - Qwen2.5-VL)

```bash
python3.11 -m venv .venv_nv
source .venv_nv/bin/activate
pip install --upgrade pip
pip install -r requirements_nv.txt
```

**Requirements** (`requirements_nv.txt`):
- `torch>=2.1.0` (CUDA-enabled)
- `transformers>=4.45.0`
- `qwen-vl-utils`
- `Pillow>=10.0.0`
- `accelerate>=0.24.0`

### 3. Setup `.SC_Libra_venv` (Libra, LLaVA-Rad)

**Note:** Requires the `Libra/` submodule to be initialized first.

```bash
python3.11 -m venv .SC_Libra_venv
source .SC_Libra_venv/bin/activate
pip install --upgrade pip

# Install Libra from submodule
cd Libra
pip install -e .
cd ..

# Install remaining dependencies
pip install torch transformers Pillow accelerate
```

**Libra Submodule:**
The Libra directory is a git submodule pointing to https://github.com/X-iZhang/Libra. If not present:
```bash
git submodule update --init Libra
```

### 4. Setup `.venv_chexagent` (CheXagent)

```bash
python3.11 -m venv .venv_chexagent
source .venv_chexagent/bin/activate
pip install --upgrade pip
pip install torch transformers Pillow accelerate
pip install git+https://github.com/haotian-liu/LLaVA.git@main  # LLaVA-Next dependencies
```

### 5. Setup `.radialog_venv` (RaDialog)

**Note:** Requires the `RaDialog-interactive-radiology-report-generation/` submodule to be initialized first.

```bash
python3.11 -m venv .radialog_venv
source .radialog_venv/bin/activate
pip install --upgrade pip

# Install RaDialog from submodule
cd RaDialog-interactive-radiology-report-generation
pip install -r requirements.txt
pip install -e .
cd ..

# Install additional dependencies if needed
pip install torch transformers Pillow accelerate
```

**RaDialog Submodule:**
The RaDialog directory is a git submodule. If not present:
```bash
git submodule update --init RaDialog-interactive-radiology-report-generation
```

## Supported Models

### Model Registry

All models use **bfloat16** precision for optimal memory usage and performance.

| Model | Key | Model ID | Batch Size | Virtual Env |
|-------|-----|----------|------------|-------------|
| **MedGemma** | `medgemma` | `google/medgemma-1.5-4b-it` | 100 | `.venv_RRG` |
| **MAIRA-2** | `maira-2` | `microsoft/maira-2` | 6 | `.venv_RRG` |
| **CXRMateED** | `cxrmateed` | `aehrc/cxrmate-single-tf` | 256 | `.venv_RRG` |
| **NV-Reason-CXR** | `nv-reason-cxr-3b` | `nvidia/NV-Reason-CXR-3B` | 128 | `.venv_nv` |
| **CheXOne** | `chexone` | `StanfordAIMI/CheXOne` | 8 | `.venv_nv` |
| **Libra** | `libra` | `X-iZhang/libra-v1.0-7b` | 24 | `.SC_Libra_venv` |
| **LLaVA-Rad** | `llavarad` | `X-iZhang/libra-llava-rad` | 24 | `.SC_Libra_venv` |
| **CheXagent** | `chexagent` | `StanfordAIMI/CheXagent-8b` | 40 | `.venv_chexagent` |
| **RaDialog** | `radialog` | (Custom) | 32 | `.radialog_venv` |

## Usage

### Command-Line Interface

Basic usage:
```bash
python -m src.benchmark.cli \
    --model <model_key> \
    --data-json <path_to_json> \
    --data <path_to_images> \
    --experiment <experiment_name> \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images <N>
```

### Parameters

- `--model`: Model key from the registry (required)
- `--data-json`: Path to dataset JSON with image metadata (required)
- `--data`: Path to image directory (required)
- `--experiment`: Experiment name (e.g., `baseline`, `occluded`) (required)
- `--output-dir`: Directory for results (default: `outputs`)
- `--cache-dir`: HuggingFace cache directory (default: `.models_cache`)
- `--single-prompt-baseline`: For `--experiment baseline`, use one shared prompt across models and save under `outputs/baseline_SP/`
- `--shared-prompt-key`: Prompt used by `--single-prompt-baseline` (default: `radiology_minimal`)
- `--device`: Device for inference (default: `cuda:0`)
- `--dtype`: Data type precision (default: `bfloat16`)
- `--trust-remote-code`: Allow loading custom model code
- `--num-images`: Number of images to process (for testing)

### Example: Running MedGemma

```bash
# Activate environment
source .venv_RRG/bin/activate

# Run inference
python -m src.benchmark.cli \
    --model medgemma \
    --data-json data/padchest-gr/chexpert-by-label/verified_samples.json \
    --data "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images" \
    --experiment baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 100
```

### Example: Shared-Prompt Baseline

```bash
python -m src.benchmark.cli \
    --model medgemma \
    --data-json data/padchest-gr/chexpert-by-label/verified_samples.json \
    --data "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images" \
    --experiment baseline \
    --single-prompt-baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 100
```

This keeps the preprocessing experiment as `baseline`, uses the shared `radiology_minimal` prompt by default, and writes to `outputs/baseline_SP/<dataset>/`.

### Example: Running CheXOne

```bash
# Activate environment
source .venv_nv/bin/activate

# Set HuggingFace token
export HF_TOKEN="your_hf_token_here"

# Run inference
python -m src.benchmark.cli \
    --model chexone \
    --data-json data/padchest-gr/chexpert-by-label/verified_samples.json \
    --data "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images" \
    --experiment baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 8
```

## Running Experiments

### SLURM Batch Submission

All baseline experiments can be submitted via SLURM batch scripts in `scripts/baseline/`.

#### Submit All Models

```bash
./scripts/baseline/submit_all.sh
```

This will submit jobs for all 9 models sequentially.

#### Dry Run (Test Without Submitting)

```bash
./scripts/baseline/submit_all.sh --dry-run
```

#### Submit Shared-Prompt Baseline

```bash
./scripts/baseline/submit_all.sh --dataset padchest-gr --single-prompt-baseline
```

#### Submit Individual Models

```bash
# Submit specific model
sbatch scripts/baseline/run_medgemma.sh
sbatch scripts/baseline/run_chexone.sh
sbatch scripts/baseline/run_nv_reason_cxr.sh
# ... etc
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Monitor output logs
tail -f logs/baseline/*.out

# View errors
tail -f logs/baseline/*.err
```

### Canceling Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel all baseline jobs
scancel -n medgemma_baseline -n chexone_baseline -n maira2_baseline
```

## Model-Specific Notes

### CheXOne & NV-Reason-CXR
- Both use Qwen2.5-VL backbone
- Require `qwen-vl-utils` package
- Share `.venv_nv` environment
- Use dynamic resolution for images

### Libra & LLaVA-Rad
- Both require `Libra/` submodule installed
- Share `.SC_Libra_venv` environment
- Support temporal imaging (prior images)
- Use LLaVA architecture with custom projectors
- Submodule: `Libra/` (https://github.com/X-iZhang/Libra)

### RaDialog
- Requires `RaDialog-interactive-radiology-report-generation/` submodule
- Uses custom interactive architecture
- Submodule needs to be initialized and installed in `.radialog_venv`

### CheXagent
- Based on LLaVA-Next architecture
- Largest model (8B parameters)
- Requires most GPU memory (~20GB in bfloat16)

### CXRMateED
- Encoder-decoder architecture (not decoder-only)
- Smallest memory footprint
- Highest throughput for batch processing

## Output Format

Results are saved as JSONL files in `outputs/<experiment>/<dataset>/<model>.jsonl`.

For PadChest-GR baseline experiments: `outputs/baseline/padchest-gr/<model>.jsonl`

**Example output:**
```json
{
  "model": "chexone",
  "model_id": "StanfordAIMI/CheXOne",
  "prompt": "Analyze this chest X-ray...",
  "image_path": "data/padchest-gr/.../image.png",
  "generated_text": "The chest X-ray shows...",
  "metadata": {...}
}
```

Each line contains:
- `model`: Model key
- `model_id`: Full HuggingFace model identifier
- `prompt`: Input prompt used
- `image_path`: Path to input image
- `generated_text`: Model's generated report
- `metadata`: Additional experiment metadata

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `--num-images` batch size
- Use `--dtype bfloat16` (already default)
- Request GPU with more memory in SLURM script

**2. Import Errors**
- Ensure correct virtual environment is activated
- Verify all dependencies installed: `pip list`
- Check Python version: `python --version` (should be 3.11.x)

**3. Model Download Fails**
- Set `HF_TOKEN` environment variable
- Check internet connectivity
- Verify cache directory permissions

**4. Submodule Not Found**
- Initialize submodules: `git submodule update --init --recursive`
- Check submodule status: `git submodule status`
- Update submodules: `git submodule update --remote`

**5. Permission Errors**
- Ensure write access to `outputs/` and `logs/` directories
- Check cache directory: `mkdir -p .models_cache`

**6. Libra/RaDialog Import Errors**
- Verify submodules are initialized: `ls Libra/` and `ls RaDialog-interactive-radiology-report-generation/`
- Ensure correct virtual environment is activated
- Reinstall from submodule: `cd Libra && pip install -e . && cd ..`

## Environment Variables

Required:
```bash
export HF_TOKEN="your_huggingface_token"
export HF_HOME="${PWD}/.models_cache"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

Optional:
```bash
export CUDA_VISIBLE_DEVICES="0"  # Specify GPU
export TRANSFORMERS_CACHE="${PWD}/.models_cache"
```

## Citation

If you use this benchmark framework, please cite:
```bibtex
@software{radiology_benchmark_2024,
  title={Radiology Vision-Language Model Benchmark},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repo}
}
```

## License

[Specify license here]

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.
