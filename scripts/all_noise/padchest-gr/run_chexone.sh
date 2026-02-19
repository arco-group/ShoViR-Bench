#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 7:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J chexone_all_noise
#SBATCH -o logs/all_noise/chexone_%j.out
#SBATCH -e logs/all_noise/chexone_%j.err

# CheXOne All Noise Experiment
# Model: StanfordAIMI/CheXOne
# Virtual Environment: .venv_nv (Reasoning Model Backbones - Qwen2.5VL)

set -euo pipefail

EXTRA_ARGS="${1:-}"

echo "=== CheXOne All Noise ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load Python module
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source .venv_nv/bin/activate

# Set environment
export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Create output directories
mkdir -p logs/all_noise

# Paths
DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo "Virtual environment: .venv_nv"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model chexone \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment all_noise \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 8 \
    ${EXTRA_ARGS}

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
