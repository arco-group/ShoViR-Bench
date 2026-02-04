#!/bin/bash
#SBATCH -A naiss2023-6-336
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J chexone_baseline
#SBATCH -o logs/baseline/chexone_%j.out
#SBATCH -e logs/baseline/chexone_%j.err

# CheXOne Baseline Experiment
# Model: StanfordAIMI/CheXOne
# Virtual Environment: .venv_nv (Reasoning Model Backbones - Qwen2.5VL)

set -euo pipefail

echo "=== CheXOne Baseline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load Python module
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source .venv_nv/bin/activate

# Set environment
export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Create output directories
mkdir -p outputs/baseline/padchest-gr
mkdir -p logs/baseline

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
    --experiment baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 8

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
