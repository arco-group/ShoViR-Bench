#!/bin/bash
#SBATCH -A naiss2023-6-336
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J libra_baseline
#SBATCH -o logs/baseline/libra_%j.out
#SBATCH -e logs/baseline/libra_%j.err

# Libra Baseline Experiment
# Model: X-iZhang/libra-v1.0-7b (7B parameters)
# Virtual Environment: .SC_Libra_venv

set -euo pipefail

echo "=== Libra Baseline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load Python module
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source .SC_Libra_venv/bin/activate

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
echo "Virtual environment: .SC_Libra_venv"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model libra \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --num-images 24

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
