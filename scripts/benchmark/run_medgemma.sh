#!/bin/bash
#SBATCH -A naiss2023-6-336
#SBATCH -p alvis
#SBATCH -t 08:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J medgemma_benchmark
#SBATCH -o logs/benchmark/medgemma_%j.out
#SBATCH -e logs/benchmark/medgemma_%j.err

# MedGemma Benchmark on PadChest-GR
# Model: google/medgemma-1.5-4b-it (4B parameters)
# GPU Memory: ~10GB in float16

set -euo pipefail

echo "=== MedGemma Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules
module purge
module load PyTorch-bundle/2.4.0-foss-2023a-CUDA-12.4.0

# Set environment
export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Create output directory
mkdir -p outputs/padchest_baseline
mkdir -p logs/benchmark

# Data path
DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"

echo "Data directory: ${DATA_DIR}"
echo "Output: outputs/padchest_baseline/medgemma.jsonl"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model medgemma \
    --data "${DATA_DIR}" \
    --experiment padchest_baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --trust-remote-code \
    --parallel \
    --num-workers 4 \
    --prefetch 8

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
