#!/bin/bash
#SBATCH -A naiss2023-6-336
#SBATCH -p alvis
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J cxrmateed_benchmark
#SBATCH -o logs/benchmark/cxrmateed_%j.out
#SBATCH -e logs/benchmark/cxrmateed_%j.err

# CXRMateED Benchmark on PadChest-GR
# Model: aehrc/cxrmate-single-tf (smaller, encoder-decoder)
# GPU Memory: ~8GB

set -euo pipefail

echo "=== CXRMateED Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules
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
echo "Output: outputs/padchest_baseline/cxrmateed.jsonl"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model cxrmateed \
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
