#!/bin/bash
#SBATCH -A naiss2023-6-336
#SBATCH -p alvis
#SBATCH -t 16:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J chexagent_benchmark
#SBATCH -o logs/benchmark/chexagent_%j.out
#SBATCH -e logs/benchmark/chexagent_%j.err

# CheXagent Benchmark on PadChest-GR
# Model: StanfordAIMI/CheXagent-8b (8B parameters)
# GPU Memory: ~20GB in bfloat16

set -euo pipefail

echo "=== CheXagent Benchmark ==="
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
echo "Output: outputs/padchest_baseline/chexagent.jsonl"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model chexagent \
    --data "${DATA_DIR}" \
    --experiment padchest_baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --parallel \
    --num-workers 4 \
    --prefetch 8

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
