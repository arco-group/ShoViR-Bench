#!/bin/bash
#SBATCH -A naiss2023-6-336
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J nv_reason_benchmark
#SBATCH -o logs/benchmark/nv_reason_%j.out
#SBATCH -e logs/benchmark/nv_reason_%j.err

# NV-Reason-CXR-3B Benchmark on PadChest-GR
# Model: nvidia/NV-Reason-CXR-3B (3B parameters)
# GPU Memory: ~12GB in float16
# NOTE: May require requirements_nv.txt environment

set -euo pipefail

echo "=== NV-Reason-CXR-3B Benchmark ==="
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
echo "Output: outputs/padchest_baseline/nv-reason-cxr-3b.jsonl"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model nv-reason-cxr-3b \
    --data "${DATA_DIR}" \
    --experiment padchest_baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype float16 \
    --parallel \
    --num-workers 4 \
    --prefetch 8

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
