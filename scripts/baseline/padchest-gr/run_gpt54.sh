#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:00:00
#SBATCH -J gpt54_baseline
#SBATCH -o logs/baseline/gpt54_%j.out
#SBATCH -e logs/baseline/gpt54_%j.err

# GPT-5.4 Baseline Experiment (OpenAI API)
# Model: gpt-5.4 via OpenAI API — no GPU used, network-bound
# Requires: OPENAI_API_KEY env var, openai Python package

set -euo pipefail

EXTRA_ARGS="${1:-}"

echo "=== GPT-5.4 Baseline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv_RRG/bin/activate

export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-}"
source "${PWD}/.gpt54_key"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p outputs/baseline/padchest-gr
mkdir -p logs/baseline

DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo ""

python -m src.benchmark.cli \
    --model gpt54 \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment baseline \
    --output-dir outputs \
    --cache-dir .models_cache \
    --num-images 24 \
    ${EXTRA_ARGS}

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
