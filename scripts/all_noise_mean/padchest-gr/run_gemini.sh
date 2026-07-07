#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p NOGPU
#SBATCH -t 02:00:00
#SBATCH -J gemini_all_noise_mean
#SBATCH -o logs/all_noise_mean/gemini_%j.out
#SBATCH -e logs/all_noise_mean/gemini_%j.err

# Gemini 2.0 Flash All Noise Mean Experiment (Google GenAI API)
# No GPU used — network-bound. Requires: GOOGLE_API_KEY

set -euo pipefail

EXTRA_ARGS="${1:-}"

echo "=== Gemini 2.0 Flash All Noise Mean (PadChest-GR) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time: $(date)"
echo ""

module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv_RRG/bin/activate

export HF_HOME="${PWD}/.models_cache"
source "${PWD}/.gemini_key"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/all_noise_mean

DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo ""

python -m src.benchmark.cli \
    --model gemini \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment all_noise_mean \
    --output-dir outputs \
    --cache-dir .models_cache \
    --num-images 24 \
    ${EXTRA_ARGS}

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
