#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 02:00:00
#SBATCH -J gpt54_all_noise_mean
#SBATCH -o logs/all_noise_mean/gpt54_%j.out
#SBATCH -e logs/all_noise_mean/gpt54_%j.err

# GPT-5.4 All Noise Mean Experiment (OpenAI API)
# No GPU used — network-bound. Requires: OPENAI_API_KEY

set -euo pipefail

EXTRA_ARGS="${1:-}"

echo "=== GPT-5.4 All Noise Mean (PadChest-GR) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time: $(date)"
echo ""

module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv_RRG/bin/activate

export HF_HOME="${PWD}/.models_cache"
source "${PWD}/.gpt54_key"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/all_noise_mean

DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo ""

python -m src.benchmark.cli \
    --model gpt54 \
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
