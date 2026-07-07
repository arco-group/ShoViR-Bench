#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p NOGPU
#SBATCH -t 02:00:00
#SBATCH -J gemini_ro
#SBATCH -o logs/ro/gemini_%j.out
#SBATCH -e logs/ro/gemini_%j.err

# Gemini 2.0 Flash Random Occlusion Experiment (Google GenAI API)
# No GPU used — network-bound. Requires: GOOGLE_API_KEY

set -euo pipefail

EXPERIMENT="${1:-ro_p100}"
SEED="${2:-3}"
EXTRA_ARGS="${3:-}"

echo "=== Gemini 2.0 Flash Random Occlusion (PadChest-GR) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment: ${EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo ""

module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv_RRG/bin/activate

export HF_HOME="${PWD}/.models_cache"
source "${PWD}/.gemini_key"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/ro

DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo ""

python -m src.benchmark.cli \
    --model gemini \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment "${EXPERIMENT}" \
    --seed "${SEED}" \
    --output-dir outputs \
    --cache-dir .models_cache \
    --num-images 24 \
    ${EXTRA_ARGS}

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
