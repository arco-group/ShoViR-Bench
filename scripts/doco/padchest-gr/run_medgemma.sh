#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 01:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J medgemma_doco
#SBATCH -o logs/doco/medgemma_%j.out
#SBATCH -e logs/doco/medgemma_%j.err

# MedGemma Drop Object Class Occlusion Experiment
# Model: google/medgemma-1.5-4b-it (4B parameters)
# Virtual Environment: .venv_RRG

set -euo pipefail

EXPERIMENT="${1:-doco_p100}"
SEED="${2:-3}"

echo "=== MedGemma Drop Object Class Occlusion ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment: ${EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo ""

module load Python/3.11.5-GCCcore-13.2.0
source .venv_RRG/bin/activate

export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/doco

DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo "Virtual environment: .venv_RRG"
echo ""

python -m src.benchmark.cli \
    --model medgemma \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment "${EXPERIMENT}" \
    --seed "${SEED}" \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 100

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
