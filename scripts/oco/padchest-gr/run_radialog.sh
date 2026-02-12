#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 01:20:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J radialog_oco
#SBATCH -o logs/oco/radialog_%j.out
#SBATCH -e logs/oco/radialog_%j.err

# RaDialog Object Class Occlusion Experiment
# Model: ChantalPellworworworworworwor/RaDialog_RRG
# Virtual Environment: .radialog_venv

set -euo pipefail

EXPERIMENT="${1:-oco_p100}"
SEED="${2:-3}"

echo "=== RaDialog Object Class Occlusion ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment: ${EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo ""

module load Python/3.11.5-GCCcore-13.2.0
source .radialog_venv/bin/activate

export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/oco

DATA_DIR="data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
DATA_JSON="data/padchest-gr/chexpert-by-label/verified_samples.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo "Virtual environment: .radialog_venv"
echo ""

python -m src.benchmark.cli \
    --model radialog \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment "${EXPERIMENT}" \
    --seed "${SEED}" \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 32

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
