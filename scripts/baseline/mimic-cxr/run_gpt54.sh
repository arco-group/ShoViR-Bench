#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 08:00:00
#SBATCH -J gpt54_baseline_mimic
#SBATCH -o logs/baseline/gpt54_%j.out
#SBATCH -e logs/baseline/gpt54_%j.err

# GPT-5.4 Baseline Experiment — MIMIC-CXR (OpenAI API)
# Model: gpt-5.4 via OpenAI API — no GPU used, network-bound
# Requires: OPENAI_API_KEY env var, openai Python package

set -euo pipefail

EXPERIMENT="${1:-baseline}"
SEED="${2:-3}"
EXTRA_ARGS=("${@:3}")

echo "=== GPT-5.4 MIMIC-CXR ==="
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
export HF_TOKEN="${HF_TOKEN:-}"
source "${PWD}/.gpt54_key"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/baseline

DATA_DIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
DATA_JSON="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/data/mimic_test_annotation.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo ""

python -m src.benchmark.cli \
    --model gpt54 \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment "${EXPERIMENT}" \
    --seed "${SEED}" \
    --output-dir outputs \
    --cache-dir .models_cache \
    --num-images 24 \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
