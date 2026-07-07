#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 08:00:00
#SBATCH -J gpt54_oco_mimic
#SBATCH -o logs/oco/gpt54_mimic_%j.out
#SBATCH -e logs/oco/gpt54_mimic_%j.err

# GPT-5.4 Object Class Occlusion Experiment — MIMIC-CXR (OpenAI API)
# No GPU used — network-bound. Requires: OPENAI_API_KEY

set -euo pipefail

EXPERIMENT="${1:-oco_p100}"
SEED="${2:-3}"

echo "=== GPT-5.4 Object Class Occlusion (MIMIC-CXR) ==="
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
source "${PWD}/.gpt54_key"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/oco

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
    --num-images 24

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
