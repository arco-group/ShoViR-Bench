#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 05:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J llavarad_oco
#SBATCH -o logs/oco/llavarad_%j.out
#SBATCH -e logs/oco/llavarad_%j.err

# LLaVA-Rad OC Occlusion Experiment
# Model: microsoft/llava-rad-v1
# Virtual Environment: .SC_Libra_venv

set -euo pipefail

EXPERIMENT="${1:-oco_p100}"
SEED="${2:-3}"

echo "=== LLaVA-Rad OC Occlusion ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment: ${EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo ""

module load Python/3.11.3-GCCcore-12.3.0
source SC_Libra_venv/bin/activate

export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/ro

DATA_DIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
DATA_JSON="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/data/mimic_test_annotation.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo "Virtual environment: SC_Libra_venv"
echo ""

python -m src.benchmark.cli \
    --model llavarad \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment "${EXPERIMENT}" \
    --seed "${SEED}" \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 16

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
