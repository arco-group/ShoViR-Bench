#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 08:20:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J nv_reason_cxr_ro
#SBATCH -o logs/ro/nv_reason_cxr_%j.out
#SBATCH -e logs/ro/nv_reason_cxr_%j.err
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.salme@unicampus.it

# NV-Reason-CXR Random Occlusion Experiment
# Model: nvidia/nv-reason-cxr-3b
# Virtual Environment: .venv_nv

set -euo pipefail

EXPERIMENT="${1:-ro_p100}"
SEED="${2:-3}"

echo "=== NV-Reason-CXR Random Occlusion ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Experiment: ${EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo ""

module load Python/3.11.3-GCCcore-12.3.0
source .venv_nv/bin/activate

export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/ro

DATA_DIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
DATA_JSON="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/data/mimic_test_annotation.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo "Virtual environment: .venv_nv"
echo ""

python -m src.benchmark.cli \
    --model nv-reason-cxr-3b \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment "${EXPERIMENT}" \
    --seed "${SEED}" \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 128

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
