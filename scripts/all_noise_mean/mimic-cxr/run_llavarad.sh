#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 07:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J llavarad_all_noise_mean_mimic
#SBATCH -o logs/all_noise_mean/llavarad_mimic_%j.out
#SBATCH -e logs/all_noise_mean/llavarad_mimic_%j.err

set -euo pipefail

EXTRA_ARGS="${1:-}"

echo "=== LLaVA-Rad MIMIC-CXR All Noise Mean ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time: $(date)"
echo ""

module purge
module load Python/3.11.5-GCCcore-13.2.0
source .SC_Libra_venv/bin/activate

export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p logs/all_noise_mean

DATA_DIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
DATA_JSON="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/data/mimic_test_annotation.json"

python -m src.benchmark.cli \
    --model llavarad \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment all_noise_mean \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 4 \
    ${EXTRA_ARGS}

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
