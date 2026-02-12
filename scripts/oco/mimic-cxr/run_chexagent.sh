#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 01:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J chexagent_baseline
#SBATCH -o logs/baseline/chexagent_%j.out
#SBATCH -e logs/baseline/chexagent_%j.err

# CheXagent Baseline Experiment
# Model: StanfordAIMI/CheXagent-3b (8B parameters)
# GPU Memory: ~20GB in bfloat16
# Virtual Environment: .venv_chexagent

set -euo pipefail

echo "=== CheXagent Baseline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load Python module
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source .venv_chexagent/bin/activate

# Set environment
export HF_HOME="${PWD}/.models_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Create output directories
mkdir -p outputs/oco/mimic-cxr-jpg
mkdir -p logs/oco

# Paths
DATA_DIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
DATA_JSON="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/data/classes_jsons/Atelectasis.json"

echo "Data directory: ${DATA_DIR}"
echo "Data JSON: ${DATA_JSON}"
echo "Virtual environment: .venv_chexagent"
echo ""

# Run inference
python -m src.benchmark.cli \
    --model chexagent \
    --data-json "${DATA_JSON}" \
    --data "${DATA_DIR}" \
    --experiment oco_p100 \
    --output-dir outputs \
    --cache-dir .models_cache \
    --device cuda:0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --num-images 40

echo ""
echo "=== Job Complete ==="
echo "End time: $(date)"
