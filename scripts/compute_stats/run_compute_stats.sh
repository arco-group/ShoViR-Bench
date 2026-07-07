#!/usr/bin/env bash
# Run compute_model_stats.py for every model inside its correct venv.
# Each model is launched as a separate SLURM job so the correct venv can be sourced.
#
# Usage:
#   # Submit one model:
#   MODEL=medgemma bash scripts/compute_stats/run_compute_stats.sh
#
#   # Submit all (loop):
#   for m in medgemma maira-2 cxrmateed chexagent chexone nv-reason-cxr-3b libra llavarad radialog; do
#       MODEL=$m bash scripts/compute_stats/run_compute_stats.sh
#   done

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-medgemma}"

# Map model key → venv path and GPU memory
declare -A VENV_MAP=(
    [medgemma]=".venv_RRG"
    [maira-2]=".venv_RRG"
    [cxrmateed]=".venv_RRG"
    [nv-reason-cxr-3b]=".venv_nv"
    [chexone]=".venv_nv"
    [libra]=".SC_Libra_venv"
    [llavarad]=".SC_Libra_venv"
    [chexagent]=".venv_chexagent"
    [radialog]=".radialog_venv"
)

VENV="${VENV_MAP[$MODEL]:-".venv_RRG"}"

sbatch <<EOF
#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -J compute_stats_${MODEL}
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -N 1
#SBATCH -o ${ROOT}/logs/compute_stats_${MODEL}_%j.out
#SBATCH -e ${ROOT}/logs/compute_stats_${MODEL}_%j.err

set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
cd ${ROOT}
source ${VENV}/bin/activate
export HF_TOKEN="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"
export HF_HOME="${ROOT}/.models_cache"
export PYTHONPATH="${ROOT}:\${PYTHONPATH:-}"

python -m src.postprocessing.compute_model_stats \\
    --model ${MODEL} \\
    --n-images 5 \\
    --device cuda:0 \\
    --dtype bfloat16
EOF

echo "Submitted SLURM job for model: ${MODEL}"
