#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 00-04:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.salme@unicampus.it

set -euo pipefail

ml purge
ml Python/3.10.4-GCCcore-11.3.0

## --- Paths ---
WORKDIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG"
VENV="${WORKDIR}/.venv_eval"

DIR="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/outputs/ro/p100/mimic-cxr-jpg"

# --- Activate venv ---
cd "${VENV}" || exit 1
source bin/activate

cd "${WORKDIR}/evaluations" || exit 1

shopt -s nullglob

echo "Evaluating JSON files in: ${DIR}"
files=( "${DIR}"/*.json )

if (( ${#files[@]} == 0 )); then
  echo "No .json files found in ${DIR}"
  deactivate
  exit 0
fi

for f in "${files[@]}"; do
  echo "==> Running CheXbert eval on: ${f}"
  python run_eval_chexbert_class.py --filepath "${f}"
done

# --- Deactivate venv ---
deactivate
echo "Done."
