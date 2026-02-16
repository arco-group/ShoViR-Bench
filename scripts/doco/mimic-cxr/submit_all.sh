#!/bin/bash
# Submit all DOCO (DOC Occlusion) jobs to SLURM
# Usage: ./scripts/doco/mimic-cxr/submit_all.sh --experiment oco_p50 --seed 3 [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$PROJECT_DIR"

# Defaults
EXPERIMENT="doco_p20"
SEED="3"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 --experiment doco_pXX --seed N [--dry-run]"; exit 1 ;;
    esac
done

# Create log directory
mkdir -p logs/doco

if $DRY_RUN; then
    echo "=== DRY RUN MODE ==="
    echo ""
fi

echo "=== MIMIC-CXR OC Occlusion Experiments ==="
echo "Project: ${PROJECT_DIR}"
echo "Experiment: ${EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Data: mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
echo "Data JSON: data/mimic_test_annotation.json"
echo ""

# Models to run
MODELS=(
    "medgemma"
    "maira2"
    "chexagent"
    "chexone"
    "libra"
    "cxrmateed"
    "nv_reason_cxr"
    "radialog"
    "llavarad"
)

echo "Models to run:"
for model in "${MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

# Submit jobs
echo "Submitting jobs..."
JOB_IDS=()

for model in "${MODELS[@]}"; do
    script="scripts/doco/mimic-cxr/run_${model}.sh"

    if [[ ! -f "$script" ]]; then
        echo "  [SKIP] $model - script not found: $script"
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY] Would submit: sbatch $script $EXPERIMENT $SEED"
    else
        job_id=$(sbatch --parsable "$script" "$EXPERIMENT" "$SEED")
        JOB_IDS+=("$job_id")
        echo "  [OK] $model - Job ID: $job_id"
    fi
done

echo ""
if ! $DRY_RUN && [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    echo "=== Submitted ${#JOB_IDS[@]} jobs ==="
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/doco/*.out"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${JOB_IDS[*]}"
fi
