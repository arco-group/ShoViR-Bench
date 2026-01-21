#!/bin/bash
# Submit all benchmark jobs to SLURM
# Usage: ./scripts/benchmark/submit_all.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Create log directory
mkdir -p logs/benchmark

# Check for dry-run flag
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
    echo ""
fi

echo "=== PadChest-GR Baseline Benchmark ==="
echo "Project: ${PROJECT_DIR}"
echo "Data: data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
echo "Output: outputs/padchest_baseline/"
echo ""

# Models to run (excluding libra which is incomplete)
MODELS=(
    "medgemma"
    "maira2"
    "chexagent"
    "cxrmateed"
    "nv_reason_cxr"
)

echo "Models to benchmark:"
for model in "${MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

# Submit jobs
echo "Submitting jobs..."
JOB_IDS=()

for model in "${MODELS[@]}"; do
    script="scripts/benchmark/run_${model}.sh"

    if [[ ! -f "$script" ]]; then
        echo "  [SKIP] $model - script not found: $script"
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY] Would submit: $script"
    else
        job_id=$(sbatch --parsable "$script")
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
    echo "  tail -f logs/benchmark/*.out"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${JOB_IDS[*]}"
fi
