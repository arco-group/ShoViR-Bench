#!/bin/bash
# Submit all baseline jobs to SLURM
# Usage: ./scripts/baseline/submit_all.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$PROJECT_DIR"

# Create log directory
mkdir -p logs/baseline

# Defaults
DRY_RUN=false
SKIP_EXISTING=false
EXPERIMENT="baseline"
SEED="3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--dry-run] [--skip-existing]"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Output-existence check (used by --skip-existing).
# Maps model key -> search pattern inside output filenames.
# nv_reason_cxr is special: CLI writes files named *nv_reason* not *nv_reason_cxr*.
# ---------------------------------------------------------------------------
declare -A _MODEL_PATTERN=(
    ["medgemma"]="medgemma"   ["maira2"]="maira2"
    ["chexagent"]="chexagent" ["chexone"]="chexone"
    ["libra"]="libra"         ["cxrmateed"]="cxrmateed"
    ["nv_reason_cxr"]="nv_reason"
    ["radialog"]="radialog"   ["llavarad"]="llavarad"
)
_output_exists() {   # _output_exists <model> <experiment> <seed>
    local model="$1" exp="$2" seed="${3:-3}"
    local pat="${_MODEL_PATTERN[$model]:-$model}"
    local dir
    if [[ "$exp" =~ ^(oco|doco|roco|ro)_(.+)$ ]]; then
        dir="outputs/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/padchest-gr"
    else
        dir="outputs/${exp}/padchest-gr"
    fi
    compgen -G "${dir}/*${pat}*::seed=${seed}.json" > /dev/null 2>&1
}

if $DRY_RUN; then
    echo "=== DRY RUN MODE ==="
    echo ""
fi

echo "=== PadChest-GR Baseline Experiments ==="
echo "Project: ${PROJECT_DIR}"
echo "Data: data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
echo "Data JSON: data/padchest-gr/chexpert-by-label/verified_samples.json"
echo "Output: outputs/baseline/padchest-gr/"
echo ""

# Models to run
MODELS=(
    "medgemma"
    "maira2"
    "chexagent"
    #"chexone"
    "libra"
    "cxrmateed"
    "nv_reason_cxr"
    "radialog"
    "llavarad"
)

echo "Models to run baseline:"
for model in "${MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

# Submit jobs
echo "Submitting jobs..."
JOB_IDS=()

for model in "${MODELS[@]}"; do
    script="scripts/baseline/padchest-gr/run_${model}.sh"

    if [[ ! -f "$script" ]]; then
        echo "  [SKIP] $model - script not found: $script"
        continue
    fi

    if $SKIP_EXISTING && _output_exists "$model" "$EXPERIMENT" "$SEED"; then
        echo "  [SKIP] $model - output already exists (seed=${SEED})"
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY] Would submit: sbatch $script"
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
    echo "  tail -f logs/baseline/*.out"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${JOB_IDS[*]}"
fi
