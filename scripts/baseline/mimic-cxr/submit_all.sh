#!/bin/bash
# Submit all baseline jobs to SLURM
# Usage: ./scripts/baseline/mimic-cxr/submit_all.sh [--experiment baseline] [--seed 3] [--dry-run] [--skip-existing] [--single-prompt-baseline]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$PROJECT_DIR"

# Defaults
EXPERIMENT="all_noise_mean"
OUTPUT_EXPERIMENT="${EXPERIMENT}"
SEED="3"
DRY_RUN=false
SKIP_EXISTING=false
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT="$2"; OUTPUT_EXPERIMENT="${EXPERIMENT}"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --single-prompt-baseline)
            EXPERIMENT="baseline"
            OUTPUT_EXPERIMENT="baseline_SP"
            EXTRA_ARGS+=(--single-prompt-baseline)
            shift
            ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--experiment baseline] [--seed N] [--dry-run] [--skip-existing] [--single-prompt-baseline]"; exit 1 ;;
    esac
done

# Create log directory
mkdir -p logs/baseline

# ---------------------------------------------------------------------------
# Output-existence check (used by --skip-existing).
# Maps model key -> search pattern inside output filenames.
# nv_reason_cxr is special: CLI writes files named *nv_reason* not *nv_reason_cxr*.
# ---------------------------------------------------------------------------
declare -A _MODEL_PATTERN=(
    ["medgemma"]="medgemma"   ["maira2"]="maira-2"
    ["chexagent"]="CheXagent" ["chexone"]="chexone"
    ["libra"]="libra"         ["cxrmateed"]="cxrmate"
    ["nv_reason_cxr"]="NV-Reason"
    ["radialog"]="RaDialog"   ["llavarad"]="llava"
    ["gpt54"]="gpt-5.4"
    ["gemini"]="gemini-2.0-flash"
)
_output_exists() {   # _output_exists <model> <experiment> <seed>
    local model="$1" exp="$2" seed="${3:-3}"
    local pat="${_MODEL_PATTERN[$model]:-$model}"
    local dir
    if [[ "$exp" =~ ^(oco|doco|roco|ro)_(.+)$ ]]; then
        dir="outputs/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/mimic-cxr-jpg"
    else
        dir="outputs/${exp}/mimic-cxr-jpg"
    fi
    compgen -G "${dir}/*${pat}*::seed=${seed}.json" > /dev/null 2>&1
}

if $DRY_RUN; then
    echo "=== DRY RUN MODE ==="
    echo ""
fi

echo "=== MIMIC-CXR Baseline Experiments ==="
echo "Project: ${PROJECT_DIR}"
echo "Experiment: ${EXPERIMENT}"
echo "Output experiment: ${OUTPUT_EXPERIMENT}"
echo "Seed: ${SEED}"
echo "Data: mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
echo "Data JSON: data/mimic_test_annotation.json"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra CLI args: ${EXTRA_ARGS[*]}"
fi
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
    "gpt54"
    "gemini"
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
    script="scripts/baseline/mimic-cxr/run_${model}.sh"

    if [[ ! -f "$script" ]]; then
        echo "  [SKIP] $model - script not found: $script"
        continue
    fi

    if $SKIP_EXISTING && _output_exists "$model" "$OUTPUT_EXPERIMENT" "$SEED"; then
        echo "  [SKIP] $model - output already exists (${OUTPUT_EXPERIMENT}, seed=${SEED})"
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY] Would submit: sbatch $script $EXPERIMENT $SEED ${EXTRA_ARGS[*]}"
    else
        job_id=$(sbatch --parsable "$script" "$EXPERIMENT" "$SEED" "${EXTRA_ARGS[@]}")
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
