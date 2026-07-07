#!/bin/bash
# Submit all all_noise_mean jobs to SLURM for MIMIC-CXR
# Usage: ./scripts/all_noise_mean/mimic-cxr/submit_all.sh [--dry-run] [--skip-existing] [--one-prompt]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$PROJECT_DIR"

mkdir -p logs/all_noise_mean

DRY_RUN=false
SKIP_EXISTING=false
EXPERIMENT="all_noise_mean"
OUTPUT_EXPERIMENT="${EXPERIMENT}"
SEED="3"
EXTRA_ARG_STRING=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --one-prompt|--single-prompt)
            OUTPUT_EXPERIMENT="all_noise_mean_SP"
            EXTRA_ARG_STRING="--prompt-key radiology_minimal --output-experiment ${OUTPUT_EXPERIMENT}"
            shift
            ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--dry-run] [--skip-existing] [--one-prompt]"; exit 1 ;;
    esac
done

declare -A _MODEL_PATTERN=(
    ["medgemma"]="medgemma"   ["maira2"]="maira-2"
    ["chexagent"]="CheXagent" ["chexone"]="chexone"
    ["libra"]="libra"         ["cxrmateed"]="cxrmate"
    ["nv_reason_cxr"]="NV-Reason"
    ["radialog"]="RaDialog"   ["llavarad"]="llava"
    ["gpt54"]="gpt-5.4"
    ["gemini"]="gemini-2.0-flash"
)
_output_exists() {
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

echo "=== MIMIC-CXR All Noise Mean Experiments ==="
echo "Project: ${PROJECT_DIR}"
echo "Experiment: ${EXPERIMENT}"
echo "Output experiment: ${OUTPUT_EXPERIMENT}"
echo "Data: mimic_test_imagenome/mimic-cxr-jpg/2.1.0/files"
echo "Data JSON: data/mimic_test_annotation.json"
echo "Output: outputs/${OUTPUT_EXPERIMENT}/mimic-cxr-jpg/"
if [[ -n "$EXTRA_ARG_STRING" ]]; then
    echo "Extra CLI args: ${EXTRA_ARG_STRING}"
fi
echo ""

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

echo "Models to run all_noise_mean:"
for model in "${MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

echo "Submitting jobs..."
JOB_IDS=()

for model in "${MODELS[@]}"; do
    script="scripts/all_noise_mean/mimic-cxr/run_${model}.sh"

    if [[ ! -f "$script" ]]; then
        echo "  [SKIP] $model - script not found: $script"
        continue
    fi

    if $SKIP_EXISTING && _output_exists "$model" "$OUTPUT_EXPERIMENT" "$SEED"; then
        echo "  [SKIP] $model - output already exists (seed=${SEED})"
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY] Would submit: sbatch $script ${EXTRA_ARG_STRING}"
    else
        job_id=$(sbatch --parsable "$script" "$EXTRA_ARG_STRING")
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
    echo "  tail -f logs/all_noise_mean/*.out"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${JOB_IDS[*]}"
fi
