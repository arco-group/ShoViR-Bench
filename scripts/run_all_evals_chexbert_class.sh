#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J eval_chexbert_class
#SBATCH -o logs/eval/eval_chexbert_class_%j.out
#SBATCH -e logs/eval/eval_chexbert_class_%j.err

# Evaluate model outputs under outputs/ with CheXbert per-target-category F1.
# Only processes files from the doco, oco, and ro experiments.
# Groups results by experiment and percentage level (pXX).
#
# Runner-specific flags:
#   --dry-run              Preview commands without running
#   --experiment <name>    Only evaluate files from this experiment (must be one of: doco, oco, ro)
#   --model <pattern>      Only evaluate files matching this pattern in filename (e.g. gpt-5.4, gemini, medgemma)
#   --parallel <N>         Max parallel evaluations (default: 4)
#
# All run_eval_chexbert_class.py flags are forwarded:
#   --output-mode          per-file (default) | per-experiment
#   --uncertain-mode       rrg- (default) | rrg+
#   --bootstrap-ci         Enable bootstrap confidence intervals
#   --n-resamples          Number of bootstrap resamples (default: 500)
#   --seed                 Random seed (default: 3)
#   --skip-existing        Skip evaluation if results already exist (do not override)
#
# Usage:
#   sbatch scripts/run_all_evals_chexbert_class.sh                          # run doco+oco+ro, per-file
#   sbatch scripts/run_all_evals_chexbert_class.sh --experiment ro          # only ro experiment
#   sbatch scripts/run_all_evals_chexbert_class.sh --model gpt-5.4          # only GPT-5.4 outputs
#   sbatch scripts/run_all_evals_chexbert_class.sh --model gemini --experiment ro  # Gemini, ro only
#   sbatch scripts/run_all_evals_chexbert_class.sh --bootstrap-ci           # with CI
#   bash   scripts/run_all_evals_chexbert_class.sh --dry-run                # preview
#   bash   scripts/run_all_evals_chexbert_class.sh --experiment oco --dry-run

set -euo pipefail

# ---------------------
# Parse arguments
# ---------------------
DRY_RUN=false
FILTER_EXPERIMENT=""
FILTER_MODEL=""
MAX_PARALLEL=4

# run_eval_chexbert_class.py arguments (with defaults matching the script)
OUTPUT_MODE="per-file"
UNCERTAIN_MODE="rrg-"
BOOTSTRAP_CI=false
N_RESAMPLES=500
SEED=3
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        # Runner-specific flags
        --dry-run)            DRY_RUN=true; shift ;;
        --experiment)         FILTER_EXPERIMENT="$2"; shift 2 ;;
        --model)              FILTER_MODEL="$2"; shift 2 ;;
        --parallel)           MAX_PARALLEL="$2"; shift 2 ;;
        # run_eval_chexbert_class.py flags (with value)
        --output-mode)        OUTPUT_MODE="$2"; shift 2 ;;
        --uncertain-mode)     UNCERTAIN_MODE="$2"; shift 2 ;;
        --n-resamples)        N_RESAMPLES="$2"; shift 2 ;;
        --seed)               SEED="$2"; shift 2 ;;
        # run_eval_chexbert_class.py boolean flags
        --bootstrap-ci)       BOOTSTRAP_CI=true; shift ;;
        --skip-existing)      SKIP_EXISTING=true; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with no args to see usage in the script header."
            exit 1 ;;
    esac
done

# Validate --experiment if provided
ALLOWED_EXPERIMENTS=("doco" "oco" "ro")
if [[ -n "$FILTER_EXPERIMENT" ]]; then
    valid=false
    for exp in "${ALLOWED_EXPERIMENTS[@]}"; do
        [[ "$FILTER_EXPERIMENT" == "$exp" ]] && valid=true && break
    done
    if ! $valid; then
        echo "Error: --experiment must be one of: ${ALLOWED_EXPERIMENTS[*]}"
        exit 1
    fi
fi

# Build the eval args string forwarded to run_eval_chexbert_class.py
EVAL_ARGS="--output-mode ${OUTPUT_MODE} --uncertain-mode ${UNCERTAIN_MODE} --n-resamples ${N_RESAMPLES} --seed ${SEED}"
$BOOTSTRAP_CI    && EVAL_ARGS="${EVAL_ARGS} --bootstrap-ci"

# ---------------------
# Setup
# ---------------------
PROJECT_DIR="$(pwd)"

# Load module and venv
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module purge
    module load Python/3.10.4-GCCcore-11.3.0
fi

if [[ ! -d .venv_eval ]]; then
    echo "ERROR: .venv_eval not found. Create it with: python -m venv .venv_eval && pip install -r requirements_eval.txt"
    exit 1
fi
source .venv_eval/bin/activate

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/.models_cache"



OUTPUTS_DIR="outputs"

if [[ ! -d "${OUTPUTS_DIR}" ]]; then
    echo "ERROR: Outputs directory '${OUTPUTS_DIR}' not found. Run model inference first."
    exit 1
fi

echo "============================================"
echo "  CheXbert Per-Class Evaluation Multi-Runner"
echo "============================================"
echo "Project:       ${PROJECT_DIR}"
echo "Outputs dir:   ${OUTPUTS_DIR}"
echo "Experiments:   ${FILTER_EXPERIMENT:-doco+oco+ro}"
echo "Filter model:  ${FILTER_MODEL:-all}"
echo "Parallel:      ${MAX_PARALLEL}"
echo "Dry run:       ${DRY_RUN}"
echo ""
echo "Eval args:     ${EVAL_ARGS}"
echo "Start time:    $(date)"
echo "============================================"
echo ""

# ---------------------
# Discover JSON files grouped by experiment/percentage
# ---------------------
TOTAL=0
DONE=0
FAILED=0

# Collect all JSON files, sorted for reproducible ordering
mapfile -t ALL_FILES < <(find "${OUTPUTS_DIR}" -name "*.json" -type f | sort)

# Filter to only doco, oco, ro experiments (and optionally a single one)
FILTERED=()
for f in "${ALL_FILES[@]}"; do
    exp=$(echo "${f#${OUTPUTS_DIR}/}" | cut -d'/' -f1)
    if [[ -n "$FILTER_EXPERIMENT" ]]; then
        [[ "$exp" == "$FILTER_EXPERIMENT" ]] && FILTERED+=("$f")
    else
        for allowed in "${ALLOWED_EXPERIMENTS[@]}"; do
            [[ "$exp" == "$allowed" ]] && FILTERED+=("$f") && break
        done
    fi
done
ALL_FILES=("${FILTERED[@]}")

# Filter by model pattern if requested (matches against filename)
if [[ -n "$FILTER_MODEL" ]]; then
    FILTERED=()
    for f in "${ALL_FILES[@]}"; do
        filename=$(basename "$f")
        if [[ "$filename" == *"${FILTER_MODEL}"* ]]; then
            FILTERED+=("$f")
        fi
    done
    ALL_FILES=("${FILTERED[@]}")
fi

# Skip existing results if requested
if $SKIP_EXISTING; then
    AFTER_SKIP=()
    for f in "${ALL_FILES[@]}"; do
        rel="${f#${OUTPUTS_DIR}/}"
        base_no_ext="${rel%.json}"
        result_csv="results/${base_no_ext}.csv"
        if [[ -f "$result_csv" ]]; then
            echo "[SKIP] ${result_csv} already exists."
        else
            AFTER_SKIP+=("$f")
        fi
    done
    ALL_FILES=("${AFTER_SKIP[@]}")
fi

TOTAL=${#ALL_FILES[@]}
echo "Found ${TOTAL} JSON files to evaluate."
echo ""

# ---------------------
# Group files by experiment/percentage for display
# ---------------------
declare -A GROUP_COUNTS
for f in "${ALL_FILES[@]}"; do
    rel="${f#${OUTPUTS_DIR}/}"
    exp=$(echo "$rel" | cut -d'/' -f1)
    second=$(echo "$rel" | cut -d'/' -f2)

    if [[ "$second" =~ ^p[0-9]+ ]]; then
        group="${exp}/${second}"
    else
        group="${exp}"
    fi

    GROUP_COUNTS["$group"]=$(( ${GROUP_COUNTS["$group"]:-0} + 1 ))
done

echo "Groups:"
for group in $(echo "${!GROUP_COUNTS[@]}" | tr ' ' '\n' | sort); do
    echo "  ${group}: ${GROUP_COUNTS[$group]} files"
done
echo ""

# ---------------------
# Run evaluations (parallel, up to MAX_PARALLEL at a time)
# ---------------------
CURRENT_GROUP=""
LOG_DIR="logs/eval"
mkdir -p "${LOG_DIR}"
STATUS_DIR=$(mktemp -d "${LOG_DIR}/status_XXXXXX")

# wait_for_slot: block until fewer than MAX_PARALLEL background jobs are running
wait_for_slot() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
        sleep 1
    done
}

# run_one_eval: run a single evaluation in the background, write exit status to a file
run_one_eval() {
    local idx="$1" filepath="$2"
    local logfile="${LOG_DIR}/eval_chexbert_class_${idx}.log"

    if python evaluations/run_eval_chexbert_class.py \
        --filepath "$filepath" \
        ${EVAL_ARGS} \
        > "$logfile" 2>&1; then
        echo "0" > "${STATUS_DIR}/${idx}"
    else
        echo "1" > "${STATUS_DIR}/${idx}"
    fi
}

for f in "${ALL_FILES[@]}"; do
    rel="${f#${OUTPUTS_DIR}/}"
    exp=$(echo "$rel" | cut -d'/' -f1)
    second=$(echo "$rel" | cut -d'/' -f2)

    if [[ "$second" =~ ^p[0-9]+ ]]; then
        group="${exp}/${second}"
    else
        group="${exp}"
    fi

    # Print group header on change
    if [[ "$group" != "$CURRENT_GROUP" ]]; then
        echo "--------------------------------------------"
        echo "  ${group}"
        echo "--------------------------------------------"
        CURRENT_GROUP="$group"
    fi

    DONE=$((DONE + 1))
    filename=$(basename "$f")

    if $DRY_RUN; then
        echo "[${DONE}/${TOTAL}] ${filename} [DRY-RUN] python evaluations/run_eval_chexbert_class.py --filepath ${f} ${EVAL_ARGS}"
        continue
    fi

    echo "[${DONE}/${TOTAL}] ${filename} -> logs/eval/eval_chexbert_class_${DONE}.log"

    wait_for_slot
    run_one_eval "$DONE" "$f" &
done

# Wait for all remaining background jobs
wait

# ---------------------
# Collect results
# ---------------------
for status_file in "${STATUS_DIR}"/*; do
    [[ -f "$status_file" ]] || continue
    if [[ "$(cat "$status_file")" != "0" ]]; then
        idx=$(basename "$status_file")
        FAILED=$((FAILED + 1))
        echo "FAILED: eval_chexbert_class_${idx}.log"
    fi
done
rm -rf "$STATUS_DIR"

# ---------------------
# Summary
# ---------------------
echo ""
echo "============================================"
echo "  Summary"
echo "============================================"
echo "Total files:   ${TOTAL}"
echo "Parallel:      ${MAX_PARALLEL}"
echo "Succeeded:     $((TOTAL - FAILED))"
echo "Failed:        ${FAILED}"
echo "End time:      $(date)"
echo "Per-file logs: ${LOG_DIR}/eval_chexbert_class_<N>.log"
echo "============================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "WARNING: ${FAILED} evaluations failed. Inspect individual logs:"
    for status_file_path in "${LOG_DIR}"/eval_chexbert_class_*.log; do
        if grep -q "Traceback\|Error\|FAILED" "$status_file_path" 2>/dev/null; then
            echo "  - $status_file_path"
        fi
    done
    exit 1
fi
