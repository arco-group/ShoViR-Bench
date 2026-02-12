#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J eval_all
#SBATCH -o logs/eval/eval_all_%j.out
#SBATCH -e logs/eval/eval_all_%j.err

# Evaluate all model outputs under outputs/.
# Groups results by experiment and percentage level (pXX).
#
# Runner-specific flags:
#   --dry-run              Preview commands without running
#   --experiment <name>    Only evaluate files from this experiment (e.g. ro, baseline)
#   --parallel <N>         Max parallel evaluations (default: 4)
#
# All run_eval.py flags are forwarded:
#   --output-mode          per-file (default) | per-experiment
#   --scorers              Comma-separated scorers (default: CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L)
#   --bootstrap-ci         Enable bootstrap confidence intervals
#   --save-breakdowns      Save CheXbert breakdown CSVs
#   --report-chexbert-f1   Save extra CheXbert detailed metrics CSV
#   --run-name             W&B run name
#   --compute-green        Compute GREEN metric
#   --green-model-name     HuggingFace model name for GREEN
#   --skip-existing        Skip evaluation if results already exist (do not override)
#
# Usage:
#   sbatch scripts/run_all_evals.sh                                    # run all, per-file
#   sbatch scripts/run_all_evals.sh --experiment ro                    # only ro experiment
#   sbatch scripts/run_all_evals.sh --bootstrap-ci --save-breakdowns   # with CI + breakdowns
#   bash   scripts/run_all_evals.sh --dry-run                          # preview
#   bash   scripts/run_all_evals.sh --experiment baseline --dry-run

set -euo pipefail

# ---------------------
# Parse arguments
# ---------------------
DRY_RUN=false
FILTER_EXPERIMENT=""
MAX_PARALLEL=4

# run_eval.py arguments (with defaults matching run_eval.py)
OUTPUT_MODE="per-file"
SCORERS="CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L"
BOOTSTRAP_CI=false
SAVE_BREAKDOWNS=false
REPORT_CHEXBERT_F1=false
RUN_NAME="mimic_cxr_eval"
COMPUTE_GREEN=false
GREEN_MODEL_NAME="StanfordAIMI/GREEN-radllama2-7b"
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        # Runner-specific flags
        --dry-run)            DRY_RUN=true; shift ;;
        --experiment)         FILTER_EXPERIMENT="$2"; shift 2 ;;
        --parallel)           MAX_PARALLEL="$2"; shift 2 ;;
        # run_eval.py flags (with value)
        --output-mode)        OUTPUT_MODE="$2"; shift 2 ;;
        --scorers)            SCORERS="$2"; shift 2 ;;
        --run-name)           RUN_NAME="$2"; shift 2 ;;
        --green-model-name)   GREEN_MODEL_NAME="$2"; shift 2 ;;
        # run_eval.py boolean flags
        --bootstrap-ci)       BOOTSTRAP_CI=true; shift ;;
        --save-breakdowns)    SAVE_BREAKDOWNS=true; shift ;;
        --report-chexbert-f1) REPORT_CHEXBERT_F1=true; shift ;;
        --compute-green)      COMPUTE_GREEN=true; shift ;;
        --skip-existing)      SKIP_EXISTING=true; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with no args to see usage in the script header."
            exit 1 ;;
    esac
done

# Build the eval args string forwarded to run_eval.py
EVAL_ARGS="--output-mode ${OUTPUT_MODE} --scorers ${SCORERS} --run-name ${RUN_NAME} --green-model-name ${GREEN_MODEL_NAME}"
$BOOTSTRAP_CI       && EVAL_ARGS="${EVAL_ARGS} --bootstrap-ci"
$SAVE_BREAKDOWNS    && EVAL_ARGS="${EVAL_ARGS} --save-breakdowns"
$REPORT_CHEXBERT_F1 && EVAL_ARGS="${EVAL_ARGS} --report-chexbert-f1"
$COMPUTE_GREEN      && EVAL_ARGS="${EVAL_ARGS} --compute-green"
$SKIP_EXISTING      && EVAL_ARGS="${EVAL_ARGS} --skip-existing"

# ---------------------
# Setup
# ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load module and venv (skip if not on SLURM or already activated)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
fi

if [[ -d .venv_eval ]]; then
    source .venv_eval/bin/activate
elif [[ -d .venv_RRG ]]; then
    source .venv_RRG/bin/activate
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/.models_cache"

mkdir -p logs/eval

OUTPUTS_DIR="outputs"

echo "============================================"
echo "  Evaluation Multi-Runner"
echo "============================================"
echo "Project:       ${PROJECT_DIR}"
echo "Outputs dir:   ${OUTPUTS_DIR}"
echo "Filter:        ${FILTER_EXPERIMENT:-all}"
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
SKIPPED=0

# Collect all JSON files, sorted for reproducible ordering
mapfile -t ALL_FILES < <(find "${OUTPUTS_DIR}" -name "*.json" -type f | sort)

# Filter by experiment if requested
if [[ -n "$FILTER_EXPERIMENT" ]]; then
    FILTERED=()
    for f in "${ALL_FILES[@]}"; do
        # experiment is the first directory component after outputs/
        exp=$(echo "$f" | sed "s|^${OUTPUTS_DIR}/||" | cut -d'/' -f1)
        if [[ "$exp" == "$FILTER_EXPERIMENT" ]]; then
            FILTERED+=("$f")
        fi
    done
    ALL_FILES=("${FILTERED[@]}")
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
    # Extract group: experiment or experiment/pXX
    # Path patterns:
    #   baseline/padchest-gr/file.json        -> baseline
    #   ro/p20/padchest-gr/file.json          -> ro/p20
    #   all_noise/padchest-gr/file.json       -> all_noise
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
    local logfile="${LOG_DIR}/eval_${idx}.log"

    if python evaluations/run_eval.py \
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
        echo "[${DONE}/${TOTAL}] ${filename} [DRY-RUN] python evaluations/run_eval.py --filepath ${f} ${EVAL_ARGS}"
        continue
    fi

    echo "[${DONE}/${TOTAL}] ${filename} -> logs/eval/eval_${DONE}.log"

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
        echo "FAILED: eval_${idx}.log"
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
echo "Per-file logs: ${LOG_DIR}/eval_<N>.log"
echo "============================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "WARNING: ${FAILED} evaluations failed. Inspect individual logs:"
    for status_file_path in "${LOG_DIR}"/eval_*.log; do
        # Show logs that ended with a traceback
        if grep -q "Traceback\|Error\|FAILED" "$status_file_path" 2>/dev/null; then
            echo "  - $status_file_path"
        fi
    done
    exit 1
fi
