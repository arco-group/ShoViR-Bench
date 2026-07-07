#!/bin/bash
#SBATCH -A NAISS2025-5-662
#SBATCH -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J eval_weighted
#SBATCH -o logs/eval/eval_weighted_%j.out
#SBATCH -e logs/eval/eval_weighted_%j.err

# Weighted-average evaluation across target categories (oco/doco/ro experiments).
# Each sample carries a target_category field; metrics are computed per category
# then aggregated as a weighted average by sample count.
#
# Runner-specific flags:
#   --dry-run              Preview commands without running
#   --experiment <name>    Only evaluate files from this experiment (e.g. oco, doco, ro)
#   --model <pattern>      Only evaluate files matching this pattern (e.g. gpt-5.4, gemini)
#   --parallel <N>         Max parallel evaluations (default: 4)
#   --skip-existing        Skip if _weighted.csv already exists
#
# Forwarded to run_eval_weighted.py:
#   --scorers              Comma-separated scorers (default: CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L)
#
# Usage:
#   sbatch scripts/run_weighted_evals.sh                              # all oco/doco/ro files
#   sbatch scripts/run_weighted_evals.sh --model gpt-5.4             # only GPT-5.4
#   sbatch scripts/run_weighted_evals.sh --experiment oco --model gpt-5.4
#   bash   scripts/run_weighted_evals.sh --dry-run
#   bash   scripts/run_weighted_evals.sh --model gpt-5.4 --dry-run

set -euo pipefail

DRY_RUN=false
FILTER_EXPERIMENT=""
FILTER_MODEL=""
MAX_PARALLEL=4
SKIP_EXISTING=false
SCORERS="CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L"

OCCLUSION_EXPERIMENTS=("ro" "doco" "oco")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true; shift ;;
        --experiment)     FILTER_EXPERIMENT="$2"; shift 2 ;;
        --model)          FILTER_MODEL="$2"; shift 2 ;;
        --parallel)       MAX_PARALLEL="$2"; shift 2 ;;
        --skip-existing)  SKIP_EXISTING=true; shift ;;
        --scorers)        SCORERS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with no args to see usage in the script header."
            exit 1 ;;
    esac
done

# ---------------------
# Setup
# ---------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
[[ ! -d "${PROJECT_DIR}/evaluations" ]] && PROJECT_DIR="$PWD"
cd "$PROJECT_DIR"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module purge
    module load Python/3.10.4-GCCcore-11.3.0
fi

if [[ ! -d .venv_eval ]]; then
    echo "ERROR: .venv_eval not found."
    exit 1
fi
source .venv_eval/bin/activate

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_DIR}/.models_cache"

OUTPUTS_DIR="outputs"
[[ ! -d "${OUTPUTS_DIR}" ]] && echo "ERROR: outputs/ not found." && exit 1

echo "============================================"
echo "  Weighted-Average Evaluation Runner"
echo "============================================"
echo "Project:       ${PROJECT_DIR}"
echo "Filter exp:    ${FILTER_EXPERIMENT:-oco+doco+ro}"
echo "Filter model:  ${FILTER_MODEL:-all}"
echo "Parallel:      ${MAX_PARALLEL}"
echo "Skip existing: ${SKIP_EXISTING}"
echo "Scorers:       ${SCORERS}"
echo "Dry run:       ${DRY_RUN}"
echo "Start time:    $(date)"
echo "============================================"
echo ""

# ---------------------
# Discover JSON files (only oco/doco/ro experiments)
# ---------------------
mapfile -t ALL_FILES < <(find "${OUTPUTS_DIR}" -name "*.json" -type f | sort)

# Keep only occlusion experiments
FILTERED=()
for f in "${ALL_FILES[@]}"; do
    exp=$(echo "${f#${OUTPUTS_DIR}/}" | cut -d'/' -f1)
    if [[ -n "$FILTER_EXPERIMENT" ]]; then
        [[ "$exp" == "$FILTER_EXPERIMENT" ]] && FILTERED+=("$f")
    else
        for occ in "${OCCLUSION_EXPERIMENTS[@]}"; do
            [[ "$exp" == "$occ" ]] && FILTERED+=("$f") && break
        done
    fi
done
ALL_FILES=("${FILTERED[@]}")

# Filter by model pattern
if [[ -n "$FILTER_MODEL" ]]; then
    FILTERED=()
    for f in "${ALL_FILES[@]}"; do
        [[ "$(basename "$f")" == *"${FILTER_MODEL}"* ]] && FILTERED+=("$f")
    done
    ALL_FILES=("${FILTERED[@]}")
fi

# Skip existing
if $SKIP_EXISTING; then
    FILTERED=()
    for f in "${ALL_FILES[@]}"; do
        rel="${f#${OUTPUTS_DIR}/}"
        weighted_csv="results/${rel%.json}_weighted.csv"
        if [[ -f "$weighted_csv" ]]; then
            echo "[SKIP] $weighted_csv already exists."
        else
            FILTERED+=("$f")
        fi
    done
    ALL_FILES=("${FILTERED[@]}")
fi

TOTAL=${#ALL_FILES[@]}
echo "Found ${TOTAL} JSON files to evaluate."
echo ""

# ---------------------
# Run (parallel)
# ---------------------
LOG_DIR="logs/eval"
mkdir -p "${LOG_DIR}"
STATUS_DIR=$(mktemp -d "${LOG_DIR}/status_XXXXXX")
DONE=0
FAILED=0

wait_for_slot() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do sleep 1; done
}

run_one() {
    local idx="$1" filepath="$2"
    local logfile="${LOG_DIR}/eval_weighted_${idx}.log"
    if python evaluations/run_eval_weighted.py \
        --filepath "$filepath" \
        --scorers "${SCORERS}" \
        > "$logfile" 2>&1; then
        echo "0" > "${STATUS_DIR}/${idx}"
    else
        echo "1" > "${STATUS_DIR}/${idx}"
    fi
}

for f in "${ALL_FILES[@]}"; do
    DONE=$((DONE + 1))
    filename=$(basename "$f")
    if $DRY_RUN; then
        echo "[${DONE}/${TOTAL}] ${filename} [DRY-RUN] python evaluations/run_eval_weighted.py --filepath ${f} --scorers ${SCORERS}"
        continue
    fi
    echo "[${DONE}/${TOTAL}] ${filename} -> ${LOG_DIR}/eval_weighted_${DONE}.log"
    wait_for_slot
    run_one "$DONE" "$f" &
done

wait

# ---------------------
# Collect results
# ---------------------
for status_file in "${STATUS_DIR}"/*; do
    [[ -f "$status_file" ]] || continue
    if [[ "$(cat "$status_file")" != "0" ]]; then
        FAILED=$((FAILED + 1))
        echo "FAILED: eval_weighted_$(basename "$status_file").log"
    fi
done
rm -rf "$STATUS_DIR"

echo ""
echo "============================================"
echo "  Summary"
echo "============================================"
echo "Total files:  ${TOTAL}"
echo "Succeeded:    $((TOTAL - FAILED))"
echo "Failed:       ${FAILED}"
echo "End time:     $(date)"
echo "============================================"

[[ $FAILED -gt 0 ]] && exit 1 || exit 0
