#!/bin/bash
# Submit all Baseline jobs for a given dataset
#
# Usage:
#   ./scripts/baseline/submit_all.sh --dataset padchest-gr [--dry-run] [--skip-existing] [--single-prompt-baseline]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET=""
DRY_RUN=false
SKIP_EXISTING=false
SINGLE_PROMPT_BASELINE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --single-prompt-baseline) SINGLE_PROMPT_BASELINE=true; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 --dataset <name> [--dry-run] [--skip-existing] [--single-prompt-baseline]"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    echo "Usage: $0 --dataset padchest-gr [--dry-run] [--skip-existing] [--single-prompt-baseline]"
    echo ""
    echo "Available datasets:"
    for d in "$SCRIPT_DIR"/*/; do
        [[ -f "${d}submit_all.sh" ]] && echo "  $(basename "$d")"
    done
    exit 1
fi

DATASET_SCRIPT="${SCRIPT_DIR}/${DATASET}/submit_all.sh"
if [[ ! -f "$DATASET_SCRIPT" ]]; then
    echo "Error: submit_all.sh not found for dataset '${DATASET}'"
    echo "Expected: ${DATASET_SCRIPT}"
    exit 1
fi

FORWARD_ARGS=()
$DRY_RUN && FORWARD_ARGS+=(--dry-run)
$SKIP_EXISTING && FORWARD_ARGS+=(--skip-existing)
$SINGLE_PROMPT_BASELINE && FORWARD_ARGS+=(--single-prompt-baseline)

echo "============================================"
echo "  Baseline — ${DATASET}"
echo "============================================"
echo ""

bash "$DATASET_SCRIPT" "${FORWARD_ARGS[@]}"

echo ""
echo "============================================"
echo "  Done"
echo "============================================"
