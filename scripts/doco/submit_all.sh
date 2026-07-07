#!/bin/bash
# Submit all DOCO percentage levels (p00..p100) for a given dataset
#
# Usage:
#   ./scripts/doco/submit_all.sh --dataset padchest-gr [--seed 3] [--dry-run] [--skip-existing]
#
# This submits 9 models x 6 levels = 54 jobs total.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET=""
SEED="3"
DRY_RUN=false
SKIP_EXISTING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 --dataset <name> [--seed N] [--dry-run] [--skip-existing]"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    echo "Usage: $0 --dataset padchest-gr [--seed 3] [--dry-run] [--skip-existing]"
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

LEVELS=(p00 p20 p40 p60 p80 p100)

echo "============================================"
echo "  DOCO — All Levels for ${DATASET}"
echo "============================================"
echo "Levels: ${LEVELS[*]}"
echo "Seed:   ${SEED}"
echo ""

FORWARD_ARGS=()
FORWARD_ARGS+=(--seed "$SEED")
$DRY_RUN && FORWARD_ARGS+=(--dry-run)
$SKIP_EXISTING && FORWARD_ARGS+=(--skip-existing)

for level in "${LEVELS[@]}"; do
    echo ">>> doco_${level}"
    bash "$DATASET_SCRIPT" --experiment "doco_${level}" "${FORWARD_ARGS[@]}"
    echo ""
done

echo "============================================"
echo "  Done — submitted ${#LEVELS[@]} levels x 9 models"
echo "============================================"
