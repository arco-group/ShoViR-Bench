#!/bin/bash
# Submit all All-Noise-Mean jobs for a given dataset
#
# Usage:
#   ./scripts/all_noise_mean/submit_all.sh --dataset padchest-gr [--dry-run] [--skip-existing] [--one-prompt]
#   ./scripts/all_noise_mean/submit_all.sh --dataset mimic-cxr [--dry-run] [--skip-existing] [--one-prompt]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET=""
DRY_RUN=false
SKIP_EXISTING=false
ONE_PROMPT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --one-prompt|--single-prompt) ONE_PROMPT=true; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 --dataset <name> [--dry-run] [--skip-existing] [--one-prompt]"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    echo "Usage: $0 --dataset padchest-gr [--dry-run] [--skip-existing] [--one-prompt]"
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
$ONE_PROMPT && FORWARD_ARGS+=(--one-prompt)

echo "============================================"
echo "  All Noise Mean - ${DATASET}"
echo "============================================"
echo ""

bash "$DATASET_SCRIPT" "${FORWARD_ARGS[@]}"

echo ""
echo "============================================"
echo "  Done"
echo "============================================"
