#!/bin/bash
# Submit compute-stats jobs for all models to SLURM.
# Each job loads the model in its correct venv, measures params / FLOPs / MACs
# / throughput on 5 synthetic images, and writes results/compute_stats/<model>.json.
#
# Usage:
#   ./scripts/compute_stats/submit_all.sh [--dry-run] [--skip-existing]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT"

mkdir -p logs

# ── Defaults ────────────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true;       shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--dry-run] [--skip-existing]"; exit 1 ;;
    esac
done

# ── All supported models ─────────────────────────────────────────────────────
MODELS=(
    medgemma
    maira-2
    cxrmateed
    chexagent
    libra
    llavarad
    radialog
    nv-reason-cxr-3b
    chexone
)

# ── Helpers ──────────────────────────────────────────────────────────────────
_json_exists() {
    [[ -f "results/compute_stats/${1}.json" ]]
}

# ── Header ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then echo "=== DRY RUN MODE ==="; echo ""; fi

echo "=== Compute-Stats Jobs ==="
echo "Root:   ${ROOT}"
echo "Output: results/compute_stats/"
echo ""
echo "Models:"
for m in "${MODELS[@]}"; do echo "  - ${m}"; done
echo ""

# ── Submit ───────────────────────────────────────────────────────────────────
echo "Submitting..."
JOB_IDS=()

for model in "${MODELS[@]}"; do
    if $SKIP_EXISTING && _json_exists "$model"; then
        echo "  [SKIP] ${model} — results/compute_stats/${model}.json already exists"
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY]  Would submit: MODEL=${model} bash ${SCRIPT_DIR}/run_compute_stats.sh"
    else
        job_id=$(MODEL="$model" bash "${SCRIPT_DIR}/run_compute_stats.sh" | grep -oP '(?<=batch job )\d+')
        JOB_IDS+=("$job_id")
        echo "  [OK]   ${model} — Job ID: ${job_id}"
    fi
done

echo ""
if ! $DRY_RUN && [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    echo "=== Submitted ${#JOB_IDS[@]} jobs ==="
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor:  squeue -u \$USER"
    echo "Logs:     tail -f logs/compute_stats_*_<jobid>.out"
    echo "Cancel:   scancel ${JOB_IDS[*]}"
    echo ""
    echo "Once all jobs finish, generate plots with:"
    echo "  source .venv_eval/bin/activate"
    echo "  python -m src.postprocessing.plots.compute_efficiency"
fi
