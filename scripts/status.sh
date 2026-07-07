#!/bin/bash
# Check status of experiment jobs: SLURM queue + output file completeness
#
# Usage:
#   ./scripts/status.sh <experiment>           # e.g. oco, baseline, ro, doco, all_noise
#   ./scripts/status.sh <experiment> [dataset]  # e.g. oco padchest-gr
#
# Can also be called from per-experiment wrappers:
#   ./scripts/oco/status.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

EXPERIMENT="${1:?Usage: $0 <experiment> [dataset]}"
DATASET_FILTER="${2:-}"

# ---------------------
# Expected models (short key -> output file pattern)
# ---------------------
declare -A MODEL_PATTERNS
MODEL_PATTERNS=(
    [medgemma]="medgemma"
    [maira2]="maira2"
    [cxrmateed]="cxrmateed"
    [nv_reason_cxr]="nv_reason"
    [chexone]="chexone"
    [libra]="libra_default"
    [llavarad]="llavarad"
    [chexagent]="chexagent"
    [radialog]="radialog"
)

MODELS=(medgemma maira2 cxrmateed nv_reason_cxr chexone libra llavarad chexagent radialog)

OUTPUTS_DIR="${PROJECT_DIR}/outputs/${EXPERIMENT}"
LOGS_DIR="${PROJECT_DIR}/logs/${EXPERIMENT}"

echo "============================================"
echo "  Status: ${EXPERIMENT}"
echo "============================================"
echo ""

# ---------------------
# 1. SLURM queue
# ---------------------
echo "--- SLURM Jobs ---"
SLURM_JOBS=$(squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null | grep "_${EXPERIMENT}" || true)
if [[ -n "$SLURM_JOBS" ]]; then
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null | head -1
    echo "$SLURM_JOBS"
else
    echo "  No running/pending jobs for *_${EXPERIMENT}"
fi
echo ""

# ---------------------
# 2. Output file completeness
# ---------------------
if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "Output directory not found: ${OUTPUTS_DIR}"
    echo ""
    echo "============================================"
    exit 0
fi

# Discover percentage subdirs (for oco/ro/doco) or flat structure (baseline/all_noise)
LEVELS=()
for d in "$OUTPUTS_DIR"/p[0-9]*/; do
    [[ -d "$d" ]] && LEVELS+=("$(basename "$d")")
done

if [[ ${#LEVELS[@]} -eq 0 ]]; then
    LEVELS=(".")
fi

# Sort levels naturally
IFS=$'\n' LEVELS=($(sort -V <<<"${LEVELS[*]}")); unset IFS

# Discover datasets
DATASETS=()
for level in "${LEVELS[@]}"; do
    if [[ "$level" == "." ]]; then
        search_dir="$OUTPUTS_DIR"
    else
        search_dir="$OUTPUTS_DIR/$level"
    fi
    for dd in "$search_dir"/*/; do
        [[ -d "$dd" ]] || continue
        ds=$(basename "$dd")
        if [[ -n "$DATASET_FILTER" && "$ds" != "$DATASET_FILTER" ]]; then
            continue
        fi
        # Add if not already in list
        local_found=false
        for existing in "${DATASETS[@]+"${DATASETS[@]}"}"; do
            [[ "$existing" == "$ds" ]] && local_found=true
        done
        $local_found || DATASETS+=("$ds")
    done
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No output datasets found."
    exit 0
fi

# ---------------------
# Build results matrix: MODEL_RESULTS[model|level|dataset] = 0 or 1
# ---------------------
declare -A MODEL_RESULTS
TOTAL_EXPECTED=0
TOTAL_FOUND=0

for level in "${LEVELS[@]}"; do
    if [[ "$level" == "." ]]; then
        level_dir="$OUTPUTS_DIR"
    else
        level_dir="$OUTPUTS_DIR/$level"
    fi

    for dataset in "${DATASETS[@]}"; do
        dataset_dir="${level_dir}/${dataset}"
        [[ -d "$dataset_dir" ]] || continue

        for model in "${MODELS[@]}"; do
            pattern="${MODEL_PATTERNS[$model]}"
            match=$(find "$dataset_dir" -maxdepth 1 -name "*${pattern}*" -type f 2>/dev/null | head -1)
            key="${model}|${level}|${dataset}"
            TOTAL_EXPECTED=$((TOTAL_EXPECTED + 1))
            if [[ -n "$match" ]]; then
                MODEL_RESULTS["$key"]=1
                TOTAL_FOUND=$((TOTAL_FOUND + 1))
            else
                MODEL_RESULTS["$key"]=0
            fi
        done
    done
done

# ---------------------
# Print table per dataset
# ---------------------
for dataset in "${DATASETS[@]}"; do
    echo "--- ${dataset} ---"
    echo ""

    if [[ "${LEVELS[0]}" == "." ]]; then
        # Flat: just show model | status
        printf "  %-16s %s\n" "MODEL" "STATUS"
        printf "  %-16s %s\n" "----------------" "------"
        for model in "${MODELS[@]}"; do
            key="${model}|.|${dataset}"
            if [[ "${MODEL_RESULTS[$key]:-0}" == "1" ]]; then
                mark="OK"
            else
                mark="--"
            fi
            printf "  %-16s %s\n" "$model" "$mark"
        done
    else
        # Table: models as rows, levels as columns
        # Header
        printf "  %-16s" "MODEL"
        for level in "${LEVELS[@]}"; do
            printf " %5s" "$level"
        done
        printf "  %s\n" "TOTAL"

        # Separator
        printf "  %-16s" "----------------"
        for level in "${LEVELS[@]}"; do
            printf " %s" "-----"
        done
        printf "  %s\n" "-----"

        # Rows
        for model in "${MODELS[@]}"; do
            printf "  %-16s" "$model"
            model_found=0
            model_total=0
            for level in "${LEVELS[@]}"; do
                key="${model}|${level}|${dataset}"
                if [[ "${MODEL_RESULTS[$key]:-}" == "1" ]]; then
                    printf "    %s" "Y"
                    model_found=$((model_found + 1))
                elif [[ "${MODEL_RESULTS[$key]:-}" == "0" ]]; then
                    printf "    %s" "-"
                else
                    printf "    %s" "."
                fi
                model_total=$((model_total + 1))
            done
            printf "  %s/%s\n" "$model_found" "$model_total"
        done

        # Footer: totals per level
        printf "  %-16s" "TOTAL"
        for level in "${LEVELS[@]}"; do
            level_found=0
            for model in "${MODELS[@]}"; do
                key="${model}|${level}|${dataset}"
                [[ "${MODEL_RESULTS[$key]:-0}" == "1" ]] && level_found=$((level_found + 1))
            done
            printf " %3s/%-1s" "$level_found" "${#MODELS[@]}"
        done
        echo ""
    fi
    echo ""
done

# ---------------------
# 3. Recent log activity
# ---------------------
echo "--- Recent Logs ---"
if [[ -d "$LOGS_DIR" ]]; then
    latest=$(find "$LOGS_DIR" -name "*.out" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -5)
    if [[ -n "$latest" ]]; then
        while IFS= read -r line; do
            file=$(echo "$line" | cut -d' ' -f2-)
            mod_time=$(stat -c '%y' "$file" 2>/dev/null | cut -d'.' -f1)
            echo "  ${mod_time}  $(basename "$file")"
        done <<< "$latest"
    else
        echo "  No log files found in ${LOGS_DIR}/"
    fi
else
    echo "  Log directory not found: ${LOGS_DIR}/"
fi

echo ""
echo "============================================"
echo "  Summary: ${TOTAL_FOUND}/${TOTAL_EXPECTED} output files present"
echo "============================================"
