#!/bin/bash
# Check status of all Baseline experiments
# Usage: ./scripts/baseline/status.sh [dataset]
exec "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/status.sh" baseline "$@"
