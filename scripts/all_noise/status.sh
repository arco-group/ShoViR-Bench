#!/bin/bash
# Check status of all All-Noise experiments
# Usage: ./scripts/all_noise/status.sh [dataset]
exec "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/status.sh" all_noise "$@"
