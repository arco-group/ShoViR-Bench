#!/bin/bash
# Check status of all OCO experiments
# Usage: ./scripts/oco/status.sh [dataset]
exec "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/status.sh" oco "$@"
