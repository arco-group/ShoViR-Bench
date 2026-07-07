#!/bin/bash
# Check status of all RO experiments
# Usage: ./scripts/ro/status.sh [dataset]
exec "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/status.sh" ro "$@"
