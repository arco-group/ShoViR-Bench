#!/bin/bash
# Check status of all DOCO experiments
# Usage: ./scripts/doco/status.sh [dataset]
exec "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/status.sh" doco "$@"
