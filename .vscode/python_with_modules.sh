#!/usr/bin/env bash
set -euo pipefail

echo "Loading modules for Python debugger..."
module purge
module load python/3.11.6_fixed   # <-- change these

# IMPORTANT: forward all args that VS Code/debugpy passes
exec python "$@"