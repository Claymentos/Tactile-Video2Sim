#!/usr/bin/env bash
# Policy-free comparison of the two glove -> Wuji mappings and signal models (kinematic replay).
# Writes outputs/tactile_comparison/{comparison.md,comparison.json}.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
"$PY_SIM" scripts/tactile/compare_tactile_mappings.py --clips "$DATA_DIR/clips" "$@" 2>&1 | grep -v "^profiling"
