#!/usr/bin/env bash
# The method x mapping study: both training methods, each with both glove mappings, on the same clips,
# all evaluated with the same metrics (scripts/policy/common.py: evaluate_policy). Sequential runs.
#
# Usage:   launch/compare_methods.sh PREFIX      (runs PREFIX_{dextag,recon}_{layout,anatomical})
# Knobs:   DEXTAG_ITERATIONS=15000 RECON_ITERATIONS=1500 DEXTAG_ENVS=1024 RECON_ENVS=2048 SPLIT=train
set -eo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
P="${1:?usage: launch/compare_methods.sh PREFIX}"
ITERATIONS="${DEXTAG_ITERATIONS:-15000}" NUM_ENVS="${DEXTAG_ENVS:-1024}" ENV_NAME=bimanual_dextag \
    "$HERE/train_dextag.sh" "${P}_dextag_layout"
ITERATIONS="${DEXTAG_ITERATIONS:-15000}" NUM_ENVS="${DEXTAG_ENVS:-1024}" ENV_NAME=bimanual_dextag_anatomical \
    "$HERE/train_dextag.sh" "${P}_dextag_anatomical"
ITERATIONS="${RECON_ITERATIONS:-1500}" NUM_ENVS="${RECON_ENVS:-2048}" ENV_NAME=bimanual_recon \
    "$HERE/train_recon.sh" "${P}_recon_anatomical"
ITERATIONS="${RECON_ITERATIONS:-1500}" NUM_ENVS="${RECON_ENVS:-2048}" ENV_NAME=bimanual_recon_layout \
    "$HERE/train_recon.sh" "${P}_recon_layout"
source "$HERE/config.sh"
"$PY_SIM" "$REPO/scripts/policy/summarize_runs.py" "${P}_dextag_layout" "${P}_dextag_anatomical" \
    "${P}_recon_anatomical" "${P}_recon_layout"
