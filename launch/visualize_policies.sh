#!/usr/bin/env bash
# Watch trained policies: one video per clip, the human head-camera video on top, then the runs (egocentric + front views,
# the reference hand as green spheres, the simulated glove against the human one).
#
# Usage:   launch/visualize_policies.sh EXP_NAME[,EXP_NAME...] [extra args]
#          launch/visualize_policies.sh ep000002_dextag,ep000002_recon        -> outputs/videos/ep000002_dextag_vs_ep000002_recon/
# Extra:   --checkpoint=500 (default: model_best.pt, else the last one)  --clips=ep000002_clip03,...
#          --name=DIR  --res=640,480  --fps=15 (slow motion)
#          --viewer    interactive Genesis viewer instead of videos (first run only, loops until Ctrl-C)
#          --compose_only  rebuild the stacked videos (human video + runs) from the per-run videos
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
RUNS="${1:?usage: launch/visualize_policies.sh EXP_NAME[,EXP_NAME...] [extra args]}"
shift
"$PY_SIM" scripts/policy/visualize_policy.py --exp_names="$RUNS" "$@" \
    2>&1 | grep --line-buffered -v '^profiling\|ast.Str\|return index ==\|Falling back to convex hull\|qpos0) exceeds'
