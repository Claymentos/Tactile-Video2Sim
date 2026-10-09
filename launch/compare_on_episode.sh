#!/usr/bin/env bash
# Both RL methods on the clips of ONE HumanTouch episode (one video), then one comparison table.
#
# rl_recon fits one demonstration by design; DexTaG is trained here on the same clips so the two are
# compared on equal data. Each method keeps its own glove mapping (DexTaG: layout, rl_recon: anatomical),
# and its training aids are rescaled to the iteration budget (DexTaG's object guide decays over the first
# half, its perturbations ramp between 30 % and 80 %; rl_recon's assist decays over the first half), so
# that the evaluated policies (no aids) are the ones the aids were removed from. Every checkpoint of both
# runs is evaluated with the same metrics and the best one is reported.
#
# Usage:   launch/compare_on_episode.sh EPISODE [PREFIX]          e.g. launch/compare_on_episode.sh 2
# Knobs:   NUM_ENVS=1024  DEXTAG_ITERATIONS=1500  RECON_ITERATIONS=500  SAVE_INTERVAL=50
#          SKIP_DEXTAG=1  SKIP_RECON=1
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
EP="${1:?usage: launch/compare_on_episode.sh EPISODE [PREFIX]}"
P="${2:-ep$(printf %06d "$EP")}"
CLIPS=$("$PY_SIM" -c "
import json, sys
idx = json.load(open('$DATA_DIR/clips/index.json'))['clips']
names = [c['name'] for c in idx if c['episode'] == $EP]
sys.exit('no clip for episode $EP: run launch/prepare_humantouch.sh $EP') if not names else print(','.join(names))")
N_CLIPS=$(echo "$CLIPS" | tr ',' '\n' | wc -l)
NUM_ENVS="${NUM_ENVS:-1024}"
DI="${DEXTAG_ITERATIONS:-1500}"
RI="${RECON_ITERATIONS:-500}"
SAVE="${SAVE_INTERVAL:-50}"
FILTER='^profiling\|ast.Str\|return index ==\|Falling back to convex hull\|qpos0) exceeds'
echo "episode $EP: $N_CLIPS clips ($CLIPS), $NUM_ENVS envs"

if [ "${SKIP_DEXTAG:-0}" != "1" ]; then
    echo "=== DexTaG: ${P}_dextag ($DI iterations) ==="
    "$PY_SIM" scripts/policy/run_ppo_dextag.py --exp_name="${P}_dextag" --num_envs="$NUM_ENVS" \
        --env.data.clips="$CLIPS" --runner.total_iterations="$DI" --runner.save_interval="$SAVE" \
        --env.object_guide_decay_steps=$((DI / 2)) \
        --env.objects.perturbation_curriculum="$((DI * 3 / 10)),$((DI * 8 / 10))" \
        2>&1 | grep --line-buffered -v "$FILTER"
    "$PY_SIM" scripts/policy/run_ppo_dextag.py --exp_name="${P}_dextag" --eval --all_checkpoints \
        2>&1 | grep -v "$FILTER"
fi
if [ "${SKIP_RECON:-0}" != "1" ]; then
    echo "=== rl_recon: ${P}_recon ($RI iterations) ==="
    "$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="${P}_recon" --num_envs="$NUM_ENVS" \
        --env.data.clips="$CLIPS" --runner.total_iterations="$RI" --runner.save_interval="$SAVE" \
        2>&1 | grep --line-buffered -v "$FILTER"
    "$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="${P}_recon" --eval --all_checkpoints \
        2>&1 | grep -v "$FILTER"
fi
echo "=== comparison (best checkpoint of each run) ==="
"$PY_SIM" scripts/policy/summarize_runs.py "${P}_dextag" "${P}_recon"
