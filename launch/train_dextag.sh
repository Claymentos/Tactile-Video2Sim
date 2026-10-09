#!/usr/bin/env bash
# DexTaG (tactile-as-guidance) teacher on HumanTouch, bimanual. Then evaluates the last checkpoint.
#
# Usage:   launch/train_dextag.sh EXP_NAME [extra overrides, e.g. --env.tactile.mapping=anatomical]
# Knobs:   NUM_ENVS=1024  ITERATIONS=15000  ENV_NAME=bimanual_dextag (or bimanual_dextag_anatomical)
#          SPLIT=train (train on the train split, evaluate on the eval split; default: all clips)
#          SKIP_TRAIN=1  SKIP_EVAL=1
# Logs:    logs/EXP_NAME/<timestamp>/{configs,checkpoints,progress.csv,tb,eval}
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
EXP="${1:?usage: launch/train_dextag.sh EXP_NAME [overrides]}"; shift
SPLIT_ARGS=(); EVAL_SPLIT=()
if [ -n "${SPLIT:-}" ]; then SPLIT_ARGS=(--env.data.split_mode="$SPLIT"); EVAL_SPLIT=(--split_mode=eval); fi
FILTER='^profiling\|ast.Str\|return index ==\|Falling back to convex hull\|qpos0) exceeds'
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== DexTaG training: $EXP ==="
    "$PY_SIM" scripts/policy/run_ppo_dextag.py --exp_name="$EXP" --env_name="${ENV_NAME:-bimanual_dextag}" \
        --num_envs="${NUM_ENVS:-1024}" --runner.total_iterations="${ITERATIONS:-15000}" "${SPLIT_ARGS[@]}" "$@" \
        2>&1 | grep --line-buffered -v "$FILTER"
fi
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo "=== DexTaG evaluation: $EXP ==="
    "$PY_SIM" scripts/policy/run_ppo_dextag.py --exp_name="$EXP" --eval "${EVAL_SPLIT[@]}" 2>&1 | grep -v "$FILTER"
fi
