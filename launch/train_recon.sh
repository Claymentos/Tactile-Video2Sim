#!/usr/bin/env bash
# Benchmark-Genesis' RL reconstruction (rl_recon residual policy) on HumanTouch, bimanual. Then scores
# every checkpoint and keeps the best one as model_best.pt (rl_recon's recon_rsl_eval.py --all).
#
# Usage:   launch/train_recon.sh EXP_NAME [extra overrides, e.g. --env.tactile.mapping=layout]
# Knobs:   NUM_ENVS=2048  ITERATIONS=1500  ENV_NAME=bimanual_recon (or bimanual_recon_layout)
#          TACTILE=0   ablation: no tactile reward        GLOVE_OBS=0  ablation: glove hidden from the policy
#          ASSIST=0    no hand-anchored object assist      SPLIT=train  train / eval split
#          SKIP_TRAIN=1  SKIP_EVAL=1
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
EXP="${1:?usage: launch/train_recon.sh EXP_NAME [overrides]}"; shift
ARGS=()
[ "${TACTILE:-1}" = "0" ] && ARGS+=(--reward_args.TactileContactF1Reward.scale=0)
[ "${GLOVE_OBS:-1}" = "0" ] && ARGS+=(--env.glove_obs=False)
[ "${ASSIST:-1}" = "0" ] && ARGS+=(--env.assist_init=0)
EVAL_SPLIT=()
if [ -n "${SPLIT:-}" ]; then ARGS+=(--env.data.split_mode="$SPLIT"); EVAL_SPLIT=(--split_mode=eval); fi
FILTER='^profiling\|ast.Str\|return index ==\|Falling back to convex hull\|qpos0) exceeds'
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== rl_recon training: $EXP ${ARGS[*]} ==="
    "$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="$EXP" --env_name="${ENV_NAME:-bimanual_recon}" \
        --num_envs="${NUM_ENVS:-2048}" --runner.total_iterations="${ITERATIONS:-1500}" "${ARGS[@]}" "$@" \
        2>&1 | grep --line-buffered -v "$FILTER"
fi
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo "=== rl_recon evaluation (all checkpoints -> model_best.pt): $EXP ==="
    "$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="$EXP" --eval --all_checkpoints "${EVAL_SPLIT[@]}" \
        2>&1 | grep -v "$FILTER"
fi
