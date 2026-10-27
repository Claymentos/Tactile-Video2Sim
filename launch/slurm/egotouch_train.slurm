#!/bin/bash
# rl_recon on one reference (one H100), wandb offline, then the evaluation of every checkpoint (launch/rl_recon.sh's
# steps 4-5). The runner itself renders a rollout video every --runner.video_interval iterations (default 500) and
# logs it to wandb (rollout/video). Submitted by launch/submit_arm_compare.sh (and launch/submit_traj_compare.sh).
#
# Env: REF (reference.npz)  EXP (logs/EXP)  ITERATIONS=2000  N_ENVS=2048  SAVE_EVERY=100
#      WANDB_RUN_GROUP  EXTRA (more run_ppo_recon.py arguments, e.g. --env.arm.enabled=True)
#SBATCH -J et_train
#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 10:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
: "${REF:?}" "${EXP:?}"
source launch/config.sh
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl TV2S_WANDB=1 WANDB_MODE=offline WANDB_SILENT=true
export WANDB_PROJECT="${WANDB_PROJECT:-tactile-video2sim}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
FILTER='^profiling\|ast.Str\|Falling back to convex hull'
# jobs starting together compiled Genesis' kernels into the same ~/.cache/quadrants and some then read a broken kernel
# (episodes of 1 step, fingers 17 cm off): each job works on its own copy of the cache
export QD_OFFLINE_CACHE_FILE_PATH="${JOBSCRATCH:-/tmp/$SLURM_JOB_ID}/quadrants"
mkdir -p "$QD_OFFLINE_CACHE_FILE_PATH"
cp -r "$HOME/.cache/quadrants/qdcache/." "$QD_OFFLINE_CACHE_FILE_PATH/" 2>/dev/null || true
echo "=== $EXP on $(hostname) ($SLURM_JOB_ID): $REF ${EXTRA:-} ==="

# shellcheck disable=SC2086
"$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="$EXP" --env.data.reference="$REF" --num_envs="${N_ENVS:-2048}" \
    --runner.total_iterations="${ITERATIONS:-2000}" --runner.save_interval="${SAVE_EVERY:-100}" ${EXTRA:-} 2>&1 \
    | grep --line-buffered -v "$FILTER"
echo "=== evaluation (every checkpoint, from frame 0 and in hand) ==="
"$PY_SIM" scripts/policy/eval_rl_recon.py --exp_name "$EXP" --all 2>&1 | grep --line-buffered -v "$FILTER"
echo "=== done: $EXP ==="
