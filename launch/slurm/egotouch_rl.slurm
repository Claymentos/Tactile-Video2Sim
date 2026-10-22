#!/bin/bash
# rl_recon on one EgoTouch reference (one H100), wandb offline: launch/rl_recon.sh's training + evaluation.
# Submitted by launch/submit_egotouch.sh (after egotouch_perception.slurm).
#
# Env: KEY (EgoTouch key)  NAME (data/references/NAME/reference.npz)  EXP (logs/EXP)  TACTILE (1|0)  ITERATIONS  N_ENVS
#      WANDB_RUN_GROUP  EXTRA (more run_ppo_recon.py arguments)
#SBATCH -J et_rl
#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
: "${KEY:?}" "${NAME:?}" "${EXP:?}" "${TACTILE:?}"
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl TV2S_WANDB=1 WANDB_MODE=offline WANDB_SILENT=true
export WANDB_PROJECT="${WANDB_PROJECT:-tactile-video2sim}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
TA=""; [ "$TACTILE" = 0 ] && TA="--env.tactile_reward=False"
echo "=== $EXP on $(hostname) ($SLURM_JOB_ID): reference $NAME, tactile reward $TACTILE ==="
SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 REFERENCE="data/references/$NAME/reference.npz" \
    ITERATIONS="${ITERATIONS:-500}" N_ENVS="${N_ENVS:-2048}" TRAIN_ARGS="$TA ${EXTRA:-}" \
    launch/rl_recon.sh "$EXP" "$KEY" 2>&1 | grep --line-buffered -v '^profiling\|ast.Str\|Falling back to convex hull'
echo "=== done: $EXP ==="
