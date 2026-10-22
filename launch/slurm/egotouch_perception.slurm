#!/bin/bash
# One EgoTouch demonstration -> reconstruction -> rl_recon reference (one H100). Submitted by launch/submit_egotouch.sh.
#
#   launch/rl_recon.sh NAME KEY, steps 2-3: launch/reconstruct.sh (SAM3, MoGe, camera, mocap, RecGen, CoTracker) and
#   scripts/rl/prepare_reference.py (reference.npz for the hand the scene says is used)
#
# Env: KEY (EgoTouch key, e.g. pick_up_hammer__20260321_085301_950)  NAME (data/references/NAME)  RECON_ARGS
#SBATCH -J et_perc
#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
: "${KEY:?}" "${NAME:?}"
export EGOTOUCH_ROOT="${EGOTOUCH_ROOT:-$SCRATCH/egotouch/data}"
export RECGEN_DIR="${RECGEN_DIR:-$WORK/ext/recgen}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
source launch/config.sh
echo "=== $KEY -> $NAME on $(hostname) ($SLURM_JOB_ID) ==="

SKIP_SCENE=1 SKIP_TRAIN=1 SKIP_EVAL=1 launch/rl_recon.sh "$NAME" "$KEY" 2>&1 | grep --line-buffered -v '^profiling'
ls -l "data/references/$NAME/reference.npz"
echo "=== done: $NAME ==="
