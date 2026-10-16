#!/bin/bash
# One EgoTouch demonstration -> simulation scene -> rl_recon reference (one H100). Submitted by launch/submit_egotouch.sh.
#
#   1  launch/reconstruct_objects.sh egotouch KEY   SAM3 masks, MoGe depth, camera registration, RecGen meshes,
#                                                   CoTracker + depth trajectories, Genesis scene + replay
#   2  build_episode_clip.py                        the episode clip (mocap hands, glove, objects)
#   3  prepare_reference.py --clip --scene          reference.npz for the hand Gemini says is used
#
# Env: KEY (EgoTouch key, e.g. pick_up_hammer__20260321_085301_950)  NAME (data/episodes/NAME, data/references/NAME)
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
export BG_SCRIPTS="${BG_SCRIPTS:-$WORK/Benchmark-Genesis/reconstruction/scripts}"
export RECGEN_DIR="${RECGEN_DIR:-$WORK/ext/recgen}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
source launch/config.sh
export PY_SAM3
echo "=== $KEY -> $NAME on $(hostname) ($SLURM_JOB_ID) ==="

launch/reconstruct_objects.sh egotouch "$KEY" data/egotouch
P=data/egotouch/perception/$KEY
"$PY_SIM" scripts/data/build_episode_clip.py --perception "$P" --hands mocap --data data/egotouch \
    --et-root "$EGOTOUCH_ROOT" --out "data/episodes/$NAME" 2>&1 | grep -v "^profiling"
CLIP=$(ls "data/episodes/$NAME"/*.pkl | head -1)
"$PY_SIM" scripts/rl/prepare_reference.py --clip "$CLIP" --scene "data/egotouch/scene/$KEY.json" \
    --out "data/references/$NAME" 2>&1 | grep -v "^profiling"
ls -l "data/references/$NAME/reference.npz"
echo "=== done: $NAME ==="
