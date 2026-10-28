#!/bin/bash
# One EgoTouch demonstration -> its rl_recon reference, the commands of README_egotouch.md's "Recette complète"
# (et_hammer_da3_contact_arm) steps 2-4, unchanged but for the output folders (one H100):
#
#   2  launch/prepare_egotouch.sh KEY       extraction and the Qwen3-VL-4B scene (data/egotouch/scene/KEY.json)
#   3  launch/reconstruct.sh KEY RECON      DEPTH=da3 HAND_CONTACT=1, both hands placed, object and frame from the scene
#   4  prepare_reference.py --scene         data/references/NAME: the hand from the scene's hands_used
#   5  check_reference.py                   data/references/NAME/quality.json (contact, slip, wrist)
#
# Env: KEY (EgoTouch key)  NAME  RECON=data/egotouch/recon/KEY_readme (the MoGe one, data/egotouch/recon/KEY, kept)
#      MAX_WRIST_Z=1.0  QUALITY=enforce|report (a rejected reference fails the job, or is only reported)
#      SAM3_POINT_START=1
# Submitted by launch/submit_readme.sh.
#SBATCH -J et_refs_readme
#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
: "${KEY:?}" "${NAME:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
source launch/config.sh
export PY_RETARGET WUJI_RETARGETING_DIR
RECON="${RECON:-data/egotouch/recon/${KEY}_readme}"
SCENE=data/egotouch/scene/$KEY.json
echo "=== $KEY -> $RECON, $NAME on $(hostname) ($SLURM_JOB_ID) ==="

echo "##### 2 extraction and Qwen-VL scene #####"
VLM=qwen MODEL=Qwen/Qwen3-VL-4B-Instruct GEMINI_FPS=4 OUT=data/egotouch launch/prepare_egotouch.sh "$KEY"
cat "$SCENE"

echo "##### 3 reconstruction (DA3 + contact) #####"
DEPTH=da3 DA3_CHECKPOINT=depth-anything/DA3METRIC-LARGE HAND_CONTACT=1 \
HANDS=mocap HAND_SIDES="left right" HAND_PLACEMENT=smooth HAND_POSE_SMOOTH=4 HAND_DEPTH=moge HAND_KNOT_FRAMES=10 \
    launch/reconstruct.sh "$KEY" "$RECON" 2>&1 | grep --line-buffered -v '^profiling'
cat "$RECON/hand_contact.json"

echo "##### 4 reference #####"
OUT=data/references/$NAME
"$PY_SIM" scripts/rl/prepare_reference.py --recon "$RECON" --scene "$SCENE" \
    --out "$OUT" --clearance-mode frame --hand-table-clearance 0.008 \
    --object-trajectory keyframes --anchor vision --object-target dense --overwrite 2>&1 | grep --line-buffered -v '^profiling'
ls -l "$OUT/reference.npz"
echo "##### 5 quality card (scripts/rl/check_reference.py) #####"
# QUALITY=enforce: a rejected reference fails the job, so its trainings never start; report: they start anyway
ENF=(); [ "${QUALITY:-enforce}" = enforce ] && ENF=(--enforce)
"$PY_SIM" scripts/rl/check_reference.py --recon "$RECON" --reference "$OUT" --scene "$SCENE" \
    --max-wrist-z "${MAX_WRIST_Z:-1.0}" "${ENF[@]}" 2>&1 | grep --line-buffered -v '^profiling'
echo "=== done: $OUT ==="
