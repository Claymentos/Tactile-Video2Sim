#!/bin/bash
# One EgoTouch video -> episode clip ready for RL (one H100). Submitted by launch/egotouch_video.sh, which first
# downloads the episode on the login node and chains the training on this job's success.
#
#   1  launch/prepare_egotouch.sh EP (VLM=qwen)        extract, retarget, Qwen-VL scene (objects, hands used), cut
#   2  launch/reconstruct_objects.sh egotouch KEY       SAM3, MoGe, camera registration, RecGen, CoTracker, scene
#   3  build_episode_clip.py --objects guide            mocap hands, glove, the video trajectory as the guide and
#                                                       each hold's tactile grasp pose (the assist's target)
#   4  check_clip_contact.py                            kinematic replay: the simulated glove must read contacts
#                                                       (grasp F1 >= MIN_F1), else this job fails and no training
#
# Env: EP (recording time, e.g. 20260314_151150_129)  CLIP_DIR (data/episodes/<name>)  MIN_F1=0.1
#SBATCH -J et_video
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
: "${EP:?}" "${CLIP_DIR:?}"
source launch/config.sh
export TRANSFORMERS_OFFLINE=1 PYOPENGL_PLATFORM=egl
KEY=$("$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from tv2s_data.egotouch import select_episodes
e = select_episodes('$EGOTOUCH_ROOT', '$EP'); assert len(e) == 1, e; print(e[0].key)")
echo "=== $KEY -> $CLIP_DIR on $(hostname) ($SLURM_JOB_ID) ==="

echo "##### 1/4 prepare (Qwen-VL scene) #####"
VLM=qwen launch/prepare_egotouch.sh "$EP"
[ -e "data/egotouch/episodes/$KEY.npz" ] || { echo "extraction skipped $KEY (see above)"; exit 1; }
[ -e "data/egotouch/scene/$KEY.json" ] || { echo "no scene for $KEY (Qwen-VL parsing failed)"; exit 1; }
echo "##### 2/4 objects #####"
launch/reconstruct_objects.sh egotouch "$KEY" data/egotouch
echo "##### 3/4 episode clip #####"
"$PY_SIM" scripts/data/build_episode_clip.py --perception "data/egotouch/perception/$KEY" --hands mocap \
    --data data/egotouch --et-root "$EGOTOUCH_ROOT" --objects guide --out "$CLIP_DIR" 2>&1 | grep -v "^profiling"
echo "##### 4/4 contact check #####"
"$PY_SIM" scripts/data/check_clip_contact.py --clip-dir "$CLIP_DIR" --min-f1 "${MIN_F1:-0.1}" 2>&1 \
    | grep -v "^profiling\|Falling back to convex hull\|qpos0) exceeds"
echo "=== done: $CLIP_DIR ==="
