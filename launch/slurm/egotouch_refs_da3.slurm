#!/bin/bash
# One EgoTouch demonstration -> the "DA3 + contact" reconstruction -> its rl_recon reference (one H100).
# Submitted by launch/submit_arm_compare.sh.
#
#   1  launch/reconstruct.sh KEY RECON   DEPTH=da3 HAND_CONTACT=1 (README_egotouch.md, "DA3 + contact"), the object and
#                                        hand from the dataset's manifest.jsonl, reference frame FRAME; the frames and
#                                        SAM3 masks copied from the MoGe reconstruction when there is one
#   2  prepare_reference.py              data/references/NAME: --clearance-mode frame
#
# Env: KEY (EgoTouch key)  NAME  OBJECT (SAM3 prompt)  SIDE (left|right)  FRAME=0  MAX_WRIST_Z=1.0
#      USE_SCENE=1: launch/rl_recon.sh's way instead: a Qwen-VL scene (data/egotouch/scene/KEY.json, made if missing)
#      gives the object, the reference frame and the hand (RECON data/egotouch/recon/KEY_da3s, nothing copied)
#SBATCH -J et_refs
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
: "${KEY:?}" "${NAME:?}" "${OBJECT:?}" "${SIDE:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
source launch/config.sh
export PY_RETARGET WUJI_RETARGETING_DIR
MOGE_RECON=data/egotouch/recon/$KEY
RECON=data/egotouch/recon/${KEY}_da3
SCENE=data/egotouch/scene/$KEY.json
if [ "${USE_SCENE:-0}" = 1 ]; then
    RECON=data/egotouch/recon/${KEY}_da3s
    if [ ! -e "$SCENE" ]; then
        echo "##### 0/2 Qwen-VL scene #####"
        IFS=$'\t' read -r VIDEO CONTEXT < <("$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from tv2s_data.egotouch import context, episode_from_key
ep = episode_from_key('$EGOTOUCH_ROOT', '$KEY'); print(ep.video, context(ep), sep='\t')")
        "$PY_QWEN" scripts/reconstruction/semantic_parsing.py --backend qwen --video "$VIDEO" --out "$SCENE" --fps 4 \
            --context "$CONTEXT"
    fi
    read -r OBJECT FRAME SIDE < <("$PY_SIM" -c "
import json, sys; sys.path.insert(0, 'src')
from tv2s_data.scene import manipulated_object
s = json.load(open('$SCENE')); o = manipulated_object(s)
print(o['id'].replace('_', ' ').replace(' ', '_'), int(round(float(o.get('visible_at_s') or 0) * 30)), s.get('hands_used'))")
    OBJECT="${OBJECT//_/ }"
    [ "$SIDE" = left ] || [ "$SIDE" = right ] || { echo "the scene's hands_used is '$SIDE': one hand only"; exit 1; }
fi
echo "=== $KEY ('$OBJECT', $SIDE hand, frame ${FRAME:-0}) -> $RECON, $NAME on $(hostname) ($SLURM_JOB_ID) ==="

if compgen -G "$RECON/*_traj_cotracker.npz" > /dev/null && [ -e "$RECON/hand_contact.json" ]; then
    echo "##### 1/2 reconstruction: $RECON exists #####"
else
    echo "##### 1/2 reconstruction (DA3 + contact) #####"
    SKIP=()
    MASK="$MOGE_RECON/video_segmentation/masks/frame_$(printf %06d "${FRAME:-0}")_masks/${OBJECT// /_}.png"
    if [ "${USE_SCENE:-0}" != 1 ] && [ -e "$MASK" ] && [ -d "$MOGE_RECON/all_frames" ] && [ ! -d "$RECON/all_frames" ]; then
        echo "  frames and SAM3 masks from $MOGE_RECON (they do not depend on the depth)"
        mkdir -p "$RECON"
        cp -r "$MOGE_RECON/all_frames" "$MOGE_RECON/video_segmentation" "$RECON/"
        cp "$MOGE_RECON"/*_rgb.mp4 "$RECON/"
    fi
    [ -d "$RECON/video_segmentation" ] && SKIP=(SKIP_FRAMES=1 SKIP_SAM3=1)
    recon() { env "${SKIP[@]}" DEPTH=da3 HAND_CONTACT=1 CONTACT_ARGS="--sides $SIDE" OBJECT="$1" FRAME="${FRAME:-0}" \
        HAND_SIDES="$SIDE" launch/reconstruct.sh "$KEY" "$RECON" 2>&1 | grep --line-buffered -v '^profiling'; }
    if ! recon "$OBJECT" && [ "${USE_SCENE:-0}" = 1 ]; then
        exit 1
    elif [ ! -e "$RECON/mocap_joints_cotracker.npz" ]; then
        OBJ_MASK="$RECON/video_segmentation/masks/frame_$(printf %06d "${FRAME:-0}")_masks/${OBJECT// /_}.png"
        "$PY_SIM" -c "import cv2,sys; m=cv2.imread(sys.argv[1],0); sys.exit(1 if m is None or (m>128).sum()<=100 else 0)" \
            "$OBJ_MASK" && exit 1                       # not the object prompt: a real failure
        # SAM3 misses the manifest's terse name (a toy saw as 'saw'): Qwen-VL's visual description of the object
        SCENE=data/egotouch/scene/$KEY.json
        IFS=$'\t' read -r VIDEO CONTEXT < <("$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from tv2s_data.egotouch import context, episode_from_key
ep = episode_from_key('$EGOTOUCH_ROOT', '$KEY'); print(ep.video, context(ep), sep='\t')")
        [ -e "$SCENE" ] || "$PY_QWEN" scripts/reconstruction/semantic_parsing.py --backend qwen --video "$VIDEO" \
            --out "$SCENE" --fps 4 --context "$CONTEXT The manipulated object: $OBJECT."
        DESC=$("$PY_SIM" -c "
import json, sys; sys.path.insert(0, 'src')
from tv2s_data.scene import manipulated_object
print(manipulated_object(json.load(open('$SCENE')))['description'].strip().rstrip('.'))")
        echo "  no '$OBJECT' at frame ${FRAME:-0}: SAM3 again with Qwen-VL's description '$DESC'"
        SKIP=(SKIP_FRAMES=1)
        recon "$DESC"
    fi
fi
echo "##### 2/2 reference #####"
OUT=data/references/$NAME
"$PY_SIM" scripts/rl/prepare_reference.py --recon "$RECON" --side "$SIDE" --out "$OUT" --clearance-mode frame --overwrite 2>&1 \
    | grep --line-buffered -v '^profiling'
ls -l "$OUT/reference.npz"
# a misplaced mocap hand (pick_up_hammer/095941 with MoGe: the left hand put on the right glove) sends the wrist far
# above the table: no training on such a reference
"$PY_SIM" -c "
import numpy as np, sys
z = np.load('$OUT/reference.npz', allow_pickle=True)['wrist_pos'][:, 2]
print(f'wrist height above the table: median {np.median(z):.2f} m, max {z.max():.2f} m (gate {sys.argv[1]} m)')
sys.exit(0 if z.max() <= float(sys.argv[1]) else 'implausible hand placement: see $RECON/hands_overlay_cotracker.mp4')" \
    "${MAX_WRIST_Z:-1.0}"
echo "=== done: $OUT ==="
