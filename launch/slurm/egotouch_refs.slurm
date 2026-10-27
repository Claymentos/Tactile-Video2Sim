#!/bin/bash
# One EgoTouch demonstration -> reconstruction -> the two rl_recon references of the object-trajectory comparison
# (one H100). Submitted by launch/submit_traj_compare.sh.
#
#   1  launch/reconstruct.sh KEY        SAM3, MoGe, camera, mocap, RecGen, CoTracker (the object and hand from the
#                                       dataset's manifest.jsonl, reference frame FRAME)
#   2  prepare_reference.py             data/references/NAME_kf: --object-trajectory keyframes (the object measured on
#                                       its reliable frames only, carried by the palm while the glove holds it)
#                                       data/references/NAME_tr: --object-trajectory tracked (CoTracker + Kalman on
#                                       every frame); the same retargeted hand (wuji_ref) for both
#
# Env: KEY (EgoTouch key)  NAME  OBJECT (SAM3 prompt)  SIDE (left|right)  FRAME=0
#SBATCH -J et_refs
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
: "${KEY:?}" "${NAME:?}" "${OBJECT:?}" "${SIDE:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
source launch/config.sh
export PY_RETARGET WUJI_RETARGETING_DIR
RECON=data/egotouch/recon/$KEY
echo "=== $KEY ('$OBJECT', $SIDE hand, frame ${FRAME:-0}) -> $NAME on $(hostname) ($SLURM_JOB_ID) ==="

if compgen -G "$RECON/*_traj_cotracker.npz" > /dev/null && [ -e "$RECON/mocap_joints_cotracker.npz" ]; then
    echo "##### 1/2 reconstruction: $RECON exists #####"
else
    echo "##### 1/2 reconstruction #####"
    recon() { OBJECT="$1" FRAME="${FRAME:-0}" HAND_SIDES="$SIDE" launch/reconstruct.sh "$KEY" "$RECON" 2>&1 \
        | grep --line-buffered -v '^profiling'; }
    if ! recon "$OBJECT"; then
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
        SKIP_FRAMES=1 recon "$DESC"
    fi
fi
echo "##### 2/2 references #####"
for V in kf tr; do
    OUT=data/references/${NAME}_$V
    mkdir -p "$OUT"
    [ "$V" = tr ] && cp -n "data/references/${NAME}_kf/wuji_ref_$SIDE.npz" "$OUT/"
    MODE=$([ "$V" = kf ] && echo keyframes || echo tracked)
    echo "--- $OUT: --object-trajectory $MODE"
    "$PY_SIM" scripts/rl/prepare_reference.py --recon "$RECON" --side "$SIDE" --object-trajectory "$MODE" --out "$OUT" 2>&1 \
        | grep --line-buffered -v '^profiling'
    ls -l "$OUT/reference.npz"
done
# a misplaced mocap hand (pick_up_hammer/095941: the left hand put on the right glove) sends the wrist far above the
# table, and the keyframes variant carries the object with it: no training on such a reference
"$PY_SIM" -c "
import numpy as np, sys
z = np.load('data/references/${NAME}_kf/reference.npz', allow_pickle=True)['wrist_pos'][:, 2]
print(f'wrist height above the table: median {z.mean():.2f} m, max {z.max():.2f} m (gate {sys.argv[1]} m)')
sys.exit(0 if z.max() <= float(sys.argv[1]) else 'implausible hand placement: see $RECON/hands_overlay_cotracker.mp4')" \
    "${MAX_WRIST_Z:-1.0}"
echo "=== done: data/references/${NAME}_{kf,tr} ==="
