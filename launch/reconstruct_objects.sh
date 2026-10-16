#!/usr/bin/env bash
# Objects of an episode, from its video: SAM3 masks, depth, RecGen meshes (after semantic_parsing.py).
#
#   1  frames    sampled frames (FPS), intrinsics, undistortion; TouchScale: its sensor depth;
#                EgoTouch: its glove masks as sam3/hands.npz                                       [PY_SIM]
#   2  sam3      a video mask of every object of scene/<key>.json, prompted with its description,
#                and of both hands                                                                [PY_SAM3]
#   3  depth     MoGe metric depth of every frame (HumanTouch, EgoTouch; TouchScale keeps its sensor depth);
#                EgoTouch first fits its camera and focal length on the hands (no intrinsics)     [PY_MOGE]
#   4  register  HumanTouch, EgoTouch: camera in the mocap world (hands in their masks, table
#                horizontal), depth scale, table plane                                           [PY_SIM]
#   5  recgen    a mesh of every object (not the table), from its best frame near visible_at_s      [PY_RECGEN]
#   6  track     FoundationPose trajectory of every object, from its RecGen pose (and its model)    [PY_FP]
#                then CoTracker + depth (Benchmark-Genesis): the trajectory the scene uses          [PY_SIM]
#   7  world     TouchScale: camera motion from the objects no hand moves, the table as the world   [PY_SIM]
#   8  scene     the simulation scene (table at z = 0, meshes, masses, trajectories), Genesis replay  [PY_SIM]
# -> DATA/perception/<key>/{frames,depth,sam3/,camera.npz,recgen/<obj>/,objects.json,foundationpose/,
#                          sim_scene.json,sim_trajectories.npz,replay.mp4}
#
# Usage:   launch/reconstruct_objects.sh humantouch EPISODE [DATA]      e.g. humantouch 2 data/humantouch
#          launch/reconstruct_objects.sh touchscale ID_PREFIX [DATA]    e.g. touchscale 5a1ae97d
#          launch/reconstruct_objects.sh egotouch KEY [DATA]            e.g. egotouch pick_up_bottle__20260314_000533_921
# Knobs:   FPS=10 (EgoTouch 15)  HEIGHT=720 (EgoTouch 480)  OBJECTS=id,id  DEPTH=moge (TouchScale: MoGe instead of the sensor)
#          NO_EROSION=1 (RecGen keeps thin parts)  POINT="id=x,y;x,y@frame" (click prompt)  PROMPT="id=text"
#          SKIP_FRAMES=1 SKIP_SAM3=1 SKIP_DEPTH=1 SKIP_REGISTER=1 SKIP_RECGEN=1 SKIP_TRACK=1 SKIP_SCENE=1
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
DATASET="${1:?usage: launch/reconstruct_objects.sh humantouch|touchscale|egotouch EPISODE [DATA]}"
EP="${2:?usage: launch/reconstruct_objects.sh humantouch|touchscale|egotouch EPISODE [DATA]}"
case "$DATASET" in
    humantouch) ROOT="$HUMANTOUCH_ROOT"; DATA="${3:-$DATA_DIR}" ;;
    touchscale) ROOT="$TOUCHSCALE_ROOT"; DATA="${3:-$REPO/data/touchscale}" ;;
    egotouch)   ROOT="$EGOTOUCH_ROOT"; DATA="${3:-$REPO/data/egotouch}"; FPS="${FPS:-15}"; HEIGHT="${HEIGHT:-480}" ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac
OBJ=(); [ -n "${OBJECTS:-}" ] && OBJ=(--objects "$OBJECTS")
QUIET='Warning\|warn(\|^profiling\| INFO \|it/s]'

echo "=== 1/8 frames ==="
if [ "${SKIP_FRAMES:-0}" != "1" ]; then
    P=$("$PY_SIM" scripts/reconstruction/prepare_frames.py --dataset "$DATASET" --data "$DATA" --root "$ROOT" \
        --episode "$EP" --fps "${FPS:-10}" --height "${HEIGHT:-720}" | tee /dev/stderr | tail -1)
else
    P=$("$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
if '$DATASET' == 'humantouch': print('$DATA/perception/episode_%06d' % int('$EP'))
elif '$DATASET' == 'egotouch': print('$DATA/perception/$EP')
else:
    from tv2s_data.touchscale import select_episodes; print('$DATA/perception/' + select_episodes('$ROOT', '$EP')[0].id)")
fi
echo "perception folder: $P"
if [ "${SKIP_SAM3:-0}" != "1" ]; then
    echo "=== 2/8 SAM3 masks ==="
    EXTRA=(); [ -n "${POINT:-}" ] && EXTRA+=(--point "$POINT"); [ -n "${PROMPT:-}" ] && EXTRA+=(--prompt "$PROMPT")
    "$PY_SAM3" scripts/reconstruction/segment_objects_sam3.py --perception "$P" "${OBJ[@]}" "${EXTRA[@]}" 2>&1 | grep --line-buffered -v "$QUIET"
    if [ "$DATASET" = "egotouch" ] && [ -e "$P/sam3/hands.npz" ]; then
        echo "  hands: the dataset's glove masks"
    else
        "$PY_SAM3" scripts/reconstruction/segment_objects_sam3.py --perception "$P" --hands 2>&1 | grep --line-buffered -v "$QUIET"
    fi
fi
if [ "${SKIP_DEPTH:-0}" != "1" ]; then
    echo "=== 3/8 depth ==="
    if [ "$DATASET" = "egotouch" ]; then
        echo "  camera and focal length from the hands (the dataset has no intrinsics)"
        rm -f "$P/camera.npz"
        "$PY_SIM" scripts/reconstruction/register_camera.py --perception "$P" --dataset egotouch --data "$DATA" --hands-only \
            2>&1 | grep --line-buffered -v "$QUIET"
    fi
    if [ "$DATASET" != "touchscale" ] || [ "${DEPTH:-}" = "moge" ]; then
        FORCE=(); [ "$DATASET" = "touchscale" ] && FORCE=(--force)
        "$PY_MOGE" scripts/reconstruction/estimate_depth_moge.py --perception "$P" "${FORCE[@]}" 2>&1 | grep --line-buffered -v "$QUIET"
    else
        echo "  sensor depth (DEPTH=moge to estimate it instead)"
    fi
fi
if [ "${SKIP_REGISTER:-0}" != "1" ] && [ "$DATASET" != "touchscale" ]; then
    echo "=== 4/8 camera registration ==="
    "$PY_SIM" scripts/reconstruction/register_camera.py --perception "$P" --dataset "$DATASET" --data "$DATA" \
        --root "$ROOT" 2>&1 | grep --line-buffered -v "$QUIET"
fi
if [ "${SKIP_RECGEN:-0}" != "1" ]; then
    echo "=== 5/8 RecGen meshes ==="
    ERO=(); [ "${NO_EROSION:-0}" = "1" ] && ERO=(--no-mask-erosion)
    "$PY_RECGEN" scripts/reconstruction/reconstruct_objects_recgen.py --perception "$P" "${OBJ[@]}" "${ERO[@]}" \
        2>&1 | grep --line-buffered -v "$QUIET"
fi
if [ "${SKIP_TRACK:-0}" != "1" ]; then
    echo "=== 6/8 FoundationPose trajectories ==="
    "$PY_FP" scripts/reconstruction/track_objects_foundationpose.py --perception "$P" "${OBJ[@]}" 2>&1 \
        | grep --line-buffered "agreement\|tracked\|Error\|Traceback"
    # the scene's trajectories (assemble_scene.py --trajectories cotracker, the default): Benchmark-Genesis'
    # CoTracker + depth, seeded on the RecGen frame (needs the models written by FoundationPose above)
    echo "=== 6/8 CoTracker trajectories ==="
    "$PY_SIM" scripts/reconstruction/track_objects_cotracker.py --perception "$P" "${OBJ[@]}" 2>&1 | grep --line-buffered -v "$QUIET"
fi
if [ "${SKIP_SCENE:-0}" != "1" ]; then
    if [ "$DATASET" = "touchscale" ]; then
        echo "=== 7/8 camera motion and world frame ==="
        "$PY_SIM" scripts/reconstruction/estimate_camera_motion.py --perception "$P" 2>&1 | grep -v "$QUIET"
    fi
    echo "=== 8/8 simulation scene ==="
    "$PY_SIM" scripts/reconstruction/assemble_scene.py --perception "$P" 2>&1 | grep -v "$QUIET"
    "$PY_SIM" scripts/reconstruction/replay_scene_genesis.py --perception "$P" 2>&1 | grep -v "$QUIET\|libEGL\|pci id"
fi
echo "masks: $P/sam3/overlay.mp4   registration: $P/register_overlay.mp4   meshes: $P/recgen/   summary: $P/objects.json   replay: $P/replay.mp4"
