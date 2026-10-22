#!/usr/bin/env bash
# One EgoTouch episode reconstructed in its head camera: the mocap hands, the object's mesh and trajectory, the
# table (Benchmark-Genesis' run_touchanything_pipeline.sh). Everything is expressed in the camera of a reference frame.
# DATASET=ace: an ACE-Data-0 Tactile take (scripts/data/extract_ace.py's episode folder: the egocentric video and the
# gloves, no hand pose), its hands from the video (HANDS=hawor, step 5 below).
#
#   1  frames      every frame of chest.mp4                                                         [PY_SAM3: ffmpeg]
#   2  sam3        video masks of the object (text prompt) and of the table; when the text misses the table at the
#                  reference frame, points picked on the table found later (table_points.py)      [PY_SAM3]
#   3  depth       intrinsics from MoGe on the reference frame, MoGe metric depth of every frame     [PY_MOGE]
#   4  camera      the camera's motion: CoTracker points on the table, lifted with the depth, Kabsch [PY_SAM3, PY_SIM]
#   5  mocap       the Rokoko hands placed in the reference camera by one similarity fitted on the episode's WiLoR
#                  2D joints (those lying on the other hand's glove dropped: masks.npz), the gloves' masks (where
#                  WiLoR misses the hand) and the depth under it, by a placement varying smoothly over time (the mocap
#                  gives the hand's shape and fingers, the video where it is);
#                  the depth's scale from the hands' measured size; when it is off 1, the depth and the camera track
#                  are rescaled into the mocap's metres and the hands placed again; a check video
#                  (check_hands_overlay.py)                                                           [PY_SIM]
#      (HANDS=hawor) SAM3 hand instances (step 2, HAND_PROMPT) as HaWoR's detections, HaWoR's MANO in each frame's
#                  camera, scaled to the depth, carried into the reference camera by the camera motion; the same
#                  rescale of the depth into the hands' metres                         [PY_SAM3, PY_HAWOR, PY_SIM]
#   6  table       the table plane from the reference depth                                           [PY_SIM]
#   7  mesh        RecGen from the reference frame's RGB, depth and mask, without mask erosion       [PY_RECGEN]
#   8  object      CoTracker points on the object, lifted, Kabsch, composed with the camera motion into the
#                  reference camera, Kalman-smoothed                                                  [PY_SAM3, PY_SIM]
# The object (its SAM3 prompt) and the reference frame come from the scene file (semantic_parsing.py): the object
# the hands manipulate, when it is first visible. Then: scripts/rl/prepare_reference.py --recon <out> (or
# launch/rl_recon.sh, which runs this first), scripts/reconstruction/replay_raw_genesis.py --reference.
#
# Usage:   launch/reconstruct.sh KEY [OUT]          e.g. pick_up_screwdriver__20260321_103227_213
#          DATASET=ace OBJECT="orange box" FRAME=140 launch/reconstruct.sh t4_000078
#          OUT: default data/<dataset>/recon/<key>
# Writes:  all_frames/, video_segmentation/ (masks, tracked_*.mp4), intrinsics.yaml, depth_moge/ (depth_moge_raw/),
#          camera_traj_cotracker.npz, mocap_joints_cotracker.npz, mocap_to_camera_cotracker.json,
#          hands_overlay_cotracker.mp4, hands_wilor_check.mp4 (check the mocap placement), ground_plane_moge.json,
#          recgen_out_moge/<object>/posed_mesh.obj, <object>_points_cotracker.mp4, <object>_traj_cotracker.npz;
#          source -> the episode
# Knobs:   SCENE=file.json (default data/egotouch/scene/<key>.json)  OBJECT="text prompt"  FRAME=n (reference frame)
#          TABLE_POINTS="x,y;x,y" (table prompt, pixels of the reference frame)  STATIC_PROMPT=table
#          HAND_SIDES=left (fit the mocap on these hands only: a frozen Vive tracker on the other wrecks the scale)
#          HAND_FIT_FRAMES=0-300  HAND_FIXED_SCALE=1.0  HAND_DRIFT_KNOTS=30  SCALE_TOLERANCE=0.02
#          HAND_SCALE_FROM=hands|fit (the depth's scale from the hands' measured size, default; or fitted from the
#          depth cue, Benchmark-Genesis')
#          HAND_PLACEMENT=smooth|per-frame|global (smooth, default: the placement varying smoothly over time, fitted on
#          WiLoR's joints, the gloves' masks and the depth of every frame together; per-frame: the measured hand placed on
#          WiLoR in every frame and smoothed; global: one similarity for the whole clip, Benchmark-Genesis')
#          HAND_KNOT_FRAMES=10 (smooth)  HAND_POSE_SMOOTH=4 (frames)  HAND_DEPTH=moge|pnp
#          HANDS=mocap|hawor (default: mocap on EgoTouch, hawor on ACE)  HAND_PROMPT="black glove" (SAM3, HANDS=hawor)
#          HAND_PER_FRAME_DEPTH=0 (keep HaWoR's own per-frame depth)
#          RECGEN_EROSION=1  SAM3_OFFLOAD=1  SMOOTH_WINDOW=9
#          SKIP_FRAMES=1 SKIP_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_MOCAP=1 SKIP_MESH=1 SKIP_OBJECT=1
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
KEY="${1:?usage: launch/reconstruct.sh KEY [OUT]}"
DATASET="${DATASET:-egotouch}"
case "$DATASET" in
    egotouch) HANDS="${HANDS:-mocap}" ;;
    ace) HANDS="${HANDS:-hawor}" ;;
    *) echo "DATASET=egotouch|ace" >&2; exit 1 ;;
esac
OUT="$(realpath -m "${2:-$REPO/data/$DATASET/recon/$KEY}")"
SCENE="${SCENE:-$REPO/data/$DATASET/scene/$KEY.json}"
R=scripts/reconstruction
# each stage in its env's interpreter, with that env's tools (ffmpeg) on the PATH
in_env() { local py="$1"; shift; PATH="$(dirname "$py"):$PATH" "$py" "$@"; }

# the episode folder, the object and the reference frame
# '|'-separated: a tab is IFS whitespace, and an empty field (no scene file: no object) would vanish
IFS='|' read -r EP_DIR STAMP TASK FPS SCENE_OBJ SCENE_FRAME < <("$PY_SIM" -c "
import json, sys; sys.path.insert(0, 'src')
from pathlib import Path
from tv2s_data.scene import manipulated_object
if '$DATASET' == 'ace':
    ep_dir = Path('$ACE_EPISODES') / '$KEY'
    if not (ep_dir / 'take.json').exists():
        sys.exit(f'no {ep_dir}/take.json: scripts/data/extract_ace.py --takes $KEY')
    stamp, task, fps = '$KEY', '$KEY', json.loads((ep_dir / 'take.json').read_text())['fps']
else:
    from tv2s_data.egotouch import FPS, episode_from_key
    ep = episode_from_key('$EGOTOUCH_ROOT', '$KEY')
    ep_dir, stamp, task, fps = ep.dir, ep.stamp, ep.task, FPS
obj, frame = '', 0
if Path('$SCENE').exists():
    o = manipulated_object(json.load(open('$SCENE')))
    obj, frame = o['id'].replace('_', ' '), int(round(float(o.get('visible_at_s') or 0) * fps))
print(ep_dir, stamp, task, fps, obj, frame, sep='|')")
OBJECT="${OBJECT:-$SCENE_OBJ}"
n="${FRAME:-$SCENE_FRAME}"
[ -n "$OBJECT" ] || { echo "no scene file $SCENE: OBJECT=\"<object>\" (or run semantic_parsing.py)"; exit 1; }
OBJ_ID="${OBJECT// /_}"
STATIC_PROMPT="${STATIC_PROMPT:-table}"
STATIC_ID="${STATIC_PROMPT// /_}"
if [ "$HANDS" = "mocap" ]; then
    if [ ! -e "$EP_DIR/wilor_hands.json" ]; then
        echo "=== downloading wilor_hands.json (the 2D hand joints the mocap is placed with) ==="
        "$PY_SIM" scripts/data/download_egotouch.py --root "$EGOTOUCH_ROOT" --tasks "$TASK" --episodes "$STAMP" --files wilor_hands.json
    fi
    for f in rokoko_hands.json wilor_hands.json; do
        [ -f "$EP_DIR/$f" ] || { echo "no $f in $EP_DIR (HANDS=hawor: from the video)" >&2; exit 1; }
    done
fi
[ -f "$EP_DIR/chest.mp4" ] || { echo "no chest.mp4 in $EP_DIR" >&2; exit 1; }
HAND_PROMPT="${HAND_PROMPT:-$([ "$DATASET" = ace ] && echo "black glove" || echo hand)}"
HAND_ID=hands
mkdir -p "$OUT"
ln -sfn "$EP_DIR" "$OUT/source"
echo "{\"fps\": $FPS, \"dataset\": \"$DATASET\", \"hands\": \"$HANDS\"}" > "$OUT/video.json"
echo "=== $KEY ($DATASET, $FPS fps, hands: $HANDS): object '$OBJECT', reference frame $n -> $OUT ==="

VIDEO_PATH="$OUT/${TASK}_rgb.mp4"
FRAMES_DIR="$OUT/all_frames"
FRAME_PATH="$FRAMES_DIR/$(printf "%06d.png" "$n")"
MASKS_ROOT="$OUT/video_segmentation/masks"
REF_MASKS="$MASKS_ROOT/frame_$(printf "%06d" "$n")_masks"
DEPTH_DIR="$OUT/depth_moge"
INTRINSICS_YAML="$OUT/intrinsics.yaml"
GROUND_PLANE="$OUT/ground_plane_moge.json"
MESH="$OUT/recgen_out_moge/$OBJ_ID/posed_mesh.obj"
TABLE_POINTS_NPZ="$OUT/${STATIC_ID}_points_cotracker.npz"
CAMERA_TRAJ="$OUT/camera_traj_cotracker.npz"
MOCAP_JOINTS="$OUT/mocap_joints_cotracker.npz"
MOCAP_FIT="$OUT/mocap_to_camera_cotracker.json"
OBJ_POINTS="$OUT/${OBJ_ID}_points_cotracker.npz"
OBJ_TRAJ_CAM="$OUT/${OBJ_ID}_traj_camera_cotracker.npz"
OBJ_TRAJ_COMPOSED="$OUT/${OBJ_ID}_traj_composed_cotracker.npz"
OBJ_TRAJ="$OUT/${OBJ_ID}_traj_cotracker.npz"
MOGE_REPO_ARG=(); [ -d "${MOGE_REPO:-}" ] && MOGE_REPO_ARG=(--moge-repo "$MOGE_REPO")
OFFLOAD=(); [ "${SAM3_OFFLOAD:-0}" = "1" ] && OFFLOAD=(--offload-video)
mask_ok() { "$PY_SIM" -c "import cv2,sys; m=cv2.imread(sys.argv[1],0); sys.exit(0 if m is not None and (m>128).sum()>100 else 1)" "$1"; }

if [ "${SKIP_FRAMES:-0}" != "1" ]; then
    echo "=== 1/8 frames ==="
    cp "$EP_DIR/chest.mp4" "$VIDEO_PATH"
    rm -rf "$FRAMES_DIR" && mkdir -p "$FRAMES_DIR"
    PATH="$(dirname "$PY_SAM3"):$PATH" ffmpeg -v error -i "$VIDEO_PATH" -vsync 0 -start_number 0 "$FRAMES_DIR/%06d.png"
fi
if [ "${SKIP_SAM3:-0}" != "1" ]; then
    echo "=== 2/8 SAM3: object '$OBJECT', static '$STATIC_PROMPT' ==="
    in_env "$PY_SAM3" $R/segment_video_sam3.py "${OFFLOAD[@]}" --video "$VIDEO_PATH" --text "$OBJECT" --obj_id "$OBJ_ID" --frame_idx "$n"
    table_points() {   # TABLE_POINTS
        local labels; labels="$(tr ';' '\n' <<< "$1" | sed 's/.*/1/' | paste -sd';')"
        in_env "$PY_SAM3" $R/segment_video_sam3.py "${OFFLOAD[@]}" --video "$VIDEO_PATH" --points "$1" --point_labels "$labels" \
            --obj_id "$STATIC_ID" --frame_idx "$n"
    }
    if [ -n "${TABLE_POINTS:-}" ]; then
        table_points "$TABLE_POINTS"
    else
        in_env "$PY_SAM3" $R/segment_video_sam3.py "${OFFLOAD[@]}" --video "$VIDEO_PATH" --text "$STATIC_PROMPT" --obj_id "$STATIC_ID" --frame_idx "$n"
        if ! mask_ok "$REF_MASKS/$STATIC_ID.png"; then
            # the text prompt can find the table only later in the video; it seeds the camera tracking at frame n
            TABLE_POINTS="$("$PY_SIM" $R/table_points.py --recon "$OUT" --frame "$n" --object "$OBJ_ID" --static "$STATIC_ID")"
            [ -n "$TABLE_POINTS" ] || { echo "no $STATIC_PROMPT found in the video: TABLE_POINTS=\"x,y;x,y\" (pixels of frame $n)"; exit 1; }
            echo "  the $STATIC_PROMPT missed at frame $n: SAM3 again, prompted with points TABLE_POINTS=\"$TABLE_POINTS\""
            table_points "$TABLE_POINTS"
        fi
    fi
    if [ "$HANDS" = "hawor" ]; then
        echo "  the hands for HaWoR: SAM3 '$HAND_PROMPT', every instance"
        in_env "$PY_SAM3" $R/segment_video_sam3.py "${OFFLOAD[@]}" --video "$VIDEO_PATH" --text "$HAND_PROMPT" \
            --obj_id "$HAND_ID" --frame_idx "$n" --instances
    fi
fi
for mask in "$REF_MASKS/$OBJ_ID.png" "$REF_MASKS/$STATIC_ID.png"; do
    mask_ok "$mask" || { echo "empty mask at the reference frame: $mask (FRAME=, OBJECT=, TABLE_POINTS=)" >&2; exit 1; }
done

echo "=== 3/8 intrinsics and depth (MoGe) ==="
INTRINSICS_INFO="$(in_env "$PY_MOGE" $R/estimate_intrinsics_moge.py --image "$FRAME_PATH" --checkpoint "$MOGE_CHECKPOINT" \
    "${MOGE_REPO_ARG[@]}" --out "$INTRINSICS_YAML")"
echo "$INTRINSICS_INFO"
FOV_X="$(sed -n 's/^FOV_X=//p' <<< "$INTRINSICS_INFO")"
if [ "${SKIP_DEPTH:-0}" != "1" ]; then
    rm -rf "$DEPTH_DIR" "${DEPTH_DIR}_raw"          # a fresh depth pass invalidates any earlier rescale
    in_env "$PY_MOGE" $R/estimate_depth_moge.py --frames-dir "$FRAMES_DIR" --out-dir "$DEPTH_DIR" --fov-x "$FOV_X" \
        --checkpoint "$MOGE_CHECKPOINT" "${MOGE_REPO_ARG[@]}"
fi

track_rigid() {  # MASK_NAME POINTS_OUT POSES_OUT SMOOTHER
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in_env "$PY_SAM3" $R/track_points_cotracker.py \
        --frames-dir "$FRAMES_DIR" --depth-dir "$DEPTH_DIR" --masks-root "$MASKS_ROOT" --object "$1" --ref-frame "$n" \
        --intrinsics "$INTRINSICS_YAML" --fps "$FPS" --out "$2" --overlay "${2%.npz}.mp4"
    "$PY_SIM" $R/poses_from_points.py --points "$2" --smoother "$4" --smooth-window "${SMOOTH_WINDOW:-9}" --out "$3"
}
if [ "${SKIP_CAMERA:-0}" != "1" ]; then
    echo "=== 4/8 camera motion from the $STATIC_PROMPT (CoTracker + Kabsch) ==="
    track_rigid "$STATIC_ID" "$TABLE_POINTS_NPZ" "$CAMERA_TRAJ" window
fi

place_hands() {  # [--fixed-scale S]
    if [ "$HANDS" = "hawor" ]; then
        local per_frame=(); [ "${HAND_PER_FRAME_DEPTH:-1}" = "0" ] && per_frame=(--no-per-frame-depth)
        "$PY_SIM" $R/place_video_hands.py --hands "$OUT/hawor/hawor_hands.npz" --camera-traj "$CAMERA_TRAJ" \
            --intrinsics "$INTRINSICS_YAML" --depth-dir "$DEPTH_DIR" --frames-dir "$FRAMES_DIR" --fps "$FPS" \
            --overlay "$OUT/hands_overlay_cotracker.mp4" --transform-out "$MOCAP_FIT" --out "$MOCAP_JOINTS" \
            --sides ${HAND_SIDES:-left right} "${per_frame[@]}" "$@"
        return
    fi
    "$PY_SIM" $R/place_mocap_hands.py --episode-dir "$EP_DIR" --camera-traj "$CAMERA_TRAJ" --intrinsics "$INTRINSICS_YAML" \
        --depth-dir "$DEPTH_DIR" --frames-dir "$FRAMES_DIR" --overlay "$OUT/hands_overlay_cotracker.mp4" \
        --transform-out "$MOCAP_FIT" --out "$MOCAP_JOINTS" --sides ${HAND_SIDES:-left right} --scale-from "${HAND_SCALE_FROM:-hands}" \
        --placement "${HAND_PLACEMENT:-smooth}" --pose-smooth "${HAND_POSE_SMOOTH:-4}" --hand-depth "${HAND_DEPTH:-moge}" \
        --knot-frames "${HAND_KNOT_FRAMES:-10}" \
        ${HAND_FIT_FRAMES:+--fit-frames "$HAND_FIT_FRAMES"} ${HAND_DRIFT_KNOTS:+--drift-knots "$HAND_DRIFT_KNOTS"} "$@"
}
if [ "${SKIP_MOCAP:-0}" != "1" ]; then
    echo "=== 5/8 $HANDS hands (${HAND_SIDES:-left right}) into the reference camera ==="
    if [ "$HANDS" = "hawor" ]; then
        in_env "$PY_HAWOR" $R/estimate_hands_hawor.py --frames-dir "$FRAMES_DIR" --intrinsics "$INTRINSICS_YAML" \
            --work "$OUT/hawor" --instances "$MASKS_ROOT/instances_$HAND_ID.npz"
    fi
    place_hands ${HAND_FIXED_SCALE:+--fixed-scale "$HAND_FIXED_SCALE"}
    SCALE="$("$PY_SIM" -c "import json,sys; print(json.load(open(sys.argv[1]))['scale'])" "$MOCAP_FIT")"
    if "$PY_SIM" -c "import sys; sys.exit(0 if abs(float(sys.argv[1]) - 1) > float(sys.argv[2]) else 1)" "$SCALE" "${SCALE_TOLERANCE:-0.02}"; then
        echo "  the depth is off the hands' metric by s=$SCALE: depth and camera track rescaled, hands refitted"
        "$PY_SIM" $R/scale_depth.py --depth-dir "$DEPTH_DIR" --scale "$SCALE" --points "$TABLE_POINTS_NPZ" --poses "$CAMERA_TRAJ"
        place_hands --fixed-scale 1.0
    fi
    if [ "$HANDS" = "mocap" ]; then
        "$PY_SIM" $R/check_hands_overlay.py --recon "$OUT"
    fi
fi

echo "=== 6/8 table plane ==="
"$PY_SIM" $R/fit_table_plane.py --depth "$DEPTH_DIR/$(printf "%06d.png" "$n")" --intrinsics "$INTRINSICS_YAML" \
    --exclude "$REF_MASKS/$OBJ_ID.png" --out "$GROUND_PLANE"

if [ "${SKIP_MESH:-0}" != "1" ]; then
    echo "=== 7/8 object mesh (RecGen): $OBJECT ==="
    EROSION=(--no-mask-erosion); [ "${RECGEN_EROSION:-0}" = "1" ] && EROSION=()
    in_env "$PY_RECGEN" $R/reconstruct_mesh_recgen.py "${EROSION[@]}" --rgb "$FRAME_PATH" --depth "$DEPTH_DIR/$(printf "%06d.png" "$n")" \
        --mask "$REF_MASKS/$OBJ_ID.png" --intrinsics "$INTRINSICS_YAML" --checkpoint "$RECGEN_CHECKPOINT" \
        --out "$OUT/recgen_out_moge/$OBJ_ID" --name "$OBJ_ID"
fi

if [ "${SKIP_OBJECT:-0}" != "1" ]; then
    echo "=== 8/8 object trajectory: CoTracker + depth + Kabsch, camera motion composed out, Kalman ==="
    track_rigid "$OBJ_ID" "$OBJ_POINTS" "$OBJ_TRAJ_CAM" none
    "$PY_SIM" $R/compose_camera_motion.py --object-traj "$OBJ_TRAJ_CAM" --camera-traj "$CAMERA_TRAJ" \
        --intrinsics "$INTRINSICS_YAML" --out "$OBJ_TRAJ_COMPOSED"
    "$PY_SIM" $R/smooth_object_kalman.py --traj "$OBJ_TRAJ_COMPOSED" --mesh "$MESH" --out "$OBJ_TRAJ"
fi
echo "reconstruction: $OUT   check: hands_wilor_check.mp4, video_segmentation/tracked_*.mp4, ${OBJ_ID}_points_cotracker.mp4"
