#!/bin/bash
# An existing reconstruction (frames, SAM3 masks, depth, camera and hands kept) with its object mesh made again by
# SAM 3D Objects (reconstruct.sh MESH_MODEL=sam3d), then the object trajectory, the contact correction, the reference
# and its quality card (one H100). For comparing the meshes on the same reconstruction.
# Env: KEY  SRC (the reconstruction, copied)  RECON (the copy)  NAME (data/references/NAME)
#SBATCH -J et_refs_sam3d
#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
cd /lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
: "${KEY:?}" "${SRC:?}" "${RECON:?}" "${NAME:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
source launch/config.sh
export PY_RETARGET WUJI_RETARGETING_DIR SAM3D_DIR
SCENE=data/egotouch/scene/$KEY.json
mkdir -p "$RECON"
OBJ=$(ls "$SRC/recgen_out_moge" | head -1)
# everything but what the object mesh changes: the mesh, the object's trajectory and the contact correction
rsync -a --exclude recgen_out_moge --exclude hand_contact.json --exclude "${OBJ}_traj_*" --exclude "${OBJ}_points_cotracker*" \
    "$SRC/" "$RECON/"
cp "$RECON/mocap_joints_cotracker_raw.npz" "$RECON/mocap_joints_cotracker.npz"      # the placement before the contact step
echo "##### reconstruction: mesh (SAM 3D Objects), object, contact #####"
SKIP_FRAMES=1 SKIP_SAM3=1 SKIP_DEPTH=1 SKIP_CAMERA=1 SKIP_MOCAP=1 MESH_MODEL=sam3d DEPTH=da3 HAND_CONTACT=1 \
HANDS=mocap HAND_SIDES="left right" launch/reconstruct.sh "$KEY" "$RECON" 2>&1 | grep --line-buffered -v '^profiling' \
    | grep --line-buffered -E "^===|\[sam3d\]|\[contact\]|grasp at|release at|Error|Traceback"
cat "$RECON/hand_contact.json"
echo "##### reference #####"
OUT=data/references/$NAME
"$PY_SIM" scripts/rl/prepare_reference.py --recon "$RECON" --scene "$SCENE" --out "$OUT" --clearance-mode frame \
    --hand-table-clearance 0.008 --object-trajectory keyframes --anchor vision --object-target dense --overwrite 2>&1 \
    | grep --line-buffered -E "candidates tested|object:|object trajectory|Error|Traceback"
"$PY_SIM" scripts/rl/check_reference.py --recon "$RECON" --reference "$OUT" --scene "$SCENE" 2>&1 | grep -v '^profiling'
echo "=== done: $OUT ==="
