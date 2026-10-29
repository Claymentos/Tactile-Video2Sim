#!/usr/bin/env bash
# Paths and conda envs used by the launch scripts. Every value can be overridden from the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMANTOUCH_ROOT="${HUMANTOUCH_ROOT:-${SCRATCH:-$HOME}/HumanTouch/X001}"            # one HumanTouch task folder
TOUCHSCALE_ROOT="${TOUCHSCALE_ROOT:-${SCRATCH:-$HOME}/TouchScale}"                 # TouchScale (shards extracted in data/)
EGOTOUCH_ROOT="${EGOTOUCH_ROOT:-${SCRATCH:-$HOME}/EgoTouch}"                     # EgoTouch (scripts/data/download_egotouch.py)
ACE_ROOT="${ACE_ROOT:-${SCRATCH:-$HOME}/ACE-Data-0}"                           # ACE-Data-0 (gated hub dataset)
ACE_EPISODES="${ACE_EPISODES:-$REPO/data/ace/episodes}"                      # scripts/data/extract_ace.py's episode folders
DATA_DIR="${DATA_DIR:-$REPO/data/humantouch}"                          # prepared HumanTouch episodes
# Jean Zay layout ($WORK / $SCRATCH) when those exist, else the original home-directory layout
if [ -n "${WORK:-}" ] && [ -d "$WORK/ext/wuji-retargeting" ]; then
    export WUJI_RETARGETING_DIR="${WUJI_RETARGETING_DIR:-$WORK/ext/wuji-retargeting}"
else
    export WUJI_RETARGETING_DIR="${WUJI_RETARGETING_DIR:-$HOME/wuji-ego-mint/eval/simulate/wuji-retargeting}"
fi
if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/venvs/genesis" ]; then
    CONDA_BASE="${CONDA_BASE:-$SCRATCH/conda}"
    PY_SIM="${PY_SIM:-$SCRATCH/venvs/genesis/bin/python}"
    PY_DATA="${PY_DATA:-$PY_SIM}"
    [ ! -d "$EGOTOUCH_ROOT" ] && [ -d "$SCRATCH/egotouch/data" ] && EGOTOUCH_ROOT="$SCRATCH/egotouch/data"
    [ -x "$SCRATCH/venvs/qwen/bin/python" ] && PY_QWEN="${PY_QWEN:-$SCRATCH/venvs/qwen/bin/python}"
    [ -x "$SCRATCH/venvs/da3/bin/python" ] && PY_DA3="${PY_DA3:-$SCRATCH/venvs/da3/bin/python}"
    [ -d "$WORK/ext/recgen" ] && export RECGEN_DIR="${RECGEN_DIR:-$WORK/ext/recgen}"
    [ -d "$WORK/ext/MoGe" ] && MOGE_REPO="${MOGE_REPO:-$WORK/ext/MoGe}"
    [ -x "$SCRATCH/conda/envs/sam3d/bin/python" ] && PY_SAM3D="${PY_SAM3D:-$SCRATCH/conda/envs/sam3d/bin/python}"
    [ -d "$WORK/ext/sam-3d-objects" ] && SAM3D_DIR="${SAM3D_DIR:-$WORK/ext/sam-3d-objects}"
fi
# Genesis 1.3.3 times its two constraint solvers on 2 calls and, on an H100, keeps the decomposed (CUDA graph) one,
# 13x slower here than the monolith (rl_recon, 2048 envs: 4.4k vs 60k env steps/s, the same physics)
export QD_PERFDISPATCH_FORCE="${QD_PERFDISPATCH_FORCE:-func_solve_body:func_solve_body_monolith}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
PY_DATA="${PY_DATA:-$CONDA_BASE/envs/lerobot/bin/python}"              # pandas + pyarrow (parquet)
PY_SAM3D="${PY_SAM3D:-$CONDA_BASE/envs/sam3d-objects/bin/python}"        # SAM 3D Objects (reconstruct.sh MESH_MODEL=sam3d)
SAM3D_DIR="${SAM3D_DIR:-$HOME/sam-3d-objects}"                             # its checkout, checkpoints in checkpoints/hf
PY_RETARGET="${PY_RETARGET:-$CONDA_BASE/envs/wuji_retargeting/bin/python}"   # pinocchio + wuji_retargeting
PY_SIM="${PY_SIM:-$CONDA_BASE/envs/genesis/bin/python}"                # genesis-world, torch, trimesh
PY_GEMINI="${PY_GEMINI:-$CONDA_BASE/envs/sam3d/bin/python}"            # google-genai (semantic_parsing.py)
PY_QWEN="${PY_QWEN:-$CONDA_BASE/envs/lerobot/bin/python}"              # transformers + qwen-vl-utils (semantic_parsing.py --backend qwen)
PY_SAM3="${PY_SAM3:-$CONDA_BASE/envs/sam3/bin/python}"                 # SAM3 video segmentation, CoTracker, ffmpeg
PY_MOGE="${PY_MOGE:-$CONDA_BASE/envs/moge/bin/python}"                 # MoGe metric depth
PY_DA3="${PY_DA3:-$HOME/venvs/da3/bin/python}"                         # Depth-Anything-3 depth (reconstruct.sh DEPTH=da3): a venv over the moge env, ~/Depth-Anything-3 installed
PY_RECGEN="${PY_RECGEN:-$CONDA_BASE/envs/recgen/bin/python}"           # RecGen meshes
PY_HAWOR="${PY_HAWOR:-$CONDA_BASE/envs/hawor/bin/python}"              # HaWoR hands from the video (reconstruct.sh HANDS=hawor)
export HAWOR_DIR="${HAWOR_DIR:-$HOME/do-as-i-do/reconstruction/modules/HaWoR}"   # HaWoR checkout
export RECGEN_DIR="${RECGEN_DIR:-$HOME/recgen}"                          # RecGen checkout
RECGEN_CHECKPOINT="${RECGEN_CHECKPOINT:-recgen_base.multiview_stereo}"    # RecGen model (reconstruct.sh)
MOGE_CHECKPOINT="${MOGE_CHECKPOINT:-Ruicheng/moge-3-vitl}"               # MoGe model (reconstruct.sh)
DA3_CHECKPOINT="${DA3_CHECKPOINT:-depth-anything/DA3METRIC-LARGE}"     # Depth-Anything-3 model (reconstruct.sh DEPTH=da3)
MOGE_REPO="${MOGE_REPO:-$HOME/MoGe}"                                      # MoGe checkout, when not installed in its env
