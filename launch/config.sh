#!/usr/bin/env bash
# Paths and conda envs used by the launch scripts. Every value can be overridden from the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMANTOUCH_ROOT="${HUMANTOUCH_ROOT:-${SCRATCH:-$HOME}/HumanTouch/X001}"            # one HumanTouch task folder
TOUCHSCALE_ROOT="${TOUCHSCALE_ROOT:-${SCRATCH:-$HOME}/TouchScale}"                 # TouchScale (shards extracted in data/)
EGOTOUCH_ROOT="${EGOTOUCH_ROOT:-${SCRATCH:-$HOME}/EgoTouch}"                     # EgoTouch (scripts/data/download_egotouch.py)
DATA_DIR="${DATA_DIR:-$REPO/data/humantouch}"                          # prepared episodes, retargeting, clips
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
fi
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
PY_DATA="${PY_DATA:-$CONDA_BASE/envs/lerobot/bin/python}"              # pandas + pyarrow (parquet)
PY_RETARGET="${PY_RETARGET:-$CONDA_BASE/envs/wuji_retargeting/bin/python}"   # pinocchio + wuji_retargeting
PY_SIM="${PY_SIM:-$CONDA_BASE/envs/genesis/bin/python}"                # genesis-world, torch, trimesh
PY_GEMINI="${PY_GEMINI:-$CONDA_BASE/envs/sam3d/bin/python}"            # google-genai (semantic_parsing.py)
PY_QWEN="${PY_QWEN:-$CONDA_BASE/envs/lerobot/bin/python}"              # transformers + qwen-vl-utils (semantic_parsing.py --backend qwen)
PY_SAM3="${PY_SAM3:-$CONDA_BASE/envs/sam3/bin/python}"                 # SAM3 video segmentation
PY_MOGE="${PY_MOGE:-$CONDA_BASE/envs/moge/bin/python}"                 # MoGe metric depth
PY_RECGEN="${PY_RECGEN:-$CONDA_BASE/envs/recgen/bin/python}"           # RecGen meshes
PY_FP="${PY_FP:-$CONDA_BASE/envs/foundationpose/bin/python}"          # FoundationPose tracking
export RECGEN_DIR="${RECGEN_DIR:-$HOME/recgen}"                          # RecGen checkout
