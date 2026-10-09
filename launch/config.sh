#!/usr/bin/env bash
# Paths and conda envs used by the launch scripts. Every value can be overridden from the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMANTOUCH_ROOT="${HUMANTOUCH_ROOT:-$HOME/HumanTouch/X001}"            # one HumanTouch task folder
DATA_DIR="${DATA_DIR:-$REPO/data/humantouch}"                          # prepared episodes, retargeting, clips
export WUJI_RETARGETING_DIR="${WUJI_RETARGETING_DIR:-$HOME/wuji-ego-mint/eval/simulate/wuji-retargeting}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
PY_DATA="${PY_DATA:-$CONDA_BASE/envs/lerobot/bin/python}"              # pandas + pyarrow (parquet)
PY_RETARGET="${PY_RETARGET:-$CONDA_BASE/envs/wuji_retargeting/bin/python}"   # pinocchio + wuji_retargeting
PY_SIM="${PY_SIM:-$CONDA_BASE/envs/genesis/bin/python}"                # genesis-world, torch, trimesh
