#!/bin/bash
# Debug of the reference render abort (SIGABRT after ~50 frames): Python traceback on abort, glibc/EGL details.
#SBATCH -J render_debug
#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm/%x_%j.out
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
source launch/config.sh
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export QD_OFFLINE_CACHE_FILE_PATH="${JOBSCRATCH:-/tmp/$SLURM_JOB_ID}/quadrants"
mkdir -p "$QD_OFFLINE_CACHE_FILE_PATH"; cp -r "$HOME/.cache/quadrants/qdcache/." "$QD_OFFLINE_CACHE_FILE_PATH/" 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv
ulimit -c 0
"$PY_SIM" -X faulthandler scripts/policy/visualize_episode.py \
    --reference "${REF:-data/references/et_pick_up_level_20260321_095407_102_rob1_rt07/reference.npz}" --out "${OUT:-outputs/episodes/render_debug}" &
pid=$!
while kill -0 $pid 2>/dev/null; do [ "${NO_SMI:-0}" = 1 ] || nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; grep -E "VmRSS|Threads" /proc/$pid/status 2>/dev/null | tr '\n' ' '; ls /proc/$pid/fd 2>/dev/null | wc -l; sleep 10; done
wait $pid; echo "exit $?"
