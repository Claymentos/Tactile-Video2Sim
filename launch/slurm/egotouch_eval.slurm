#!/bin/bash
# eval_rl_recon.py again on finished runs (one H100), e.g. after a new metric: the keyframe object errors.
# Env: EXPS (space-separated experiment names, logs/<exp>)  CKPTS="2000 best" (numbers, or best / best_inhand)
#SBATCH -J et_eval
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
: "${EXPS:?}"
source launch/config.sh
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
export QD_OFFLINE_CACHE_FILE_PATH="${JOBSCRATCH:-/tmp/$SLURM_JOB_ID}/quadrants"
mkdir -p "$QD_OFFLINE_CACHE_FILE_PATH"
cp -r "$HOME/.cache/quadrants/qdcache/." "$QD_OFFLINE_CACHE_FILE_PATH/" 2>/dev/null || true
FILTER='^profiling\|ast.Str\|Falling back to convex hull\|watertight'
for EXP in $EXPS; do
    RUN=$(ls -d "logs/$EXP"/*/ | tail -1)
    for C in ${CKPTS:-2000 best}; do
        case "$C" in best) P="$RUN/checkpoints/model_best.pt" ;; best_inhand) P="$RUN/checkpoints/model_best_inhand.pt" ;;
                     *) P="$C" ;; esac
        echo "=== $EXP checkpoint $C ==="
        "$PY_SIM" scripts/policy/eval_rl_recon.py --exp_name "$EXP" --checkpoint "$P" 2>&1 | grep --line-buffered -v "$FILTER" \
            | grep --line-buffered -E "from frame|in hand from|Error|Traceback" || echo "  (failed)"
    done
done
echo "=== done ==="
