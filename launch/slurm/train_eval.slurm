#!/bin/bash
# One training run on one H100 (Jean Zay), with periodic rollout videos and wandb (offline; synced from the
# login node by launch/wandb_sync_loop.sh). Resumes the experiment if a previous job hit the time limit, and
# evaluates once training is complete. Submit through launch/submit_study.sh, or directly:
#
#   METHOD=dextag EXP=s1_dextag_layout ENV_NAME=bimanual_dextag NUM_ENVS=1024 ITERATIONS=15000 \
#       sbatch launch/slurm/train_eval.slurm
#
# Env: METHOD (dextag|recon) EXP ENV_NAME NUM_ENVS ITERATIONS VIDEO_EVERY (iterations between videos)
#      WANDB_RUN_GROUP EXTRA (extra overrides, e.g. "--env.data.max_clips=20")
#      EVAL_ALL=1 evaluate every checkpoint (best -> model_best.pt) for DexTaG too
#SBATCH -J tv2s
#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 20:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
source launch/config.sh
: "${METHOD:?}" "${EXP:?}" "${ENV_NAME:?}" "${NUM_ENVS:?}" "${ITERATIONS:?}"
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl TV2S_WANDB=1 WANDB_MODE=offline WANDB_SILENT=true
export WANDB_PROJECT="${WANDB_PROJECT:-tactile-video2sim}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
FILTER='^profiling\|ast.Str\|return index ==\|Falling back to convex hull\|qpos0) exceeds'
SCRIPT=scripts/policy/run_ppo_${METHOD}.py
EXP_DIR=logs/$EXP
echo "=== $EXP on $(hostname) ($SLURM_JOB_ID): $METHOD $ENV_NAME, $NUM_ENVS envs, $ITERATIONS iterations ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

done_marker() { ls -d "$EXP_DIR"/*/TRAINING_DONE >/dev/null 2>&1; }
has_ckpt() { ls "$EXP_DIR"/*/checkpoints/checkpoint_*.pt >/dev/null 2>&1; }

# ---------------------------------------------------------------- videos during training
RENDER_PID=
if ! done_marker; then
    # restarted if it dies before training ends (a renderer once exited silently); full log in logs/slurm
    (
        for attempt in 1 2 3 4 5; do
            "$PY_SIM" scripts/policy/render_rollouts.py --exp_name="$EXP" --method="$METHOD" --watch \
                --every="${VIDEO_EVERY:-0}" --poll=120 >> "logs/slurm/${EXP}_${SLURM_JOB_ID}_video.log" 2>&1
            code=$?
            echo "[video] renderer exited ($code), attempt $attempt"
            done_marker && break
            sleep 60
        done
    ) &
    RENDER_PID=$!
fi

# ---------------------------------------------------------------- train (or resume)
if ! done_marker; then
    RESUME=()
    has_ckpt && RESUME=(--resume_checkpoint=latest) && echo "resuming from the latest checkpoint"
    # shellcheck disable=SC2086
    "$PY_SIM" "$SCRIPT" --exp_name="$EXP" --env_name="$ENV_NAME" --num_envs="$NUM_ENVS" \
        --runner.total_iterations="$ITERATIONS" "${RESUME[@]}" ${EXTRA:-} 2>&1 | grep --line-buffered -v "$FILTER"
fi
if [ -n "$RENDER_PID" ]; then
    # the renderer exits after the final checkpoint's video
    for _ in $(seq 60); do kill -0 "$RENDER_PID" 2>/dev/null || break; sleep 30; done
    kill "$RENDER_PID" 2>/dev/null || true
fi
done_marker || { echo "training not finished (time limit?): the next job resumes"; exit 0; }

# ---------------------------------------------------------------- evaluate + final video + wandb
echo "=== evaluation: $EXP ==="
if [ "$METHOD" = recon ]; then
    "$PY_SIM" "$SCRIPT" --exp_name="$EXP" --eval --all_checkpoints 2>&1 | grep -v "$FILTER"
    "$PY_SIM" scripts/policy/render_rollouts.py --exp_name="$EXP" --method="$METHOD" --checkpoint=best --clips=6 --cols=3 \
        2>&1 | grep -v "$FILTER" || echo "final video failed (non-fatal)"
elif [ "${EVAL_ALL:-0}" = 1 ]; then
    "$PY_SIM" "$SCRIPT" --exp_name="$EXP" --eval --all_checkpoints 2>&1 | grep -v "$FILTER"
    "$PY_SIM" scripts/policy/render_rollouts.py --exp_name="$EXP" --method="$METHOD" --checkpoint=best --clips=6 --cols=3 \
        2>&1 | grep -v "$FILTER" || echo "final video failed (non-fatal)"
else
    "$PY_SIM" "$SCRIPT" --exp_name="$EXP" --eval 2>&1 | grep -v "$FILTER"
    "$PY_SIM" scripts/policy/render_rollouts.py --exp_name="$EXP" --method="$METHOD" --checkpoint=latest --clips=6 --cols=3 \
        2>&1 | grep -v "$FILTER" || echo "final video failed (non-fatal)"
    RUN=$(ls -d "$EXP_DIR"/*/ | sort | tail -1)
    LAST=$(ls "$RUN"checkpoints/checkpoint_*.pt | sort | tail -1)
    cp "$RUN"videos/rollout_"$(basename "$LAST" .pt)".mp4 "$RUN"videos/rollout_model_best.mp4 2>/dev/null || true
fi
(cd scripts/policy && "$PY_SIM" wandb_report.py eval "$EXP")
echo "=== done: $EXP ==="
