#!/usr/bin/env bash
# Benchmark-Genesis' RL reconstruction (rl_recon) of one demonstration, in this repo (rl_recon_env.py).
#
#   1  hands     Gemini: which hands the demonstration uses -> <OUT>/scene.json (hands_used; skipped if present)  [PY_GEMINI]
#   2  reference the per-episode reference.npz, for those hands (scripts/rl/prepare_reference.py):              [PY_SIM]
#                  BG_RECON=<Benchmark-Genesis reconstruction of the video>  Benchmark-Genesis' own chain (one hand)
#                  CLIP=<episode clip .pkl of this pipeline>                 this pipeline's hands and objects
#   3  train     PPO on the env rl_recon (rl_recon/train.py's recipe: 2048 envs, 32 steps, assist decayed over
#                half of the run)                                                                              [PY_SIM]
#   4  eval      every checkpoint from frame 0 and in hand, best ones copied (rl_recon/train.py's evaluation)  [PY_SIM]
#
# Usage:   launch/rl_recon.sh NAME VIDEO
#          NAME: the run (references in data/references/NAME, logs in logs/NAME)
#          VIDEO: the demonstration's head-camera video (Gemini's input)
# Knobs:   BG_RECON=dir [EPISODE=dir] [OBJECT=id] | CLIP=file.pkl     the reference's source (one of them)
#          N_ENVS=2048 ITERATIONS=2000  TRAIN_ARGS="--env.tactile_reward=False ..."   (run_ppo_recon.py arguments)
#          RESUME=ckpt.pt FINETUNE=1   a fine-tune of a checkpoint (iterations and assist schedule restart)
#          REFERENCE=path/reference.npz   train on an existing reference (default data/references/NAME/reference.npz)
#          SKIP_HANDS=1 SKIP_REFERENCE=1 SKIP_TRAIN=1 SKIP_EVAL=1   MODEL=gemini-3.6-flash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
NAME="${1:?usage: launch/rl_recon.sh NAME VIDEO}"
VIDEO="${2:?usage: launch/rl_recon.sh NAME VIDEO}"
OUT="$REPO/data/references/$NAME"
REF="${REFERENCE:-$OUT/reference.npz}"
mkdir -p "$OUT"

if [ "${SKIP_HANDS:-0}" != "1" ] && [ ! -e "$OUT/scene.json" ]; then
    echo "=== 1/4 Gemini: hands used ==="
    [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set (or SKIP_HANDS=1 with SIDE=left|right in PREP_ARGS)"; exit 1; }
    "$PY_GEMINI" scripts/reconstruction/semantic_parsing.py --video "$VIDEO" --out "$OUT/scene.json" \
        --model "${MODEL:-gemini-3.6-flash}" --fps 4 ${CONTEXT:+--context "$CONTEXT"}
fi
if [ "${SKIP_REFERENCE:-0}" != "1" ]; then
    echo "=== 2/4 reference ==="
    SRC=()
    if [ -n "${BG_RECON:-}" ]; then
        SRC=(--bg-recon "$BG_RECON")
        [ -n "${EPISODE:-}" ] && SRC+=(--episode "$EPISODE")
        [ -n "${OBJECT:-}" ] && SRC+=(--object "$OBJECT")
    elif [ -n "${CLIP:-}" ]; then
        SRC=(--clip "$CLIP")
    else
        echo "BG_RECON=<Benchmark-Genesis reconstruction> or CLIP=<episode clip .pkl>"; exit 1
    fi
    SCENE=(); [ -e "$OUT/scene.json" ] && SCENE=(--scene "$OUT/scene.json")
    # shellcheck disable=SC2086
    "$PY_SIM" scripts/rl/prepare_reference.py "${SRC[@]}" "${SCENE[@]}" --out "$OUT" ${PREP_ARGS:-} 2>&1 | grep -v "^profiling"
fi
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== 3/4 PPO ==="
    RES=(); [ -n "${RESUME:-}" ] && RES=(--resume_checkpoint="$RESUME")
    [ "${FINETUNE:-0}" = "1" ] && RES+=(--reset_iterations)
    # shellcheck disable=SC2086
    "$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="$NAME" --env_name=rl_recon \
        --env.data.reference="$REF" --num_envs="${N_ENVS:-2048}" \
        --runner.total_iterations="${ITERATIONS:-2000}" "${RES[@]}" ${TRAIN_ARGS:-}
fi
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo "=== 4/4 evaluation (from frame 0 and in hand) ==="
    "$PY_SIM" scripts/policy/eval_rl_recon.py --exp_name "$NAME" --all
fi
echo "reference: $REF   run: logs/$NAME/<time>/ (checkpoints/model_best.pt, model_best_inhand.pt, eval_rl_recon/)"
