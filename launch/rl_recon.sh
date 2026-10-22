#!/usr/bin/env bash
# One EgoTouch demonstration -> reconstruction -> RL reference -> rl_recon policy (Benchmark-Genesis' RL
# reconstruction, in this repo).
#
#   1  scene     Gemini (VLM=qwen: Qwen-VL, local) on the head video: which hands the demonstration uses and which
#                objects it shows -> data/egotouch/scene/<key>.json (skipped if present)     [PY_GEMINI / PY_QWEN]
#   2  recon     launch/reconstruct.sh: SAM3, MoGe, the camera from the table, the mocap placed on WiLoR + depth,
#                the RecGen mesh and CoTracker trajectory of the object the hands manipulate
#                -> data/egotouch/recon/<key> (skipped if done)                      [PY_SAM3, PY_MOGE, PY_RECGEN, PY_SIM]
#   3  reference scripts/rl/prepare_reference.py: the hand retargeted onto the Wuji, the simulated world, the
#                object's axis from its masks, the in-hand grasp pose -> data/references/NAME/reference.npz
#                                                                                         [PY_RETARGET, PY_SIM]
#   4  train     PPO on the env rl_recon (rl_recon/train.py's recipe: 2048 envs, 32 steps, assist decayed over half
#                of the run)                                                                                 [PY_SIM]
#   5  eval      every checkpoint from frame 0 and in hand, best ones copied (rl_recon/train.py's evaluation) [PY_SIM]
#
# Usage:   launch/rl_recon.sh NAME KEY      e.g. et_screwdriver pick_up_screwdriver__20260321_103227_213
#          NAME: the run (reference in data/references/NAME, logs in logs/NAME); KEY: the EgoTouch episode
# Knobs:   N_ENVS=2048 ITERATIONS=2000  TRAIN_ARGS="--env.train_start=random --env.tactile_reward=False ..."
#          RESUME=ckpt.pt FINETUNE=1   a fine-tune of a checkpoint (iterations and assist schedule restart)
#          REFERENCE=path/reference.npz   train on an existing reference (default data/references/NAME/reference.npz)
#          RECON_ARGS="FRAME=40 HAND_SIDES=left OBJECT=hammer TABLE_POINTS=..."   (launch/reconstruct.sh knobs)
#          PREP_ARGS="--side left --clearance-mode frame"   (prepare_reference.py options)
#          SKIP_SCENE=1 SKIP_RECON=1 SKIP_REFERENCE=1 SKIP_TRAIN=1 SKIP_EVAL=1   MODEL=gemini-3.6-flash   VLM=qwen
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
NAME="${1:?usage: launch/rl_recon.sh NAME KEY}"
KEY="${2:?usage: launch/rl_recon.sh NAME KEY}"
OUT="$REPO/data/references/$NAME"
REF="${REFERENCE:-$OUT/reference.npz}"
SCENE="$REPO/data/egotouch/scene/$KEY.json"
RECON="$REPO/data/egotouch/recon/$KEY"
export PY_RETARGET WUJI_RETARGETING_DIR

if [ "${SKIP_SCENE:-0}" != "1" ] && [ ! -e "$SCENE" ]; then
    IFS=$'\t' read -r VIDEO CONTEXT < <("$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from tv2s_data.egotouch import context, episode_from_key
ep = episode_from_key('$EGOTOUCH_ROOT', '$KEY'); print(ep.video, context(ep), sep='\t')")
    if [ "${VLM:-gemini}" = "qwen" ]; then
        echo "=== 1/5 Qwen-VL (local): hands and objects ==="
        PARSE=("$PY_QWEN" scripts/reconstruction/semantic_parsing.py --backend qwen ${MODEL:+--model "$MODEL"})
    else
        echo "=== 1/5 Gemini: hands and objects ==="
        [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set (VLM=qwen: a local model; or SKIP_SCENE=1 with RECON_ARGS=\"OBJECT=...\")"; exit 1; }
        PARSE=("$PY_GEMINI" scripts/reconstruction/semantic_parsing.py --model "${MODEL:-gemini-3.6-flash}")
    fi
    "${PARSE[@]}" --video "$VIDEO" --out "$SCENE" --fps 4 --context "$CONTEXT"
fi
if [ "${SKIP_RECON:-0}" != "1" ]; then
    if compgen -G "$RECON/*_traj_cotracker.npz" > /dev/null && [ -e "$RECON/mocap_joints_cotracker.npz" ]; then
        echo "=== 2/5 reconstruction: $RECON exists ==="
    else
        echo "=== 2/5 reconstruction of $KEY ==="
        # shellcheck disable=SC2086
        env ${RECON_ARGS:-} launch/reconstruct.sh "$KEY" "$RECON"
    fi
fi
if [ "${SKIP_REFERENCE:-0}" != "1" ]; then
    echo "=== 3/5 reference ==="
    SCENE_ARG=(); [ -e "$SCENE" ] && SCENE_ARG=(--scene "$SCENE")
    # shellcheck disable=SC2086
    "$PY_SIM" scripts/rl/prepare_reference.py --recon "$RECON" "${SCENE_ARG[@]}" --out "$OUT" ${PREP_ARGS:-} 2>&1 | grep -v "^profiling"
fi
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== 4/5 PPO ==="
    RES=(); [ -n "${RESUME:-}" ] && RES=(--resume_checkpoint="$RESUME")
    [ "${FINETUNE:-0}" = "1" ] && RES+=(--reset_iterations)
    # shellcheck disable=SC2086
    "$PY_SIM" scripts/policy/run_ppo_recon.py --exp_name="$NAME" --env.data.reference="$REF" --num_envs="${N_ENVS:-2048}" \
        --runner.total_iterations="${ITERATIONS:-2000}" "${RES[@]}" ${TRAIN_ARGS:-}
fi
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo "=== 5/5 evaluation (from frame 0 and in hand) ==="
    "$PY_SIM" scripts/policy/eval_rl_recon.py --exp_name "$NAME" --all
fi
echo "reference: $REF   run: logs/$NAME/<time>/ (checkpoints/model_best.pt, model_best_inhand.pt, eval_rl_recon/)"
