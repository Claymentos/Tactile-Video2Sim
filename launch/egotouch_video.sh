#!/usr/bin/env bash
# One EgoTouch video, end to end, nothing to adapt per video: download (here, on the login node: compute nodes
# have no internet), then a GPU job from the raw episode to an episode clip that passed the contact check
# (launch/slurm/egotouch_video.slurm), then the training from frame 0 with rollout videos on wandb
# (launch/slurm/train_eval.slurm), which only starts if the first job succeeded.
#
#   launch/egotouch_video.sh EP            EP: recording time, e.g. 20260314_151150_129 (Home/pick_up_charger)
#
# Knobs: METHOD=recon (|dextag)  ITERATIONS=1500  NUM_ENVS=2048  VIDEO_EVERY=500  NAME=et_<task>_<time>
#        WANDB_RUN_GROUP=<NAME>  MIN_F1=0.1  SKIP_PIPELINE=1 (clip already built)  QOS=qos_gpu_h100-t3 (training)  PIPELINE_QOS=qos_gpu_h100-dev  TRAIN_TIME=12:00:00
#        TRAIN_EXTRA="--env.assist_gain=..."  (more run_ppo_<method>.py overrides)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source launch/config.sh
EP="${1:?usage: launch/egotouch_video.sh RECORDING_TIME}"
METHOD="${METHOD:-recon}"
mkdir -p logs/slurm

if [ "${SKIP_PIPELINE:-0}" != "1" ]; then
    echo "=== download $EP -> $EGOTOUCH_ROOT ==="
    "$PY_SIM" scripts/data/download_egotouch.py --root "$EGOTOUCH_ROOT" --episodes "$EP"
fi
KEY=$("$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from tv2s_data.egotouch import select_episodes
e = select_episodes('$EGOTOUCH_ROOT', '$EP'); assert len(e) == 1, f'{len(e)} episodes match $EP'; print(e[0].key)")
NAME="${NAME:-et_${KEY//__/_}}"
CLIP_DIR=data/episodes/$NAME
SB=(--qos="${QOS:-qos_gpu_h100-t3}" --parsable --export=ALL)

DEP=()
if [ "${SKIP_PIPELINE:-0}" != "1" ]; then
    J=$(EP="$EP" CLIP_DIR="$CLIP_DIR" sbatch "${SB[@]}" --qos="${PIPELINE_QOS:-qos_gpu_h100-dev}" -t 02:00:00 \
        -J "pipe_$NAME" launch/slurm/egotouch_video.slurm)
    echo "pipeline job $J: logs/slurm/pipe_${NAME}_$J.out"
    DEP=(--dependency=afterok:"$J")
fi
# the trained setup, the same for every video (rl_recon's assist carries the object to the tactile grasp pose)
C="--env.data.clip_dir=$CLIP_DIR --env.active_hands=auto --env.train_start=first --runner.save_interval=${VIDEO_EVERY:-500}"
[ "$METHOD" = recon ] && C="$C --env.assist_gain=1.0 --env.assist_tau=0.03"
T=$(METHOD="$METHOD" EXP="${NAME}_$METHOD" ENV_NAME="bimanual_$METHOD" NUM_ENVS="${NUM_ENVS:-2048}" \
    ITERATIONS="${ITERATIONS:-1500}" VIDEO_EVERY="${VIDEO_EVERY:-500}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$NAME}" \
    EXTRA="$C ${TRAIN_EXTRA:-}" sbatch "${SB[@]}" "${DEP[@]}" -t "${TRAIN_TIME:-12:00:00}" -J "${NAME}_$METHOD" \
    launch/slurm/train_eval.slurm)
echo "training job $T (${DEP[*]:-no dependency}): logs/slurm/${NAME}_${METHOD}_$T.out, wandb group ${WANDB_RUN_GROUP:-$NAME}"
