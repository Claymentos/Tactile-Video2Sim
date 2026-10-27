#!/usr/bin/env bash
# Object-trajectory comparison on EgoTouch: per one-hand demonstration, one job reconstructs it and builds two rl_recon
# references that differ only by the object's trajectory, then one rl_recon training per reference, in one wandb group:
#   kf  --object-trajectory keyframes   the object measured on its reliable frames only (at rest), carried by the palm
#                                       while the glove holds it
#   tr  --object-trajectory tracked     CoTracker + Kalman on every frame of the video
# The object (SAM3 prompt) and the hand come from the dataset's manifest.jsonl: nothing is set per video.
#
#   launch/submit_traj_compare.sh GROUP EPISODES     EPISODES: comma-separated, e.g.
#                                                    Workbench/pick_up_screwdriver/20260321_103227_213,...
# Knobs: ITERATIONS=2000 N_ENVS=2048 VIDEO_EVERY=500 SAVE_EVERY=100 FRAME=0 (reference frame)  SKIP_REFS=1
#        AFTER=<job id> (with SKIP_REFS=1: the trainings wait for that references job)
#        QOS=qos_gpu_h100-t3 (trainings)  REFS_QOS=qos_gpu_h100-dev  EXTRA="..." (run_ppo_recon.py overrides, both variants)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source launch/config.sh
G="${1:?usage: launch/submit_traj_compare.sh GROUP EPISODES}"
EPS="${2:?usage: launch/submit_traj_compare.sh GROUP EPISODES}"
mkdir -p logs/slurm
SB=(--qos="${QOS:-qos_gpu_h100-t3}" --parsable --export=ALL)
# jobs starting together compile Genesis' kernels into the same ~/.cache/quadrants and some then read a broken
# kernel (episodes of 1 step, fingers 17 cm off): the trainings start 2 min apart
I=0
export WANDB_RUN_GROUP="$G" ITERATIONS="${ITERATIONS:-2000}" N_ENVS="${N_ENVS:-2048}" VIDEO_EVERY="${VIDEO_EVERY:-500}"

for EP in ${EPS//,/ }; do
    IFS='|' read -r KEY OBJECT SIDE < <(python3 -c "
import json, sys
rows = [json.loads(l) for l in open('$EGOTOUCH_ROOT/manifest.jsonl')]
r = [r for r in rows if r['episode'] == '$EP']
if not r: sys.exit('$EP not in manifest.jsonl')
r = r[0]; print(r['task'].split('/')[1] + '__' + r['episode'].split('/')[-1], r['object'], r['side'], sep='|')")
    NAME=et_${KEY//__/_}
    DEP=(); [ -n "${AFTER:-}" ] && DEP=(--dependency=afterok:$AFTER)
    if [ "${SKIP_REFS:-0}" != 1 ]; then
        J=$(KEY="$KEY" NAME="$NAME" OBJECT="$OBJECT" SIDE="$SIDE" sbatch "${SB[@]}" --qos="${REFS_QOS:-qos_gpu_h100-dev}" -t 02:00:00 -J "refs_$NAME" launch/slurm/egotouch_refs.slurm)
        DEP=(--dependency=afterok:$J)
        echo "$KEY ('$OBJECT', $SIDE): references $J -> logs/slurm/refs_${NAME}_$J.out"
    fi
    for V in kf tr; do
        EXP="${G}_${NAME}_$V"
        J2=$(KEY="$KEY" REF="data/references/${NAME}_$V/reference.npz" EXP="$EXP" sbatch "${SB[@]}" "${DEP[@]}" \
            --begin=now+$((120 * I)) -J "$EXP" launch/slurm/egotouch_train.slurm)
        I=$((I + 1))
        echo "    $V: $J2 -> logs/slurm/${EXP}_$J2.out, logs/$EXP"
    done
done
echo "wandb group $G (offline; launch/wandb_sync_loop.sh uploads from the login node)"
