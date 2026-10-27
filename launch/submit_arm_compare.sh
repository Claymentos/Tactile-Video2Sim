#!/usr/bin/env bash
# Franka comparison on EgoTouch: per one-hand demonstration, one job builds the "DA3 + contact" reference
# (launch/slurm/egotouch_refs_da3.slurm), then two rl_recon trainings on it, in one wandb group:
#   float  the Wuji hand on its virtual 6-DoF base (default)
#   arm    the Wuji hand on a Franka Panda (--env.arm.enabled=True, the base placed by IK over the clip)
# The object (SAM3 prompt) and the hand come from the dataset's manifest.jsonl: nothing is set per video.
#
#   launch/submit_arm_compare.sh GROUP EPISODES     EPISODES: comma-separated, e.g.
#                                                   Workbench/pick_up_screwdriver/20260321_103227_213,...
# Knobs: ITERATIONS=2000 N_ENVS=2048 SAVE_EVERY=100 FRAME=0 (reference frame)  SKIP_REFS=1
#        QOS=qos_gpu_h100-t3 (trainings)  REFS_QOS=qos_gpu_h100-dev  EXTRA="..." (run_ppo_recon.py overrides, both variants)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source launch/config.sh
G="${1:?usage: launch/submit_arm_compare.sh GROUP EPISODES}"
EPS="${2:?usage: launch/submit_arm_compare.sh GROUP EPISODES}"
mkdir -p logs/slurm
SB=(--qos="${QOS:-qos_gpu_h100-t3}" --parsable --export=ALL)
export WANDB_RUN_GROUP="$G" ITERATIONS="${ITERATIONS:-2000}" N_ENVS="${N_ENVS:-2048}"

for EP in ${EPS//,/ }; do
    IFS='|' read -r KEY OBJECT SIDE < <(python3 -c "
import json, sys
rows = [json.loads(l) for l in open('$EGOTOUCH_ROOT/manifest.jsonl')]
r = [r for r in rows if r['episode'] == '$EP']
if not r: sys.exit('$EP not in manifest.jsonl')
r = r[0]; print(r['task'].split('/')[1] + '__' + r['episode'].split('/')[-1], r['object'], r['side'], sep='|')")
    NAME=et_${KEY//__/_}_da3c
    DEP=()
    if [ "${SKIP_REFS:-0}" != 1 ]; then
        J=$(KEY="$KEY" NAME="$NAME" OBJECT="$OBJECT" SIDE="$SIDE" sbatch "${SB[@]}" --qos="${REFS_QOS:-qos_gpu_h100-dev}" \
            -J "refs_$NAME" launch/slurm/egotouch_refs_da3.slurm)
        DEP=(--dependency=afterok:$J)
        echo "$KEY ('$OBJECT', $SIDE): reference $J -> logs/slurm/refs_${NAME}_$J.out"
    fi
    for V in float arm; do
        EXP="${G}_${NAME}_$V"
        X="${EXTRA:-}"; [ "$V" = arm ] && X="--env.arm.enabled=True $X"
        J2=$(REF="data/references/$NAME/reference.npz" EXP="$EXP" EXTRA="$X" sbatch "${SB[@]}" "${DEP[@]}" -J "$EXP" \
            launch/slurm/egotouch_train.slurm)
        echo "    $V: $J2 -> logs/slurm/${EXP}_$J2.out, logs/$EXP"
    done
done
echo "wandb group $G (offline; launch/wandb_sync_loop.sh uploads from the login node)"
