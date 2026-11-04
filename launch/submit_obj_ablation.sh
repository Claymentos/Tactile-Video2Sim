#!/usr/bin/env bash
# Object-information ablation of the rl_recon reward (hands only, kf1's *_rob1_rt07 references), 5 EgoTouch clips x 4
# rewards. wandb name egotouch_<task>_<reward>, group obj1 (one group: filter by reward in the run name).
#   tac           tactile + hand tracking only: no object term, no object termination
#   tac_objterm   same reward, but the episode still ends when the object leaves its dense trajectory by > 20 cm
#   obj_keyframes object terms at the measured keyframes only (termination there too)
#   obj_dense     object terms on every frame of the reference trajectory
# Usage (login node): [ACCOUNT=iyg@h100] [QOS=qos_gpu_h100-t3|qos_gpu_h100-dev] launch/submit_obj_ablation.sh [GROUP]
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GROUP=${1:-obj1}
export ITERATIONS=${ITERATIONS:-2000} N_ENVS=${N_ENVS:-2048} SAVE_EVERY=${SAVE_EVERY:-25}
# clip -> name in wandb (the second hammer carries its clip time: two hammers otherwise share a name)
declare -A TASK=(
  [pick_up_hammer_20260321_095958_211]=pick_up_hammer
  [pick_up_hammer_20260321_095941_028]=pick_up_hammer_095941
  [pick_up_wrench_20260321_094952_783]=pick_up_wrench
  [pick_up_screwdriver_20260321_103242_339]=pick_up_screwdriver
  [pick_up_level_20260321_095407_102]=pick_up_level
)
declare -A RW=(
  [tac]="--env.obj_reward=dense --env.w_obj=0 --env.w_obj_axis=0 --env.term_obj=1000"
  [tac_objterm]="--env.obj_reward=dense --env.w_obj=0 --env.w_obj_axis=0"
  [obj_keyframes]="--env.obj_reward=keyframes"
  [obj_dense]="--env.obj_reward=dense"
)
# QOS=qos_gpu_h100-t3 (default): one 1-GPU job per run (launch/slurm/egotouch_train.slurm, 10 h), staggered starts
# (jobs starting together raced on Genesis' kernel cache); QOS=qos_gpu_h100-dev: one 2-h job per clip, its 4 rewards
# side by side on 4 GPUs (launch/slurm/egotouch_train_multi.slurm; the dev QoS allows 10 submitted jobs per user)
QOS=${QOS:-qos_gpu_h100-t3}
i=0
for k in "${!TASK[@]}"; do
  REF=data/references/et_${k}_rob1_rt07/reference.npz
  [ -f "$REF" ] || { echo "missing $REF"; exit 1; }
  if [ "$QOS" = qos_gpu_h100-dev ]; then
    RUNS=""
    for r in tac tac_objterm obj_keyframes obj_dense; do RUNS+="egotouch_${TASK[$k]}_$r=${RW[$r]};"; done
    id=$(GROUP=$GROUP REF=$REF RUNS="${RUNS%;}" \
         sbatch -A "${ACCOUNT:-iyg@h100}" --qos="$QOS" --parsable --export=ALL \
           -J "${GROUP}_egotouch_${TASK[$k]}" launch/slurm/egotouch_train_multi.slurm)
    echo "$id ${GROUP}_egotouch_${TASK[$k]}"
    continue
  fi
  for r in tac tac_objterm obj_keyframes obj_dense; do
    name=egotouch_${TASK[$k]}_$r
    id=$(WANDB_RUN_GROUP=$GROUP WANDB_RUN_NAME=$name REF=$REF EXP=${GROUP}_$name EXTRA="${RW[$r]}" \
         sbatch -A "${ACCOUNT:-iyg@h100}" --qos="$QOS" --begin=now+$((i * 30)) --parsable --export=ALL \
           -J "${GROUP}_$name" launch/slurm/egotouch_train.slurm)
    echo "$id ${GROUP}_$name"
    i=$((i + 1))
  done
done
