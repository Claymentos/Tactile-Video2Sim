#!/usr/bin/env bash
# README_egotouch.md's "Recette complète" (et_hammer_da3_contact_arm) on several EgoTouch demonstrations: per episode,
# one job builds the reference (launch/slurm/egotouch_refs_readme.slurm: Qwen3-VL-4B scene, DA3 + contact, --scene),
# then two trainings on it (launch/slurm/egotouch_train.slurm, the README's step 7), in one wandb group:
#   float  the Wuji hand on its virtual 6-DoF base
#   arm    the Wuji hand on a Franka Panda (--env.arm.enabled=True)
#
#   launch/submit_readme.sh GROUP KEYS      KEYS: comma-separated, e.g. pick_up_hammer__20260321_095958_211,...
# Knobs: ITERATIONS=2000 N_ENVS=2048 SAVE_EVERY=25 (the runner's default, as in the README)  SKIP_REFS=1
#        QOS=qos_gpu_h100-t3 (trainings)  REFS_QOS=qos_gpu_h100-dev  SUFFIX=rda3 (the references' and recons' suffix)
#        QUALITY=enforce|report (egotouch_refs_readme.slurm)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G="${1:?usage: launch/submit_readme.sh GROUP KEYS}"
KEYS="${2:?usage: launch/submit_readme.sh GROUP KEYS}"
mkdir -p logs/slurm
SB=(--qos="${QOS:-qos_gpu_h100-t3}" --parsable --export=ALL)
export WANDB_RUN_GROUP="$G" ITERATIONS="${ITERATIONS:-2000}" N_ENVS="${N_ENVS:-2048}" SAVE_EVERY="${SAVE_EVERY:-25}"

source launch/config.sh
# compute nodes have no internet: every file of the episodes (jq_pressure.json, ...) fetched here first
for KEY in ${KEYS//,/ }; do
    "$PY_SIM" scripts/data/download_egotouch.py --root "$EGOTOUCH_ROOT" --tasks "${KEY%%__*}" --episodes "${KEY##*__}" \
        | tail -1
done
for KEY in ${KEYS//,/ }; do
    NAME=et_${KEY//__/_}_${SUFFIX:-rda3}
    DEP=()
    if [ "${SKIP_REFS:-0}" != 1 ]; then
        J=$(KEY="$KEY" NAME="$NAME" RECON="data/egotouch/recon/${KEY}_${SUFFIX:-readme}" sbatch "${SB[@]}" --qos="${REFS_QOS:-qos_gpu_h100-dev}" -J "refs_$NAME" \
            launch/slurm/egotouch_refs_readme.slurm)
        DEP=(--dependency=afterok:$J)
        echo "$KEY: reference $J -> logs/slurm/refs_${NAME}_$J.out"
    fi
    for V in float arm; do
        EXP="${G}_${NAME}_$V"
        X=""; [ "$V" = arm ] && X="--env.arm.enabled=True"
        # wandb name: <dataset>_<task (the object)>_<franka | mains>
        RN="egotouch_${KEY%%__*}_$([ "$V" = arm ] && echo franka || echo mains)"
        J2=$(WANDB_RUN_NAME="$RN" REF="data/references/$NAME/reference.npz" EXP="$EXP" EXTRA="$X" sbatch "${SB[@]}" "${DEP[@]}" -J "$EXP" \
            launch/slurm/egotouch_train.slurm)
        echo "    $V: $J2 -> logs/slurm/${EXP}_$J2.out, logs/$EXP"
    done
done
echo "wandb group $G (offline; launch/wandb_sync_loop.sh uploads from the login node)"
