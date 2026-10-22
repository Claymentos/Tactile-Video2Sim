#!/usr/bin/env bash
# EgoTouch study on Jean Zay: per one-hand demonstration, a perception job (scene + reference) then two rl_recon
# jobs, with and without the tactile reward, in one wandb group. Run after launch/prepare_egotouch.sh (Gemini scenes).
#
#   launch/submit_egotouch.sh PREFIX KEYS        KEYS: comma-separated EgoTouch keys (or @file with them)
#
# Demonstrations whose Gemini scene says hands_used = both (or has no scene) are skipped.
# Knobs: ITERATIONS=500 N_ENVS=2048 SKIP_PERCEPTION=1 (references already built)  QOS=qos_gpu_h100-t3
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P="${1:?usage: launch/submit_egotouch.sh PREFIX KEYS}"
KEYS="${2:?usage: launch/submit_egotouch.sh PREFIX KEYS}"
[ "${KEYS:0:1}" = @ ] && KEYS=$(cat "${KEYS:1}")
mkdir -p logs/slurm
export WANDB_RUN_GROUP="$P" ITERATIONS="${ITERATIONS:-500}" N_ENVS="${N_ENVS:-2048}"
SB=(--qos="${QOS:-qos_gpu_h100-t3}" --parsable)

for KEY in ${KEYS//,/ }; do
    SCENE=data/egotouch/scene/$KEY.json
    [ -e "$SCENE" ] || { echo "skip $KEY: no scene"; continue; }
    HANDS=$(python3 -c "import json; print(json.load(open('$SCENE')).get('hands_used', '?'))")
    [ "$HANDS" = left ] || [ "$HANDS" = right ] || { echo "skip $KEY: hands_used=$HANDS"; continue; }
    NAME=et_${KEY//__/_}
    DEP=()
    if [ "${SKIP_PERCEPTION:-0}" != 1 ]; then
        J=$(KEY="$KEY" NAME="$NAME" sbatch "${SB[@]}" -J "perc_$NAME" --export=ALL launch/slurm/egotouch_perception.slurm)
        DEP=(--dependency=afterok:$J)
        echo "$KEY ($HANDS): perception $J"
    fi
    for T in 1 0; do
        V=$([ $T = 1 ] && echo tac || echo notac)
        J2=$(KEY="$KEY" NAME="$NAME" EXP="${P}_${NAME}_$V" TACTILE=$T sbatch "${SB[@]}" "${DEP[@]}" -J "${V}_$NAME" --export=ALL \
            launch/slurm/egotouch_rl.slurm)
        echo "    $V: $J2 -> logs/${P}_${NAME}_$V"
    done
done
