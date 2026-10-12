#!/usr/bin/env bash
# Method x mapping study on Jean Zay: 4 Slurm jobs in parallel (one H100 each), all logged to wandb in one group.
#
#   launch/submit_study.sh smoke PREFIX     dev QoS (<= 2 h): short runs, checks training / videos / eval / wandb
#   launch/submit_study.sh episode PREFIX EP  one HumanTouch episode (one video, compare_on_episode.sh's recipe:
#                                           DexTaG 1500 it, rl_recon 500 it, 1024 envs, aids rescaled, every
#                                           checkpoint evaluated), both methods x both mappings
#   launch/submit_study.sh full  PREFIX     repo defaults (DexTaG 15000 it x 1024 envs, rl_recon 1500 it x 2048 envs);
#                                           DexTaG is chained over DEXTAG_CHAIN jobs of 20 h (resume), then a
#                                           CPU job logs the comparison run once all four are done
# Knobs: DEXTAG_ITERATIONS RECON_ITERATIONS DEXTAG_ENVS RECON_ENVS DEXTAG_CHAIN VIDEO_EVERY_DEXTAG VIDEO_EVERY_RECON
#        EXTRA (overrides passed to every run)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:?usage: launch/submit_study.sh smoke|full PREFIX}"
P="${2:?usage: launch/submit_study.sh smoke|full PREFIX}"
mkdir -p logs/slurm
export WANDB_RUN_GROUP="$P" EXTRA="${EXTRA:-}"

EXTRA_DEXTAG="" EXTRA_RECON=""
if [ "$MODE" = episode ]; then
    EP="${3:?usage: launch/submit_study.sh episode PREFIX EPISODE}"
    source launch/config.sh
    CLIPS=$("$PY_SIM" -c "
import json, sys
idx = json.load(open('$DATA_DIR/clips/index.json'))['clips']
names = [c['name'] for c in idx if c['episode'] == $EP]
sys.exit('no clip for episode $EP') if not names else print(','.join(names))")
    echo "episode $EP: $CLIPS"
    SB=(--qos=qos_gpu_h100-t3 -t 12:00:00)
    DI="${DEXTAG_ITERATIONS:-1500}" RI="${RECON_ITERATIONS:-500}"
    VD="${VIDEO_EVERY_DEXTAG:-100}" VR="${VIDEO_EVERY_RECON:-50}"
    CHAIN=1
    SAVE_DEXTAG="--runner.save_interval=50" SAVE_RECON="--runner.save_interval=50"
    DEXTAG_ENVS="${DEXTAG_ENVS:-1024}" RECON_ENVS="${RECON_ENVS:-1024}"
    EXTRA="$EXTRA --env.data.clips=$CLIPS"
    EXTRA_DEXTAG="--env.object_guide_decay_steps=$((DI / 2)) --env.objects.perturbation_curriculum=$((DI * 3 / 10)),$((DI * 8 / 10))"
    export EVAL_ALL=1
elif [ "$MODE" = smoke ]; then
    SB=(--qos=qos_gpu_h100-dev -t 01:30:00)
    DI="${DEXTAG_ITERATIONS:-200}" RI="${RECON_ITERATIONS:-100}"
    VD="${VIDEO_EVERY_DEXTAG:-100}" VR="${VIDEO_EVERY_RECON:-50}"
    CHAIN=1
    SAVE_DEXTAG="--runner.save_interval=50" SAVE_RECON="--runner.save_interval=25"
else
    SB=(--qos=qos_gpu_h100-t3 -t 20:00:00)
    DI="${DEXTAG_ITERATIONS:-15000}" RI="${RECON_ITERATIONS:-1500}"
    VD="${VIDEO_EVERY_DEXTAG:-250}" VR="${VIDEO_EVERY_RECON:-100}"
    CHAIN="${DEXTAG_CHAIN:-3}"
    SAVE_DEXTAG="--runner.save_interval=250" SAVE_RECON="--runner.save_interval=25"   # videos at checkpoints
fi

JOBS=()
submit() {  # method env_name mapping num_envs iterations video_every chain
    local method=$1 env=$2 mapping=$3 envs=$4 its=$5 every=$6 chain=$7 exp="${P}_$1_$3" dep=() id save
    [ "$method" = dextag ] && save="$SAVE_DEXTAG $EXTRA_DEXTAG" || save="$SAVE_RECON $EXTRA_RECON"
    for _ in $(seq "$chain"); do
        id=$(METHOD=$method EXP=$exp ENV_NAME=$env NUM_ENVS=$envs ITERATIONS=$its VIDEO_EVERY=$every \
             EXTRA="$EXTRA $save" sbatch --parsable "${SB[@]}" -J "$exp" "${dep[@]}" launch/slurm/train_eval.slurm)
        echo "$exp: job $id ${dep[*]}"
        dep=(--dependency=afterany:$id)
    done
    JOBS+=("$id")
}
submit dextag bimanual_dextag            layout     "${DEXTAG_ENVS:-1024}" "$DI" "$VD" "$CHAIN"
submit dextag bimanual_dextag_anatomical anatomical "${DEXTAG_ENVS:-1024}" "$DI" "$VD" "$CHAIN"
submit recon  bimanual_recon_layout      layout     "${RECON_ENVS:-2048}"  "$RI" "$VR" 1
submit recon  bimanual_recon             anatomical "${RECON_ENVS:-2048}"  "$RI" "$VR" 1

DEP=$(IFS=:; echo "${JOBS[*]}")
id=$(sbatch --parsable -A iyg@cpu --qos=qos_cpu-dev -t 00:20:00 -c 4 -J "${P}_compare" -o "logs/slurm/%x_%j.out" \
     --dependency=afterany:"$DEP" --wrap "cd $PWD && source launch/config.sh && export WANDB_MODE=offline WANDB_RUN_GROUP=$P && \
     \$PY_SIM scripts/policy/summarize_runs.py ${P}_dextag_layout ${P}_dextag_anatomical ${P}_recon_layout ${P}_recon_anatomical && \
     cd scripts/policy && \$PY_SIM wandb_report.py compare $P ${P}_dextag_layout ${P}_dextag_anatomical ${P}_recon_layout ${P}_recon_anatomical")
echo "comparison: job $id (after $DEP)"
