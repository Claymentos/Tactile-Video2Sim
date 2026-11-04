#!/bin/bash
# Several egotouch_train.slurm runs (training + evaluation) side by side in one job, one GPU each: the dev QoS allows
# 10 submitted jobs per user, so launch/submit_obj_ablation.sh packs an episode's rewards into one 4-GPU job.
# Each run logs to logs/slurm/<EXP>_<jobid>.out and works on its own copy of Genesis' kernel cache.
# Env: REF  GROUP (wandb group, EXP prefix)  RUNS ("NAME=EXTRA;NAME=EXTRA;...", NAME the wandb run name)
#      ITERATIONS N_ENVS SAVE_EVERY (egotouch_train.slurm's)
#SBATCH -J et_train_multi
#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -o pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
: "${REF:?}" "${GROUP:?}" "${RUNS:?}"
IFS=';' read -ra SPECS <<< "$RUNS"
pids=()
for i in "${!SPECS[@]}"; do
    name=${SPECS[$i]%%=*}; extra=${SPECS[$i]#*=}
    exp=${GROUP}_$name
    echo "GPU $i: $exp ($extra)"
    CUDA_VISIBLE_DEVICES=$i QD_OFFLINE_CACHE_FILE_PATH="${JOBSCRATCH:-/tmp/$SLURM_JOB_ID}/quadrants_$i" \
    WANDB_RUN_GROUP=$GROUP WANDB_RUN_NAME=$name EXP=$exp EXTRA="$extra" \
        bash launch/slurm/egotouch_train.slurm > "logs/slurm/${exp}_${SLURM_JOB_ID}.out" 2>&1 &
    pids+=($!)
    sleep 30          # staggered kernel compilation
done
code=0
for p in "${pids[@]}"; do wait "$p" || code=1; done
echo "all done (exit $code)"
exit $code
