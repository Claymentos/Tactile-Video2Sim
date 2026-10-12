#!/usr/bin/env bash
# Compute nodes have no internet: jobs log wandb offline under logs/. Run this on a login node to upload them.
# Every PERIOD seconds (default 600), each offline run not yet complete is synced with a time-boxed
# `wandb beta sync --live` (SLICE seconds, default 180), one at a time: Jean Zay login nodes allow 500 tasks
# per user and every uploader is a multi-threaded process. A sync that ends before its time box means the run
# is complete: it is marked (<run>.synced_final) and skipped from then on.
#   nohup launch/wandb_sync_loop.sh > logs/wandb_sync.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
WANDB="$(dirname "$PY_SIM")/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-tactile-video2sim}" GOMAXPROCS=2
while true; do
    mapfile -t RUNS < <(find "$REPO/logs" "$REPO/outputs" -type d -name "offline-run-*" 2>/dev/null | sort)
    for run in "${RUNS[@]}"; do
        [ -e "$run.synced_final" ] && continue
        timeout "${SLICE:-180}" "$WANDB" beta sync --live -p "$WANDB_PROJECT" "$run" > "$run.sync.log" 2>&1
        code=$?
        if [ $code = 0 ]; then
            touch "$run.synced_final"; echo "$(date '+%F %T') complete: $run"
        elif [ $code = 124 ]; then
            echo "$(date '+%F %T') running, synced so far: $run"
        else
            echo "$(date '+%F %T') sync error ($code): $run"; tail -3 "$run.sync.log"
        fi
    done
    [ "${ONCE:-0}" = 1 ] && break
    sleep "${PERIOD:-600}"
done
