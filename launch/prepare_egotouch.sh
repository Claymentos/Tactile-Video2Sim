#!/usr/bin/env bash
# EgoTouch (huggingface: zhouzhoujy/EgoTouch) -> episodes checked, Gemini scenes.
#
#   1  download  with DOWNLOAD=1, the selected tasks' episodes (TASKS, PER_TASK) into EGOTOUCH_ROOT      [PY_SIM]
#   2  extract   episodes -> npz (Rokoko / Vive Unity worlds -> right-handed z-up, stalled streams flagged): the
#                episodes whose hands are valid too rarely are reported and skipped                       [PY_SIM]
#   3  parse     Gemini (or, VLM=qwen, Qwen-VL run locally) on each episode's chest video, with the task as
#                context -> data/egotouch/scene/<key>.json: the hands used and the objects (skipped if present;
#                Gemini needs GEMINI_API_KEY)                                              [PY_GEMINI / PY_QWEN]
# Then, per one-hand episode: launch/rl_recon.sh <name> <key> (reconstruction, reference, RL).
#
# Usage:   launch/prepare_egotouch.sh [EPISODES]      keys, tasks or recording times, comma-separated (default: all downloaded)
# Knobs:   DOWNLOAD=1 TASKS=pick_up_bottle,twist_bottle_cap PER_TASK=5   MODEL=gemini-3.6-flash  GEMINI_FPS=4
#          VLM=qwen (local Qwen-VL instead of Gemini; MODEL: a Hugging Face Qwen-VL, default Qwen/Qwen3-VL-4B-Instruct)
#          SKIP_PARSE=1  OVERWRITE=1  OUT=data/egotouch
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
EPISODES="${1:-}"
OUT="${OUT:-$REPO/data/egotouch}"
SEL=(); [ -n "$EPISODES" ] && SEL=(--episodes "$EPISODES")
OW=(); [ "${OVERWRITE:-0}" = "1" ] && OW=(--overwrite)

if [ "${DOWNLOAD:-0}" = "1" ]; then
    echo "=== 1/3 download ($EGOTOUCH_ROOT) ==="
    "$PY_SIM" scripts/data/download_egotouch.py --root "$EGOTOUCH_ROOT" --tasks "${TASKS:?TASKS=task,task}" --per-task "${PER_TASK:-5}"
fi
echo "=== 2/3 extract ==="
"$PY_SIM" scripts/data/extract_egotouch.py --root "$EGOTOUCH_ROOT" --out "$OUT" "${SEL[@]}" "${OW[@]}"
if [ "${SKIP_PARSE:-0}" != "1" ]; then
    if [ "${VLM:-gemini}" = "qwen" ]; then
        echo "=== 3/3 Qwen-VL scene parsing (local) ==="
        PARSE=("$PY_QWEN" scripts/reconstruction/semantic_parsing.py --backend qwen ${MODEL:+--model "$MODEL"})
    else
        echo "=== 3/3 Gemini scene parsing ==="
        [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set (VLM=qwen: a local model; SKIP_PARSE=1)"; exit 1; }
        PARSE=("$PY_GEMINI" scripts/reconstruction/semantic_parsing.py --model "${MODEL:-gemini-3.6-flash}")
    fi
    "$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from tv2s_data.egotouch import select_episodes, context
for e in select_episodes('$EGOTOUCH_ROOT', '$EPISODES' or None):
    if (Path('$OUT') / 'episodes' / f'{e.key}.npz').exists():
        print(e.key, e.video, context(e), sep='\t')" | while IFS=$'\t' read -r key video ctx; do
        out="$OUT/scene/$key.json"
        if [ -e "$out" ]; then echo "  $key: exists"; continue; fi
        "${PARSE[@]}" --video "$video" --out "$out" --fps "${GEMINI_FPS:-4}" --context "$ctx" < /dev/null \
            || echo "  $key: parsing failed"
    done
fi
echo "episodes: $OUT/episodes   scenes: $OUT/scene   next: launch/rl_recon.sh <name> <key>"
