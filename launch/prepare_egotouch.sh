#!/usr/bin/env bash
# EgoTouch (huggingface: zhouzhoujy/EgoTouch) -> episodes, retargeted hands, Gemini scenes, cuts, tactile mapping.
#
#   1  download  with DOWNLOAD=1, the selected tasks' episodes (TASKS, PER_TASK) into EGOTOUCH_ROOT      [PY_SIM]
#   2  extract   episodes -> npz (Rokoko / Vive Unity worlds -> right-handed z-up, stalled streams flagged)
#                + the glove layout assets/tactile/egotouch_glove.json                                  [PY_SIM]
#   3  retarget  both Rokoko skeletons -> Wuji Hand joints + palm pose, at HZ                         [PY_RETARGET]
#   4  parse     Gemini (or, VLM=qwen, Qwen-VL run locally) on each episode's chest video, with the task as
#                context -> data/egotouch/scene/<key>.json (skipped if present; Gemini needs GEMINI_API_KEY)
#                                                                                          [PY_GEMINI / PY_QWEN]
#   5  cut       one cut per episode (the scene's clips if parsed), glove per hand, one mp4 per cut   [PY_SIM]
#   6  mapping   glove taxels -> MANO -> Wuji: assets/tactile/mano_egotouch_{left,right}.npz           [PY_SIM]
# Then, per episode: launch/reconstruct_objects.sh egotouch <key>, and scripts/data/build_episode_clip.py.
#
# Usage:   launch/prepare_egotouch.sh [EPISODES]      keys, tasks or recording times, comma-separated (default: all downloaded)
# Knobs:   DOWNLOAD=1 TASKS=pick_up_bottle,twist_bottle_cap PER_TASK=5   HZ=30  MODEL=gemini-3.6-flash  GEMINI_FPS=4
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
    echo "=== 1/6 download ($EGOTOUCH_ROOT) ==="
    "$PY_SIM" scripts/data/download_egotouch.py --root "$EGOTOUCH_ROOT" --tasks "${TASKS:?TASKS=task,task}" --per-task "${PER_TASK:-5}"
fi
echo "=== 2/6 extract ==="
"$PY_SIM" scripts/data/extract_egotouch.py --root "$EGOTOUCH_ROOT" --out "$OUT" "${SEL[@]}" "${OW[@]}"
echo "=== 3/6 retarget both hands onto the Wuji Hand ==="
"$PY_RETARGET" scripts/data/retarget_humantouch.py --data "$OUT" --hz "${HZ:-30}" "${OW[@]}"
if [ "${SKIP_PARSE:-0}" != "1" ]; then
    if [ "${VLM:-gemini}" = "qwen" ]; then
        echo "=== 4/6 Qwen-VL scene parsing (local) ==="
        PARSE=("$PY_QWEN" scripts/reconstruction/semantic_parsing.py --backend qwen ${MODEL:+--model "$MODEL"})
    else
        echo "=== 4/6 Gemini scene parsing ==="
        [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set (VLM=qwen: a local model; SKIP_PARSE=1: whole episodes only)"; exit 1; }
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
echo "=== 5/6 cuts + videos ==="
"$PY_SIM" scripts/data/cut_clips.py --dataset egotouch --root "$EGOTOUCH_ROOT" --data "$OUT" "${SEL[@]}" 2>&1 | grep -v "^profiling"
echo "=== 6/6 glove -> Wuji tactile mapping ==="
"$PY_SIM" scripts/tactile/build_tactile_mappings.py --dataset egotouch 2>&1 | grep -v "^profiling\|RuntimeWarning\|barycentric\|inverse_denominator"
echo "episodes: $OUT/episodes   retarget: $OUT/retarget   scenes: $OUT/scene   cuts: $OUT/cuts"
echo "next: launch/reconstruct_objects.sh egotouch <key>, then scripts/data/build_episode_clip.py --hands mocap --data $OUT"
