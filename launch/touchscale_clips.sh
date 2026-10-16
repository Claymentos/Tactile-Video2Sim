#!/usr/bin/env bash
# TouchScale -> Gemini scene parsing -> clips with the glove contact and a video per clip.
#
#   1  extract   with EXTRACT=1, the shard .tar files not extracted yet (data/train-XXXXX-of-00430/, ~2.5 GB each)
#   2  parse     Gemini on each episode's head video, with the dataset's task and items as context
#                -> data/touchscale/scene/<episode_id>.json (skipped if present; needs GEMINI_API_KEY) [PY_GEMINI]
#   3  cut       the scene's clips (else the task segment), glove per hand, one mp4 per clip
#                -> data/touchscale/cuts/{<id>_cKK.json, .npz, videos/<id>_cKK.mp4, index.json}   [PY_SIM]
# TouchScale has no hand pose: these cuts are for inspection / perception, not simulation clips.
#
# Usage:   launch/touchscale_clips.sh [EPISODE_IDS]      ids or id prefixes, comma-separated
# Knobs:   EXTRACT=1  MAX_EPISODES=5 (when no ids)  MODEL=gemini-flash-lite-latest  GEMINI_FPS=2  SKIP_PARSE=1
#          OUT=data/touchscale  PAD_S=0.5
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
IDS="${1:-}"
OUT="${OUT:-$REPO/data/touchscale}"
SEL=(--max-episodes "${MAX_EPISODES:-5}"); [ -n "$IDS" ] && SEL=(--episodes "$IDS")

if [ "${EXTRACT:-0}" = "1" ]; then
echo "=== 1/3 extract shards ($TOUCHSCALE_ROOT/data) ==="
for tar in "$TOUCHSCALE_ROOT"/data/train-*-of-*.tar; do
    [ -e "$tar" ] || continue
    dir="${tar%.tar}"
    if [ ! -d "$dir" ]; then echo "  $tar"; mkdir -p "$dir" && tar -xf "$tar" -C "$dir"; fi
done
fi

if [ "${SKIP_PARSE:-0}" != "1" ]; then
    echo "=== 2/3 Gemini scene parsing ==="
    [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set (SKIP_PARSE=1 to cut the task segments only)"; exit 1; }
    "$PY_SIM" -c "
import sys; sys.path.insert(0, 'src')
from tv2s_data.touchscale import select_episodes, context
for e in select_episodes('$TOUCHSCALE_ROOT', '$IDS' or None, None if '$IDS' else ${MAX_EPISODES:-5}):
    print(e.id, e.video, context(e), sep='\t')" | while IFS=$'\t' read -r id video ctx; do
        out="$OUT/scene/$id.json"
        if [ -e "$out" ]; then echo "  ${id:0:8}: exists"; continue; fi
        "$PY_GEMINI" scripts/reconstruction/semantic_parsing.py --video "$video" --out "$out" \
            --model "${MODEL:-gemini-flash-lite-latest}" --fps "${GEMINI_FPS:-2}" --context "$ctx"
    done
fi

echo "=== 3/3 cuts + videos ==="
"$PY_SIM" scripts/data/cut_clips.py --dataset touchscale --root "$TOUCHSCALE_ROOT" --data "$OUT" "${SEL[@]}" \
    --set pad_s="${PAD_S:-0.5}"
