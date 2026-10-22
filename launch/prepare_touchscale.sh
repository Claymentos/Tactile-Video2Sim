#!/usr/bin/env bash
# TouchScale -> extracted shards, Gemini scenes.
#
#   1  extract   with EXTRACT=1, the shard .tar files not extracted yet (data/train-XXXXX-of-00430/, ~2.5 GB each)
#   2  parse     Gemini on each episode's head video, with the dataset's task and items as context
#                -> data/touchscale/scene/<episode_id>.json (skipped if present; needs GEMINI_API_KEY)    [PY_GEMINI]
# TouchScale has no hand pose: no simulation reference is built from it.
#
# Usage:   launch/prepare_touchscale.sh [EPISODE_IDS]      ids or id prefixes, comma-separated
# Knobs:   EXTRACT=1  MAX_EPISODES=5 (when no ids)  MODEL=gemini-flash-lite-latest  GEMINI_FPS=2  SKIP_PARSE=1
#          OUT=data/touchscale
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
IDS="${1:-}"
OUT="${OUT:-$REPO/data/touchscale}"

if [ "${EXTRACT:-0}" = "1" ]; then
echo "=== 1/2 extract shards ($TOUCHSCALE_ROOT/data) ==="
for tar in "$TOUCHSCALE_ROOT"/data/train-*-of-*.tar; do
    [ -e "$tar" ] || continue
    dir="${tar%.tar}"
    if [ ! -d "$dir" ]; then echo "  $tar"; mkdir -p "$dir" && tar -xf "$tar" -C "$dir"; fi
done
fi

if [ "${SKIP_PARSE:-0}" != "1" ]; then
    echo "=== 2/2 Gemini scene parsing ==="
    [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set (SKIP_PARSE=1)"; exit 1; }
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

echo "scenes: $OUT/scene"
