#!/usr/bin/env bash
# HumanTouch -> episodes: the parquet episodes extracted to npz (Unity's y-up world -> right-handed z-up)  [PY_DATA]
#
# Usage:   launch/prepare_humantouch.sh [EPISODES]
#          EPISODES: comma list / ranges (1-20,35) or a count via MAX_EPISODES=20 (default: 20 episodes)
# Knobs:   OVERWRITE=1
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
EPISODES="${1:-}"
SEL=(--max-episodes "${MAX_EPISODES:-20}")
[ -n "$EPISODES" ] && SEL=(--episodes "$EPISODES")
OW=(); [ "${OVERWRITE:-0}" = "1" ] && OW=(--overwrite)

echo "=== extract HumanTouch episodes ($HUMANTOUCH_ROOT) ==="
"$PY_DATA" scripts/data/extract_humantouch.py --root "$HUMANTOUCH_ROOT" --out "$DATA_DIR" "${SEL[@]}" "${OW[@]}"
echo "episodes: $DATA_DIR/episodes"
