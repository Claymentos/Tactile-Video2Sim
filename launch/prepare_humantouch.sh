#!/usr/bin/env bash
# HumanTouch -> bimanual training clips, then the glove -> Wuji tactile mappings.
#
#   1  extract   parquet episodes -> npz (world mirrored to right-handed)           [PY_DATA]
#   2  retarget  both skeletons -> Wuji Hand joints + palm pose, at HZ             [PY_RETARGET]
#   3  clips     manipulation cycles, glove targets, tactile-inferred proxy objects [PY_SIM]
#   4  mappings  glove cells on both Wuji hands: layout (DexTaG), anatomical (BG)   [PY_SIM]
#
# Usage:   launch/prepare_humantouch.sh [EPISODES]
#          EPISODES: comma list / ranges (1-20,35) or a count via MAX_EPISODES=20 (default: 20 episodes)
# Knobs:   HZ=30  FINGER_TIP_ROW=22  CLIP_ARGS="--set contact_threshold=8"  OVERWRITE=1
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"
EPISODES="${1:-}"
SEL=(--max-episodes "${MAX_EPISODES:-20}")
[ -n "$EPISODES" ] && SEL=(--episodes "$EPISODES")
OW=(); [ "${OVERWRITE:-0}" = "1" ] && OW=(--overwrite)

echo "=== 1/4 extract HumanTouch episodes ($HUMANTOUCH_ROOT) ==="
"$PY_DATA" scripts/data/extract_humantouch.py --root "$HUMANTOUCH_ROOT" --out "$DATA_DIR" "${SEL[@]}" "${OW[@]}"
echo "=== 2/4 retarget both hands onto the Wuji Hand ==="
"$PY_RETARGET" scripts/data/retarget_humantouch.py --data "$DATA_DIR" --hz "${HZ:-30}" "${OW[@]}"
echo "=== 3/4 build bimanual clips + proxy objects ==="
# shellcheck disable=SC2086
"$PY_SIM" scripts/data/build_clips.py --data "$DATA_DIR" --dataset-root "$HUMANTOUCH_ROOT" \
    --finger-tip-row "${FINGER_TIP_ROW:-22}" ${CLIP_ARGS:-} 2>&1 | grep -v "^profiling"
echo "=== 4/4 glove -> Wuji tactile mappings ==="
"$PY_SIM" scripts/tactile/build_tactile_mappings.py --data "$DATA_DIR" --dataset-root "$HUMANTOUCH_ROOT" \
    --finger-tip-row "${FINGER_TIP_ROW:-22}" 2>&1 | grep -v "^profiling\|RuntimeWarning\|barycentric\|inverse_denominator"
echo "clips: $DATA_DIR/clips   mappings: $REPO/assets/tactile"
