#!/usr/bin/env python3
"""The JQ glove's 256 raw cells -> EgoTouch's 21x21 hand grid, learned on EgoTouch (assets/tactile/jq_glove_map.json).

EgoTouch records the JQ gloves twice: raw (jq_pressure.json, 256 8-bit cells per hand) and laid out on the 21x21 grid
whose cells tv2s_data.egotouch.glove_cells places on the hand (pressure_grids.npz, baseline-corrected, normalised).
Each grid cell is one raw cell (the palm's 19 grid columns repeat some of the 15 raw ones, a bend strip repeats one
sensor): per episode, the raw cell the grid cell correlates with (> 0.99) is a vote, and the majority is kept. Any
dataset recorded with the same gloves (ACE-Data-0's Tactile shards: 256 cells and an IMU quaternion per hand) is then
laid out on the grid, and from there on MANO and the Wuji taxels, without a layout of its own.

Run in the `genesis` env (EgoTouch downloaded under EGOTOUCH_ROOT):
    python scripts/data/fit_jq_glove_map.py --root ~/EgoTouch
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.egotouch import GRID, SIDES, glove_cells, list_episodes, load_episode  # noqa: E402

OUT = REPO / "assets" / "tactile" / "jq_glove_map.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/EgoTouch", help="EgoTouch download")
    ap.add_argument("--min-corr", type=float, default=0.99, help="a vote: the grid cell is this raw cell")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = [load_episode(e) for e in list_episodes(args.root)]
    if not data:
        raise SystemExit(f"no EgoTouch episode under {args.root}")
    spec = dict(source=f"EgoTouch, {len(data)} episodes", grid_shape=list(GRID), n_raw=256, cells={}, votes={})
    for s in SIDES:
        cells, bend = glove_cells(s)
        channels = sorted({c[0] for c in cells} | {b[1] for b in bend})
        votes = collections.defaultdict(collections.Counter)
        for d in data:
            grid, raw = d[f"{s}_tactile"], d[f"{s}_raw"]
            active = raw.std(0) > 0
            for ch in channels:
                y = grid[:, ch]
                if (y > 0.05).sum() < 5:                       # the cell must fire for its correlation to mean anything
                    continue
                corr = np.zeros(256)
                corr[active] = [np.corrcoef(raw[:, k], y)[0, 1] for k in np.flatnonzero(active)]
                k = int(np.nan_to_num(corr).argmax())
                if corr[k] > args.min_corr:
                    votes[ch][k] += 1
        mapping = {ch: v.most_common(1)[0][0] for ch, v in votes.items()}
        split = sum(len(v) > 1 for v in votes.values())
        print(f"{s}: {len(mapping)}/{len(channels)} grid cells mapped ({split} with split votes), "
              f"{len(set(mapping.values()))} distinct raw cells; unmapped (never fired): "
              f"{sorted(set(channels) - set(mapping))}")
        spec["cells"][s] = {str(ch): k for ch, k in sorted(mapping.items())}
        spec["votes"][s] = {str(ch): dict(v.most_common()) for ch, v in sorted(votes.items())}
    Path(args.out).write_text(json.dumps(spec, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
