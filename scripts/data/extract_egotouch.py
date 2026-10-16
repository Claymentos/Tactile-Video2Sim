#!/usr/bin/env python3
"""Extract EgoTouch episodes (scripts/data/download_egotouch.py) into one npz per episode (tv2s_data/egotouch.py).

Rokoko's and Vive's Unity-style worlds are converted to right-handed z-up ones; stalled / absent samples are
flagged. Writes <out>/episodes/<key>.npz (key = <task>__<recording time>) with the arrays of
tv2s_data.egotouch.load_episode plus fps, key, task, scene, and the glove layout
assets/tactile/egotouch_glove.json. Prints each stream's share of valid frames; --min-valid skips the
episodes whose hands are valid less often than that (a stalled Rokoko stream).

    python scripts/data/extract_egotouch.py --root ~/EgoTouch --out data/egotouch [--episodes pick_up_bottle]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.egotouch import FPS, SIDES, check_layout, load_episode, quality, select_episodes, write_layout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path.home() / "EgoTouch"), help="the downloaded dataset")
    ap.add_argument("--out", required=True, help="prepared data folder")
    ap.add_argument("--episodes", default=None, help="comma list of keys, tasks or recording times (prefixes)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--min-valid", type=float, default=0.5, help="least share of valid frames of each hand")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    print(f"glove layout -> {write_layout()}")
    eps = select_episodes(args.root, args.episodes, args.max_episodes)
    if not eps:
        raise SystemExit(f"no EgoTouch episode under {args.root} matches {args.episodes!r}")
    for ep in eps:
        path = out / f"{ep.key}.npz"
        if path.exists() and not args.overwrite:
            print(f"{ep.key}: exists")
            continue
        data = load_episode(ep)
        for s in SIDES:
            check_layout(data[f"{s}_tactile"].reshape(len(data["time_s"]), 21, 21), s)
        q = quality(data)
        msg = ", ".join(f"{k} {v:.0%}" for k, v in q.items())
        if min(q[f"{s}_skeleton"] for s in SIDES) < args.min_valid:
            print(f"{ep.key}: skipped, hands valid too rarely ({msg})")
            continue
        np.savez_compressed(path, **data, fps=np.array(FPS), key=np.array(ep.key), task=np.array(ep.task),
                            scene=np.array(ep.scene), relative=np.array(ep.relative), world=np.array("zup"))
        T = len(data["time_s"])
        print(f"{ep.key}: {T} frames ({T / FPS:.1f} s), valid: {msg} -> {path}")


if __name__ == "__main__":
    main()
