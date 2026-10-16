#!/usr/bin/env python3
"""Extract HumanTouch episodes (LeRobot parquet) into one npz per episode (tv2s_data/humantouch.py).

Unity's left-handed y-up world is converted to a right-handed z-up one (see tv2s_data.humantouch). Needs pandas + pyarrow.
Writes <out>/episodes/episode_XXXXXX.npz.

    python scripts/data/extract_humantouch.py --root ~/HumanTouch/X001 --out data/humantouch --max-episodes 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.humantouch import episode_parquet, extract_episode, list_episodes  # noqa: E402


def parse_selection(spec: str) -> list[int]:
    """'1-20,35' -> [1, ..., 20, 35]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="one HumanTouch task folder (e.g. HumanTouch/X001)")
    ap.add_argument("--out", required=True, help="prepared data folder")
    ap.add_argument("--episodes", default=None, help="comma list / ranges of episode indices")
    ap.add_argument("--max-episodes", type=int, default=20, help="first N episodes when --episodes is not given")
    ap.add_argument("--no-mirror", action="store_true", help="keep Unity's left-handed y-up world")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)

    available = list_episodes(root)
    if args.episodes:
        wanted = parse_selection(args.episodes)
    else:
        wanted = [e for e in available if episode_parquet(root, e).exists()][: args.max_episodes]
    missing = [e for e in wanted if not episode_parquet(root, e).exists()]
    if missing:
        print(f"skipping {len(missing)} episodes without a parquet file: {missing[:10]}")
    for e in wanted:
        if e in missing:
            continue
        path = out / f"episode_{e:06d}.npz"
        if path.exists() and not args.overwrite:
            print(f"episode {e:6d}: exists")
            continue
        data = extract_episode(root, e, zup=not args.no_mirror)
        np.savez_compressed(path, **data)
        T = len(data["head_pose"])
        print(f"episode {e:6d}: {T} frames ({T / float(data['fps']):.1f} s) -> {path}")


if __name__ == "__main__":
    main()
