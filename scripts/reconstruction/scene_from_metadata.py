#!/usr/bin/env python3
"""A scene file from the dataset's own metadata, when Gemini is not available (semantic_parsing.py's format).

TouchScale lists, per episode, the task and its items: each item becomes an object (its text is the SAM3
prompt), the support surface ('table') is added, and the task segment is the only clip. What Gemini
would add -- the time an object is unoccluded, material, shape, size, and per hand the object manipulated
-- is left unknown: visible_at_s is the start of the video (SAM3 also tries the clip's start and end),
the hands' objects are 'none'. Items that are supports (mat, tray, pad...) are marked static, like the
table. Replace the file with semantic_parsing.py's output when a Gemini key is available.

    python scripts/reconstruction/scene_from_metadata.py --dataset touchscale --root ~/TouchScale --episode 29029fe2 --data data/touchscale
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

STATIC_WORDS = ("mat", "tray", "pad", "table", "counter", "desk", "board", "rack", "block", "drawer")
NONE = {"action": "none", "object": "none", "from": "none", "to": "none"}


def slug(text: str, taken: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "object"
    name, k = base, 2
    while name in taken:
        name, k = f"{base}_{k}", k + 1
    taken.add(name)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("touchscale",), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--episode", required=True, help="episode id or id prefix")
    ap.add_argument("--data", required=True, help="prepared folder: writes scene/<episode_id>.json")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing (e.g. Gemini) scene file")
    args = ap.parse_args()
    from tv2s_data import touchscale as ts

    eps = ts.select_episodes(args.root, args.episode)
    if len(eps) != 1:
        raise SystemExit(f"{len(eps)} episodes match {args.episode!r}")
    ep = eps[0]
    out = Path(args.data) / "scene" / f"{ep.id}.json"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists (--overwrite to replace it)")
    taken: set = set()
    objects = []
    for item in dict.fromkeys(["table top"] + list(ep.meta.get("items", []))):
        if "water" in item.lower():                                   # liquids: no mesh to reconstruct
            continue
        static = any(w in item.lower().split() for w in STATIC_WORDS)
        objects.append(dict(id=slug(item.replace("table top", "table"), taken), description=item, visible_at_s=0.0,
                            material="unknown", shape="irregular", size_cm=[0.0, 0.0, 0.0], is_container=False,
                            is_static=static, initial_location="unknown"))
    s, e = ts.task_window(ep)
    ids = [o["id"] for o in objects]
    scene = dict(video=str(ep.video), source="dataset metadata (no Gemini)", context=ts.context(ep),
                 scene=ep.meta["task_name"], objects=objects,
                 clips=[dict(start_s=s / ts.FPS, end_s=e / ts.FPS, caption=ep.meta["task_name"], left_hand=dict(NONE),
                             right_hand=dict(NONE), objects_present=ids, objects_manipulated=[], objects_used=[])])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, indent=2, ensure_ascii=False))
    print(f"{ep.short_id}: {len(objects)} objects ({', '.join(f'{o['id']}' + (' [static]' if o['is_static'] else '') for o in objects)}) -> {out}")


if __name__ == "__main__":
    main()
