#!/usr/bin/env python3
"""Cut the retargeted HumanTouch episodes into bimanual clips with proxy objects (tv2s_data/clips.py).

Reads <data>/episodes (extract_humantouch.py) and <data>/retarget (retarget_humantouch.py), writes
<data>/clips/<name>.pkl (one dict per clip, the format BimanualTactileEnv loads) and <data>/clips/index.json.
Clip names are ep<episode>_c<k>: the train/eval split hashes the part before '_' (the episode).

    python scripts/data/build_clips.py --data data/humantouch --dataset-root ~/HumanTouch/X001 [--set contact_threshold=8]
"""

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.clips import ClipConfig, build_episode_clips  # noqa: E402
from tv2s_data.glove import load_glove_layout  # noqa: E402
from tv2s_data.humantouch import SIDES, dataset_paths  # noqa: E402
from tv2s_env.robots.wuji_model import WujiHand  # noqa: E402


def clip_config(overrides: list[str]) -> ClipConfig:
    cfg = ClipConfig()
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    upd = {}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown ClipConfig field {key}: {sorted(fields)}")
        current = getattr(cfg, key)
        if isinstance(current, tuple):
            parts = [p for p in value.strip("()[]").split(",") if p]
            upd[key] = tuple(type(current[0])(p) if current else p for p in parts)
        else:
            upd[key] = type(current)(value)
    return dataclasses.replace(cfg, **upd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--finger-tip-row", type=int, default=22)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ClipConfig overrides")
    args = ap.parse_args()
    data = Path(args.data)
    cfg = clip_config(args.set)
    mapping_json = dataset_paths(Path(args.dataset_root).expanduser())["tactile_mapping"].resolve()
    layouts = {s: load_glove_layout(mapping_json, s, finger_tip_row=args.finger_tip_row) for s in SIDES}
    hands = {s: WujiHand(s) for s in SIDES}
    out = data / "clips"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pkl"):
        old.unlink()

    index = []
    retargets = sorted((data / "retarget").glob("episode_*.npz"))
    if not retargets:
        raise SystemExit(f"no retargeted episodes in {data / 'retarget'}")
    for rpath in retargets:
        episode = dict(np.load(data / "episodes" / rpath.name))
        retarget = dict(np.load(rpath))
        clips = build_episode_clips(episode, retarget, layouts, hands, cfg)
        e = int(episode["episode_index"])
        for clip in clips:
            name = f"ep{e:06d}_c{clip['metadata']['clip']:02d}"
            clip["metadata"].update(name=name, tactile_mapping=str(mapping_json), finger_tip_row=args.finger_tip_row,
                                    clip_config=dataclasses.asdict(cfg))
            with open(out / f"{name}.pkl", "wb") as f:
                pickle.dump(clip, f)
            objs = {s: clip["hands"][s]["object"] for s in SIDES}
            index.append(dict(
                name=name, episode=e, frames=int(clip["metadata"]["end"] - clip["metadata"]["start"]),
                types=clip["metadata"]["types"],
                objects={s: None if o is None else dict(radius=round(o["radius"], 4), length=round(o["length"], 4),
                                                         grasp=list(o["grasp_interval"])) for s, o in objs.items()},
            ))
        print(f"{rpath.stem}: {len(clips)} clips", flush=True)

    with open(out / "index.json", "w") as f:
        json.dump(dict(clip_config=dataclasses.asdict(cfg), clips=index), f, indent=1)
    n_obj = {s: sum(c["objects"][s] is not None for c in index) for s in SIDES}
    frames = sum(c["frames"] for c in index)
    print(f"{len(index)} clips, {frames} frames; objects: {n_obj['left']} left, {n_obj['right']} right -> {out}")


if __name__ == "__main__":
    main()
