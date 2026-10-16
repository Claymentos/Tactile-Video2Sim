#!/usr/bin/env python3
"""Stage 2 (HumanTouch): turn the cuts into simulation clips (tv2s_data/clips.py: sim_clip).

Reads the cuts of stage 1 (<data>/cuts, scripts/data/cut_clips.py: window, scene description, glove) and,
per episode, the retargeted reference (<data>/retarget, retarget_humantouch.py) and the head pose
(<data>/episodes). Per cut: the reference Wuji hands on the window at the retargeting rate, the cut's
glove at those frames, and per grasping hand the proxy object (a cylinder inferred from the glove and the
hand: HumanTouch has no object pose); cuts where no hand grasps are skipped. Writes
<data>/clips/<name>.pkl (the format BimanualTactileEnv loads, same name as the cut) and
<data>/clips/index.json. Only the episodes processed are replaced (--clean empties clips/ first).

    python scripts/data/build_sim_clips.py --data data/humantouch [--episodes 2] [--set mass=0.2]
(the window, glove and contacts come from the cut: here ClipConfig sets the grasp and proxy parameters)
"""

import argparse
import dataclasses
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
from extract_humantouch import parse_selection  # noqa: E402
from tv2s_data.clips import clip_config, sim_clip, wuji_reference_geometry  # noqa: E402
from tv2s_data.glove import load_glove_layout  # noqa: E402
from tv2s_data.humantouch import SIDES  # noqa: E402
from tv2s_env.robots.wuji_model import WujiHand  # noqa: E402


def _agreement(hand: dict) -> str:
    glove, gemini = hand["object"] is not None, hand["semantic"]["object"] != "none"
    return {(True, True): "both", (True, False): "glove_only", (False, True): "gemini_only"}.get((glove, gemini), "none")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepared HumanTouch folder: cuts/, retarget/, episodes/")
    ap.add_argument("--episodes", default=None, help="comma list / ranges of episodes (default: every cut one)")
    ap.add_argument("--clean", action="store_true", help="remove every existing simulation clip first")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ClipConfig overrides")
    args = ap.parse_args()
    data = Path(args.data)
    cfg = clip_config(args.set)
    cut_index = json.loads((data / "cuts" / "index.json").read_text())
    if cut_index["dataset"] != "humantouch":
        raise SystemExit(f"{data / 'cuts'}: {cut_index['dataset']} cuts have no hand pose to simulate")
    by_episode = defaultdict(list)
    for c in cut_index["cuts"]:
        by_episode[int(c["episode"].split("_")[1])].append(c["name"])
    if args.episodes:
        wanted = set(parse_selection(args.episodes))
        by_episode = {e: n for e, n in by_episode.items() if e in wanted}
    out = data / "clips"
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in out.glob("*.pkl"):
            old.unlink()
    old_index = [] if args.clean or not (out / "index.json").exists() else json.loads((out / "index.json").read_text())["clips"]
    hands = {s: WujiHand(s) for s in SIDES}

    index, layouts = [], None
    for e, names in sorted(by_episode.items()):
        rpath = data / "retarget" / f"episode_{e:06d}.npz"
        if not rpath.exists():
            print(f"episode {e}: not retargeted ({rpath}), skipped")
            continue
        for old in out.glob(f"ep{e:06d}_*.pkl"):
            old.unlink()
        retarget = dict(np.load(rpath))
        head_pose = np.load(data / "episodes" / rpath.name)["head_pose"]
        geoms = {s: wuji_reference_geometry(hands[s], retarget[f"{s}_qpos"], retarget[f"{s}_palm_pos"],
                                            retarget[f"{s}_palm_quat"]) for s in SIDES}
        n = 0
        for name in names:
            cut = json.loads((data / "cuts" / f"{name}.json").read_text())
            m = cut["metadata"]
            if layouts is None:
                layouts = {s: load_glove_layout(m["tactile_mapping"], s, finger_tip_row=m["finger_tip_row"]) for s in SIDES}
            clip = sim_clip(cut, dict(np.load(data / "cuts" / f"{name}.npz")), retarget, geoms, head_pose, layouts, cfg)
            if clip is None:
                continue
            clip["metadata"]["clip_config"] = dataclasses.asdict(cfg)
            with open(out / f"{name}.pkl", "wb") as f:
                pickle.dump(clip, f)
            objs = {s: clip["hands"][s]["object"] for s in SIDES}
            scene = clip["metadata"]["scene"]
            index.append(dict(
                name=name, episode=e, frames=int(clip["metadata"]["end"] - clip["metadata"]["start"]),
                types=clip["metadata"]["types"], segments=m["segments"],
                caption=clip["metadata"]["captions"][0] if scene else None,
                objects_manipulated=None if scene is None else {s: scene[f"{s}_hand"]["object"] for s in SIDES},
                # per hand: does the glove grasp agree with Gemini's manipulated object?
                glove_vs_gemini=None if scene is None else {s: _agreement(clip["hands"][s]) for s in SIDES},
                objects={s: None if o is None else dict(radius=round(o["radius"], 4), length=round(o["length"], 4),
                                                         grasp=list(o["grasp_interval"])) for s, o in objs.items()},
            ))
            n += 1
        print(f"episode_{e:06d}: {n} simulation clips from {len(names)} cuts", flush=True)

    done = set(by_episode)
    index = sorted([c for c in old_index if c["episode"] not in done] + index, key=lambda c: c["name"])
    with open(out / "index.json", "w") as f:
        json.dump(dict(clip_config=dataclasses.asdict(cfg), clips=index), f, indent=1)
    n_obj = {s: sum(c["objects"][s] is not None for c in index) for s in SIDES}
    frames = sum(c["frames"] for c in index)
    print(f"{len(index)} clips, {frames} frames; objects: {n_obj['left']} left, {n_obj['right']} right -> {out}")


if __name__ == "__main__":
    main()
