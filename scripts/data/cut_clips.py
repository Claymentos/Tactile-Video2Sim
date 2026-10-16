#!/usr/bin/env python3
"""Stage 1: cut episodes into clips (tv2s_data/cuts.py) -- windows, scene description, glove, video.

Per episode, the clips are the Gemini scene parsing's clips when <data>/scene/<key>.json exists
(scripts/reconstruction/semantic_parsing.py), else the dataset's segmentation (--segments forces one).
Needs only the episode's video and glove: no hand pose, no retargeting. Writes <data>/cuts/:
  <name>.json         metadata (window in video frames and seconds, caption, scene description, per hand
                      the object Gemini names and the glove grasp)
  <name>.npz          the glove at every frame of the cut: <side>_<signal> (dataset-specific, see
                      tv2s_data/cuts.py) and <side>_contact
  videos/<name>.mp4   the head camera with the caption, the objects per hand and the glove grasp
  index.json
Only the episodes processed are replaced (--clean empties cuts/ first).
Stage 2 (HumanTouch: scripts/data/build_sim_clips.py) adds the retargeted hands and the objects.

    python scripts/data/cut_clips.py --dataset humantouch --data data/humantouch --root ~/HumanTouch/X001 [--episodes 2]
    python scripts/data/cut_clips.py --dataset touchscale --data data/touchscale --root ~/TouchScale [--episodes 5a1ae97d]
    python scripts/data/cut_clips.py --dataset egotouch --data data/egotouch --root ~/EgoTouch [--episodes pick_up_bottle]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
from extract_humantouch import parse_selection  # noqa: E402
from tv2s_data.clip_video import write_clip_video  # noqa: E402
from tv2s_data.clips import clip_config  # noqa: E402
from tv2s_data.cuts import cut_episode, egotouch_sources, humantouch_sources, touchscale_sources, video_view  # noqa: E402
from tv2s_data.humantouch import SIDES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("humantouch", "touchscale", "egotouch"), required=True)
    ap.add_argument("--data", required=True, help="prepared folder: scene/ is read, cuts/ written "
                                                  "(HumanTouch / EgoTouch: episodes/ from extract_*.py)")
    ap.add_argument("--root", required=True, help="the dataset folder (videos, glove mapping / shards)")
    ap.add_argument("--episodes", default=None, help="HumanTouch: indices / ranges (2, 1-20); "
                                                     "TouchScale: episode ids or id prefixes; "
                                                     "EgoTouch: keys, tasks or recording times")
    ap.add_argument("--max-episodes", type=int, default=None, help="TouchScale / EgoTouch, without --episodes")
    ap.add_argument("--segments", choices=("auto", "scene", "dataset"), default="auto",
                    help="auto: the scene parsing when the episode has one, else the dataset's segmentation")
    ap.add_argument("--finger-tip-row", type=int, default=22)
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--video-height", type=int, default=360)
    ap.add_argument("--clean", action="store_true", help="remove every existing cut first")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ClipConfig overrides (pad_s, ...)")
    args = ap.parse_args()
    data, root = Path(args.data), Path(args.root).expanduser()
    cfg = clip_config(args.set)
    if args.dataset == "humantouch":
        sources = humantouch_sources(data, root, set(parse_selection(args.episodes)) if args.episodes else None,
                                     cfg, args.finger_tip_row)
    elif args.dataset == "egotouch":
        sources = egotouch_sources(data, root, args.episodes, cfg, args.max_episodes)
    else:
        sources = touchscale_sources(root, args.episodes, args.max_episodes)
    out = data / "cuts"
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in [*out.glob("*.json"), *out.glob("*.npz"), *out.glob("videos/*.mp4")]:
            old.unlink()
    old_index = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else None
    if old_index is not None and old_index["dataset"] != args.dataset:
        raise SystemExit(f"{out} holds {old_index['dataset']} cuts: use another --data")

    index, done = [], set()
    for src in sources:
        done.add(src.key)
        for old in [*out.glob(f"{src.prefix}_c*.json"), *out.glob(f"{src.prefix}_c*.npz"),
                    *out.glob(f"videos/{src.prefix}_c*.mp4")]:
            old.unlink()
        scene_path = data / "scene" / f"{src.key}.json"
        scene = None
        if args.segments != "dataset" and scene_path.exists():
            scene = json.loads(scene_path.read_text())
        elif args.segments == "scene":
            print(f"{src.key}: no {scene_path}, the dataset's segmentation")
        videos = not args.no_videos and src.video.exists()
        if not args.no_videos and not videos:
            print(f"{src.key}: no video {src.video}")
        cuts = cut_episode(src, scene, cfg)
        for cut, arrays in cuts:
            m = cut["metadata"]
            (out / f"{m['name']}.json").write_text(json.dumps(cut, ensure_ascii=False))
            np.savez_compressed(out / f"{m['name']}.npz", **arrays)
            if videos:
                write_clip_video(video_view(cut), src.video, out / "videos" / f"{m['name']}.mp4",
                                 args.video_height, src.fps)
            index.append(dict(
                name=m["name"], dataset=src.dataset, episode=src.key, start_s=m["start_s"], end_s=m["end_s"],
                frames=m["end"] - m["start"], segments=m["segments"],
                caption=m["captions"][0] if scene is not None or src.dataset != "humantouch" else " + ".join(m["types"]),
                objects_manipulated=None if scene is None else {s: cut["hands"][s]["semantic"]["object"] for s in SIDES},
                grasp={s: cut["hands"][s]["grasp_interval"] for s in SIDES},
            ))
        print(f"{src.key}: {len(cuts)} cuts ({'scene parsing' if scene is not None else 'dataset segmentation'})",
              flush=True)

    kept = [c for c in (old_index or {}).get("cuts", []) if c["episode"] not in done]
    (out / "index.json").write_text(json.dumps(dict(dataset=args.dataset, clip_config=dataclasses.asdict(cfg),
                                                    cuts=sorted(kept + index, key=lambda c: c["name"])),
                                               indent=1, ensure_ascii=False))
    print(f"{len(index)} cuts from {len(done)} episodes ({len(kept)} other cuts kept) -> {out}")


if __name__ == "__main__":
    main()
