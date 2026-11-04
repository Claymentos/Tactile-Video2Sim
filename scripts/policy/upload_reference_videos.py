#!/usr/bin/env python3
"""Reference videos (scripts/policy/visualize_episode.py --reference, launch/slurm/render_references.slurm) uploaded to
wandb, one run per video, from a login node: the video as reference/video and its summary json as the run's summary.

    python scripts/policy/upload_reference_videos.py --group obj1 \
        outputs/episodes/obj1_references/et_pick_up_hammer_20260321_095958_211_rob1_rt07_reference.mp4=egotouch_pick_up_hammer_reference
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="VIDEO.mp4=WANDB_RUN_NAME")
    ap.add_argument("--group", default=None)
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "tactile-video2sim"))
    args = ap.parse_args()
    import wandb

    for item in args.videos:
        path, name = item.split("=", 1)
        path = Path(path)
        summary = path.with_suffix(".json")
        info = json.loads(summary.read_text()) if summary.exists() else {}
        w = wandb.init(project=args.project, name=name, group=args.group, job_type="reference",
                       tags=["reference"], config=dict(video=str(path), **info), reinit="finish_previous")
        w.log({"reference/video": wandb.Video(str(path), format="mp4")})
        w.summary.update({k: v for k, v in info.items() if isinstance(v, (int, float))})
        print(f"{name}: {w.url}")
        w.finish()


if __name__ == "__main__":
    main()
