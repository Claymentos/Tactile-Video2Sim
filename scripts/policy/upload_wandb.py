#!/usr/bin/env python3
"""A finished run (logs/<exp>/<time>/) uploaded to wandb after the fact, for a training launched without TV2S_WANDB=1
(wandb >= 0.30 has no `wandb sync --sync-tensorboard`).

Logged as the runner does (onpolicy_runner.MetricLogger, common.py's wandb_init): project $WANDB_PROJECT (default
tactile-video2sim), name $WANDB_RUN_NAME or the experiment, group $WANDB_RUN_GROUP, the run's configs; x axis
`iteration`. Uploaded: every row of progress.csv, the rollout videos (videos/*_checkpoint_<it>.mp4 -> rollout/video)
and the evaluations (eval_rl_recon/checkpoint_<it>.json -> eval/<start>/<metric>), which the runner never logs.

Run in the `genesis` env:
    python scripts/policy/upload_wandb.py logs/et_hammer_arm/20261029_113850
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="logs/<exp>/<time>")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()
    import wandb

    run = Path(args.run_dir)
    exp = run.parent.name
    cfg = {p.stem: json.loads(p.read_text()) for p in sorted((run / "configs").glob("*.json"))}
    env = cfg.get("env_args", {})
    w = wandb.init(project=os.environ.get("WANDB_PROJECT", "tactile-video2sim"), name=os.environ.get("WANDB_RUN_NAME") or exp,
                   group=os.environ.get("WANDB_RUN_GROUP") or None, job_type="train", dir=str(run),
                   tags=[env.get("method", "recon"), env.get("tactile", {}).get("mapping", ""), env.get("env_name", ""), "uploaded"],
                   config=dict(env=env, algo=cfg.get("algo_cfg"), runner=cfg.get("runner_args"), run_dir=str(run)))
    w.define_metric("iteration")
    w.define_metric("*", step_metric="iteration")

    with open(run / "progress.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        w.log({k: float(v) for k, v in row.items() if v not in ("", None)})
    print(f"{len(rows)} rows of progress.csv")

    evals = sorted((run / "eval_rl_recon").glob("checkpoint_*.json"))
    for p in evals:
        it = int(re.search(r"checkpoint_(\d+)", p.name).group(1))
        d = json.loads(p.read_text())
        w.log({"iteration": it, **{f"eval/{start}/{k}": v for start, m in d.items() if isinstance(m, dict)
                                   for k, v in m.items() if isinstance(v, (int, float))}})
    print(f"{len(evals)} evaluations")

    videos = [] if args.no_videos else sorted((run / "videos").glob("*_checkpoint_*.mp4"))
    for p in videos:
        it = int(re.search(r"checkpoint_(\d+)", p.name).group(1))
        w.log({"iteration": it, "rollout/video": wandb.Video(str(p), format="mp4")})
    print(f"{len(videos)} videos")
    w.finish()


if __name__ == "__main__":
    main()
