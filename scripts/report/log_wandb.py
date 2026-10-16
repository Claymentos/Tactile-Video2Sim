#!/usr/bin/env python3
"""Send every result of the reconstruction pipeline to wandb: one run per episode (job_type "results").

Per episode (EPISODES of collect_results.py), from what collect_results.py gathered into outputs/report/
(run it first) and the replays of scripts/policy/visualize_episode.py:
  train/*      the training curves (logs/<run>/*/progress.csv, resumed runs joined), x = iteration
  eval/*       the evaluation of every checkpoint (whole demonstration from frame 0), x = checkpoint
  replay/*     the best policy's replay frame by frame against the reference: wrist / object errors,
               contact F1, resyncs, x = time (s)
  trajectories palm and object positions, policy against reference, per axis (line plots), and the
               palm paths in 3D (policy blue, reference orange)
  media        perception videos (SAM3, registration, CoTracker), reference and policy replays,
               HaWoR vs mocap; tables of the scene objects, the HaWoR errors and the evaluations

    python scripts/report/collect_results.py && python scripts/report/log_wandb.py [--episodes humantouch touchscale]

Project $WANDB_PROJECT (default tactile-video2sim); each call makes new runs (named <run>_results).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import EPISODES, REPO  # noqa: E402

SIDES = ("left", "right")


def progress_rows(run: str) -> list[dict]:
    rows = {}
    for f in sorted((REPO / "logs" / run).glob("*/progress.csv")):  # a resumed run continues the curve
        for r in csv.DictReader(open(f)):
            it = int(float(r.pop("iteration")))
            rows[it] = {k: float(v) for k, v in r.items() if v not in (None, "")}
    return [dict(iteration=k, **rows[k]) for k in sorted(rows)]


def evaluations(run: str) -> list[dict]:
    out = {}
    for f in sorted((REPO / "logs" / run).glob("*/eval/checkpoint_*.json")):
        out[int(f.stem.split("_")[-1])] = json.loads(f.read_text())["summary"]
    return [dict(checkpoint=k, **out[k]) for k in sorted(out)]


def series(wandb, x, ys: dict, title: str, xname: str):
    keys = list(ys)
    return wandb.plot.line_series(xs=list(map(float, x)), ys=[list(map(float, ys[k])) for k in keys], keys=keys,
                                  title=title, xname=xname)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", default=list(EPISODES), choices=list(EPISODES))
    ap.add_argument("--report", default=str(REPO / "outputs" / "report"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "tactile-video2sim"))
    args = ap.parse_args()
    import wandb

    report = Path(args.report)
    data = json.loads((report / "data.json").read_text())
    for key in args.episodes:
        e, d = EPISODES[key], data[key]
        run = wandb.init(project=args.project, name=f"{e['run']}_results", group="results", job_type="results",
                         tags=[key, "results"], config=dict(episode=d["name"], run=e["run"], clip=e["clip"]),
                         dir=str(report), reinit="finish_previous")
        run.define_metric("iteration")
        run.define_metric("train/*", step_metric="iteration")
        run.define_metric("eval/checkpoint")
        run.define_metric("eval/*", step_metric="eval/checkpoint")
        run.define_metric("replay/time_s")
        run.define_metric("replay/*", step_metric="replay/time_s")

        for r in progress_rows(e["run"]):
            run.log({"iteration": r.pop("iteration"), **{f"train/{k}": v for k, v in r.items()}})
        evals = evaluations(e["run"])
        for v in evals:
            run.log({"eval/checkpoint": v["checkpoint"], **{f"eval/{k}": x for k, x in v.items() if k != "checkpoint"}})
        if evals:
            cols = list(evals[0])
            run.log({"tables/evaluations": wandb.Table(columns=cols, data=[[v[c] for c in cols] for v in evals])})

        out = REPO / "outputs" / "episodes"
        pol = next(iter(sorted(out.glob(f"{e['run']}_*.npz"))), None)
        ref = out / f"{e['reference']}.npz"
        if pol is not None and ref.exists():
            P, R = np.load(pol), np.load(ref)
            n = min(len(P["t"]), len(R["t"]))
            t = np.arange(n) / 30.0
            for k in range(n):
                run.log({"replay/time_s": t[k], "replay/wrist_error_cm": P["e_wrist"][k] * 100,
                         "replay/object_error_cm": P["e_obj"][k] * 100, "replay/contact_f1": P["tac_f1"][k],
                         "replay/resync": float(P["resync"][k])})
            plots = {}
            for h, s in enumerate(SIDES):
                for a, ax in enumerate("xyz"):
                    plots[f"trajectories/palm_{s}_{ax}"] = series(
                        wandb, t, {"politique": P["palm_pos"][:n, h, a], "référence": R["palm_pos"][:n, h, a]},
                        f"Paume {s}, {ax} (m)", "temps (s)")
            for o, oid in enumerate(P["object_ids"]):
                for a, ax in enumerate("xyz"):
                    plots[f"trajectories/object_{oid}_{ax}"] = series(
                        wandb, t, {"politique": P["objects_pos"][:n, o, a], "référence": R["objects_pos"][:n, o, a]},
                        f"{oid}, {ax} (m)", "temps (s)")
            paths = []
            for arr, rgb in ((P, (42, 120, 214)), (R, (230, 140, 40))):
                pts = arr["palm_pos"][:n].reshape(-1, 3)
                paths.append(np.hstack([pts, np.tile(rgb, (len(pts), 1))]))
            plots["trajectories/palms_3d"] = wandb.Object3D(np.vstack(paths).astype(np.float32))
            for o, oid in enumerate(P["object_ids"]):
                obj = [np.hstack([a["objects_pos"][:n, o], np.tile(rgb, (n, 1))]) for a, rgb in ((P, (42, 120, 214)), (R, (230, 140, 40)))]
                plots[f"trajectories/{oid}_3d"] = wandb.Object3D(np.vstack(obj).astype(np.float32))
            run.log(plots)
            run.summary.update({f"replay_summary/{k}": v for k, v in json.loads(pol.with_suffix(".json").read_text()).items()})

        media = {}
        for m in d["media"]:
            name = "media/" + Path(m["file"]).stem.removeprefix(f"{key}_")
            f = report / m["file"]
            media[name] = wandb.Video(str(f), caption=f"{m['title']}: {m['note']}", format="mp4") if m["kind"] == "video" \
                else wandb.Image(str(f), caption=m["title"])
        if d["objects"]:
            cols = ["id", "fixed", "material", "mass_g", "extent_cm", "gemini_cm"]
            media["tables/scene_objects"] = wandb.Table(columns=cols, data=[[str(o.get(c)) if isinstance(o.get(c), list) else o.get(c)
                                                                             for c in cols] for o in d["objects"]])
        if d["hawor"]:
            cols = sorted({c for r in d["hawor"].values() for c in r})
            media["tables/hawor_vs_mocap"] = wandb.Table(columns=["hand"] + cols, data=[[s] + [r.get(c) for c in cols] for s, r in d["hawor"].items()])
        run.log(media)
        print(f"{key}: {run.url}")
        run.finish()


if __name__ == "__main__":
    main()
