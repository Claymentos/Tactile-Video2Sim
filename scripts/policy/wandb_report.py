#!/usr/bin/env python3
"""Evaluation and comparison results to wandb (project $WANDB_PROJECT, default tactile-video2sim).

  eval EXP                       the run's evaluated checkpoints (<run>/eval/*.json) into its training run:
                                 eval/* against the iteration, the best checkpoint's summary, a per-clip
                                 table, and the best checkpoint's rollout video when rendered
  compare GROUP EXP [EXP ...]    one 'compare' run: method x mapping table (best evaluated checkpoint of
                                 each run, as summarize_runs.py) and one bar chart per metric
  mapping DIR                    the policy-free mapping comparison (compare_tactile_mappings.py's
                                 comparison.json / comparison.md) as tables

Run on a node with internet, or with WANDB_MODE=offline then `wandb sync`.
"""

import json
import os
import sys
from pathlib import Path

from common import REPO, latest_run_dir

PROJECT = os.environ.get("WANDB_PROJECT", "tactile-video2sim")
COLS = ["success", "completion", "tac_f1", "tac_precision", "tac_recall", "tac_map_similarity", "e_wrist_cm",
        "e_tip_mm", "e_obj_cm", "reward"]


def ckpt_iter(name: str) -> int:
    tail = name.split("_")[-1]
    return int(tail) if tail.isdigit() else -1


def load_evals(exp: str) -> tuple[Path, list[tuple[str, dict]]]:
    run_dir = latest_run_dir(REPO / "logs" / exp)
    evals = [(p.stem, json.loads(p.read_text())) for p in sorted((run_dir / "eval").glob("*.json"))]
    return run_dir, evals


def best_eval(evals: list[tuple[str, dict]]) -> tuple[str, dict]:
    return max(evals, key=lambda r: (round(r[1]["summary"]["tac_f1"], 3), r[1]["summary"]["completion"]))


def describe(exp: str) -> dict:
    run_dir = latest_run_dir(REPO / "logs" / exp)
    env = json.loads((run_dir / "configs" / "env_args.json").read_text())
    return dict(method=env.get("method", "?"), mapping=env["tactile"]["mapping"], split=env["data"]["split_mode"])


def cmd_eval(exp: str) -> None:
    import wandb

    run_dir, evals = load_evals(exp)
    if not evals:
        raise SystemExit(f"{exp}: no evaluation in {run_dir / 'eval'}")
    id_file = run_dir.parent / "wandb_id"
    run = wandb.init(project=PROJECT, id=id_file.read_text().strip() if id_file.exists() else None,
                     name=exp, resume="allow", dir=str(run_dir))
    run.define_metric("eval/*", step_metric="eval/iteration")
    for name, res in sorted(evals, key=lambda r: ckpt_iter(r[0])):
        if ckpt_iter(name) >= 0:
            run.log({"eval/iteration": ckpt_iter(name), **{f"eval/{k}": v for k, v in res["summary"].items()}})
    name, best = best_eval(evals)
    for k, v in best["summary"].items():
        run.summary[f"best/{k}"] = v
    run.summary["best/checkpoint"] = name
    keys = list(best["clips"][0])
    run.log({"eval/per_clip": wandb.Table(columns=keys, data=[[c[k] for k in keys] for c in best["clips"]])})
    for video in [run_dir / "videos" / "rollout_model_best.mp4", run_dir / "videos" / f"rollout_{name}.mp4"]:
        if video.exists():
            run.log({"eval/best_rollout": wandb.Video(str(video), caption=f"{exp} {name}", format="mp4")})
            break
    run.finish()
    print(f"{exp}: {len(evals)} evaluations logged, best {name} (F1 {best['summary']['tac_f1']:.3f})")


def cmd_compare(group: str, exps: list[str]) -> None:
    import wandb

    rows = []
    for exp in exps:
        try:
            _, evals = load_evals(exp)
        except FileNotFoundError:
            print(f"{exp}: no run")
            continue
        if not evals:
            print(f"{exp}: not evaluated")
            continue
        name, best = best_eval(evals)
        d = describe(exp)
        rows.append([exp, d["method"], d["mapping"], name] + [best["summary"].get(c, float("nan")) for c in COLS])
    if not rows:
        raise SystemExit("nothing to compare")
    run = wandb.init(project=PROJECT, name=f"{group}_comparison", group=group, job_type="compare",
                     dir=str(REPO / "logs"))
    columns = ["run", "method", "mapping", "checkpoint"] + COLS
    table = wandb.Table(columns=columns, data=rows)
    log = {"comparison/table": table}
    for c in COLS:
        bar = wandb.Table(columns=["config", c], data=[[f"{r[1]} / {r[2]}", r[columns.index(c)]] for r in rows])
        log[f"comparison/{c}"] = wandb.plot.bar(bar, "config", c, title=f"{c} (best checkpoint, eval clips)")
    run.log(log)
    run.finish()
    print("\n".join(" ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) for r in rows))


def _flatten(prefix: str, obj, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)) and obj and not isinstance(obj[0], (dict, list)):
        out.append([prefix, json.dumps(obj)])
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}/{i}", v, out)
    else:
        out.append([prefix, obj if isinstance(obj, (int, float)) else json.dumps(obj)])


def cmd_mapping(folder: str) -> None:
    import wandb

    folder = Path(folder)
    res = json.loads((folder / "comparison.json").read_text())
    run = wandb.init(project=PROJECT, name="tactile_mapping_replay", job_type="mapping_comparison",
                     group=os.environ.get("WANDB_RUN_GROUP") or None, dir=str(folder))
    rows = []
    _flatten("", res, rows)
    log = {"mapping/all_values": wandb.Table(columns=["key", "value"], data=[[k, str(v)] for k, v in rows])}
    numeric = {f"mapping/{k}": v for k, v in rows if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for k, v in numeric.items():
        run.summary[k] = v
    md = folder / "comparison.md"
    if md.exists():
        log["mapping/report"] = wandb.Html(f"<pre>{md.read_text()}</pre>")
    for png in sorted((REPO / "assets" / "tactile").glob("mappings_*.png")):
        log[f"mapping/{png.stem}"] = wandb.Image(str(png))
    run.log(log)
    run.finish()
    print(f"mapping comparison logged ({len(numeric)} numbers)")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "eval":
        for exp in args:
            cmd_eval(exp)
    elif cmd == "compare":
        cmd_compare(args[0], args[1:])
    elif cmd == "mapping":
        cmd_mapping(args[0])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
