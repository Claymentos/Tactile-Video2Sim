"""On-policy training loop with logging and periodic checkpoints (DexTaG's gs_agent/runners/onpolicy_runner.py)."""

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from tv2s_agent.runners.config.schema import RunnerArgs


class MetricLogger:
    """stdout summary + one CSV row per logged iteration + TensorBoard (if installed) + wandb (TV2S_WANDB=1).

    wandb: project $WANDB_PROJECT (default tactile-video2sim), group $WANDB_RUN_GROUP, run name = experiment.
    The run id is kept in logs/<exp>/wandb_id so that a resumed training continues the same wandb run.
    Rollout videos written by scripts/policy/render_rollouts.py into <run>/videos are uploaded as they appear."""

    def __init__(self, folder: Path, wandb_init: dict | None = None) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.folder / "progress.csv"
        self._fields: list[str] | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(str(self.folder / "tb"))
        except ImportError:
            self._tb = None
        self._wandb = None
        self._videos_logged: set[str] = set()
        self._env_steps = 0
        self._tic = time.time()
        if wandb_init is not None and os.environ.get("TV2S_WANDB", "0") == "1":
            import wandb

            id_file = self.folder.parent / "wandb_id"
            run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
            id_file.write_text(run_id)
            self._wandb = wandb.init(project=os.environ.get("WANDB_PROJECT", "tactile-video2sim"), id=run_id,
                                     resume="allow", dir=str(self.folder), **wandb_init)
            self._wandb.define_metric("iteration")
            self._wandb.define_metric("*", step_metric="iteration")
            (self.folder / "wandb_run.json").write_text(json.dumps(dict(id=run_id, project=self._wandb.project)))

    def _new_videos(self) -> dict[str, Any]:
        return new_rollout_videos(self.folder, self._videos_logged)

    def finish(self) -> None:
        if self._wandb is not None:
            extra = self._new_videos()
            if extra:
                self._wandb.log(extra)
            self._wandb.finish()

    def dump(self, step: int, metrics: dict[str, float]) -> None:
        self._env_steps += int(metrics.get("speed/rollout_step", 0)) * max(1, int(metrics.pop("_iterations_since_log", 1)))
        if self._wandb is not None:
            self._wandb.log({"iteration": step, "progress/env_steps": self._env_steps,
                             "progress/wall_hours": (time.time() - self._tic) / 3600, **metrics, **self._new_videos()})
        row = {"iteration": step, **metrics}
        if self._fields is None:
            self._fields = list(row)
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, self._fields, extrasaction="ignore").writeheader()
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, self._fields, extrasaction="ignore").writerow(row)
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)


def new_rollout_videos(folder: Path, logged: set[str]) -> dict[str, Any]:
    """wandb items for the rollout videos written into <run>/videos since the last call (render_rollouts.py),
    with the clips' contact F1 and success from their .json; `logged` keeps the names already sent."""
    import wandb

    out = {}
    for mp4 in sorted((Path(folder) / "videos").glob("rollout_*.mp4")):
        if mp4.name in logged or mp4.name.endswith(".tmp.mp4"):
            continue
        logged.add(mp4.name)
        meta_path = mp4.with_suffix(".json")
        caption = mp4.stem
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            caption = f"{meta['checkpoint']}: " + ", ".join(
                f"{c['clip']} F1 {c['tac_f1']:.2f}{' ok' if c['success'] else ''}" for c in meta["clips"])
            out["video_clips/mean_tac_f1"] = sum(c["tac_f1"] for c in meta["clips"]) / len(meta["clips"])
            out["video_clips/success"] = sum(c["success"] for c in meta["clips"]) / len(meta["clips"])
        out["rollout_video"] = wandb.Video(str(mp4), caption=caption, format="mp4")
    return out


def iteration_summary(it: int, info: dict[str, Any], elapsed_s: float) -> str:
    """One stdout line per logged iteration."""
    m, r = info["metrics"], info["rollout"]
    return (f"it {it:6d} | reward {r['mean_reward']:8.3f} len {r['mean_length']:6.1f} | "
            f"tac F1 {m['tac_f1']:.3f} map {m['tac_map_similarity']:.3f} | wrist {100 * m['e_wrist']:.1f} cm "
            f"tip {1000 * m['e_tip']:.0f} mm obj {100 * m['e_obj']:.1f} cm | "
            f"lr {info['train']['learning_rate']:.1e} kl {info['train']['kl_mean']:.4f} "
            f"std {info['train']['mean_std']:.3f} | {info['speed']['rollout_fps']:.0f} fps | "
            f"{elapsed_s / 60:.1f} min")


def flatten(metrics: dict[str, Any]) -> dict[str, float]:
    return {f"{group}/{k}": float(v) for group, d in metrics.items() if isinstance(d, dict) for k, v in d.items()}


class OnPolicyRunner:
    def __init__(self, algorithm: Any, runner_args: RunnerArgs, run_dir: Path) -> None:
        self.algorithm, self.args = algorithm, runner_args
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, logger: MetricLogger, start_iteration: int = 0) -> dict[str, Any]:
        env = self.algorithm.env
        total = self.args.total_iterations
        tic = time.time()
        last_log = start_iteration - 1
        for it in range(start_iteration, total):
            env.set_training_progress(it, total)
            info = self.algorithm.train_one_iteration()
            if it % self.args.log_interval == 0 or it == total - 1:
                flat = flatten(info)
                flat["_iterations_since_log"] = it - last_log
                last_log = it
                logger.dump(it, flat)
                print(iteration_summary(it, info, time.time() - tic), flush=True)
            if (it + 1) % self.args.save_interval == 0 or it == total - 1:
                path = self.checkpoint_dir / f"checkpoint_{it + 1:05d}.pt"
                self.algorithm.save(path)
                print(f"Checkpoint saved: {path}")
        return dict(total_time=time.time() - tic, iterations=total - start_iteration)
