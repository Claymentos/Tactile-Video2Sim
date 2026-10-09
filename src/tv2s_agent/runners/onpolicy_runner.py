"""On-policy training loop with logging and periodic checkpoints (DexTaG's gs_agent/runners/onpolicy_runner.py)."""

import csv
import time
from pathlib import Path
from typing import Any

from tv2s_agent.runners.config.schema import RunnerArgs


class MetricLogger:
    """stdout summary + one CSV row per logged iteration + TensorBoard (if installed)."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.folder / "progress.csv"
        self._fields: list[str] | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(str(self.folder / "tb"))
        except ImportError:
            self._tb = None

    def dump(self, step: int, metrics: dict[str, float]) -> None:
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
        for it in range(start_iteration, total):
            env.set_training_progress(it, total)
            info = self.algorithm.train_one_iteration()
            if it % self.args.log_interval == 0 or it == total - 1:
                flat = flatten(info)
                logger.dump(it, flat)
                m, r = info["metrics"], info["rollout"]
                print(f"it {it:6d} | reward {r['mean_reward']:8.3f} len {r['mean_length']:6.1f} | "
                      f"tac F1 {m['tac_f1']:.3f} map {m['tac_map_similarity']:.3f} | wrist {100 * m['e_wrist']:.1f} cm "
                      f"tip {1000 * m['e_tip']:.0f} mm obj {100 * m['e_obj']:.1f} cm | "
                      f"lr {info['train']['learning_rate']:.1e} kl {info['train']['kl_mean']:.4f} "
                      f"std {info['train']['mean_std']:.3f} | {info['speed']['rollout_fps']:.0f} fps | "
                      f"{(time.time() - tic) / 60:.1f} min", flush=True)
            if (it + 1) % self.args.save_interval == 0 or it == total - 1:
                path = self.checkpoint_dir / f"checkpoint_{it + 1:05d}.pt"
                self.algorithm.save(path)
                print(f"Checkpoint saved: {path}")
        return dict(total_time=time.time() - tic, iterations=total - start_iteration)
