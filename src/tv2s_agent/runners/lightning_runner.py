"""The on-policy training loop as a Lightning module: a variant of OnPolicyRunner with the same PPO.

One Lightning `training_step` is one PPO iteration: the rollouts (the env steps itself, there is no dataset:
the dataloader only yields the iteration numbers), then the epochs x minibatches update with manual
optimisation (two optimisers, the actor's learning rate driven by the KL, the actor frozen during the critic
warm-up). The losses, the KL-adaptive learning rate, the normalisers and the checkpoints are PPO's own
(tv2s_agent.algos.ppo), so a run of either runner is evaluated the same way (checkpoints/checkpoint_*.pt).

Logging: wandb from the module (`self.logger.experiment`, a WandbLogger's run) when TV2S_WANDB=1, x axis the
PPO iteration; progress.csv and TensorBoard as OnPolicyRunner (MetricLogger without wandb).
"""

import json
import os
import random
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from tv2s_agent.runners.config.schema import RunnerArgs
from tv2s_agent.runners.onpolicy_runner import MetricLogger, flatten, iteration_summary


class PPOLightningModule(L.LightningModule):
    def __init__(self, ppo: Any, runner_args: RunnerArgs, run_dir: Path, csv_logger: MetricLogger,
                 start_iteration: int = 0) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.ppo, self.args, self.run_dir = ppo, runner_args, Path(run_dir)
        # registered as submodules: Lightning sees (and places) the parameters PPO optimises
        self.actor, self.critic = ppo._actor, ppo._critic
        self.csv_logger = csv_logger
        self.start_iteration = start_iteration
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._env_steps = 0
        self._last_log = start_iteration - 1
        self._tic = time.time()
        self._rng_state = rng_state()

    # ------------------------------------------------------------------ Lightning plumbing
    def configure_optimizers(self):
        # PPO's optimisers: its KL-adaptive learning rate keeps acting on the actor's param groups
        return [self.ppo._actor_optimizer, self.ppo._critic_optimizer]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(range(self.start_iteration, self.args.total_iterations), batch_size=None)

    @property
    def wandb_run(self):
        """The wandb run of the WandbLogger, None without one."""
        from lightning.pytorch.loggers import WandbLogger

        return self.logger.experiment if isinstance(self.logger, WandbLogger) else None

    def on_train_start(self) -> None:
        self._tic = time.time()
        run = self.wandb_run
        if run is not None:
            run.define_metric("iteration")
            run.define_metric("*", step_metric="iteration")      # every curve against the PPO iteration
            (self.run_dir / "wandb_run.json").write_text(json.dumps(dict(id=run.id, project=run.project)))

    # ------------------------------------------------------------------ one PPO iteration
    def training_step(self, batch: Any, batch_idx: int) -> None:
        it, ppo, total = int(batch), self.ppo, self.args.total_iterations
        if self._rng_state is not None:       # the Trainer drew random numbers (dataloader seed): undo them, so
            set_rng_state(self._rng_state)    # that both runners train identically from the same seed
            self._rng_state = None
        ppo.env.set_training_progress(it, total)
        t0 = time.time()
        rollout = ppo._collect_rollouts()
        t1 = time.time()
        opt_actor, opt_critic = self.optimizers()
        stats = []
        for mb in ppo.minibatches():
            loss, s = ppo.compute_losses(mb)
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt_actor, gradient_clip_val=ppo.cfg.max_grad_norm, gradient_clip_algorithm="norm")
            self.clip_gradients(opt_critic, gradient_clip_val=ppo.cfg.max_grad_norm, gradient_clip_algorithm="norm")
            opt_critic.step()
            if ppo.actor_steps:
                opt_actor.step()
            stats.append(s)
        info = ppo.finish_iteration(rollout, stats, t0, t1)

        self.log("reward", info["rollout"]["mean_reward"], prog_bar=True, logger=False)
        self.log("tac_f1", info["metrics"]["tac_f1"], prog_bar=True, logger=False)
        if it % self.args.log_interval == 0 or it == total - 1:
            self._log_iteration(it, info)
        if (it + 1) % self.args.save_interval == 0 or it == total - 1:
            path = self.checkpoint_dir / f"checkpoint_{it + 1:05d}.pt"
            ppo.save(path)
            print(f"Checkpoint saved: {path}")

    def _log_iteration(self, it: int, info: dict[str, Any]) -> None:
        flat = flatten(info)
        self._env_steps += int(flat["speed/rollout_step"]) * (it - self._last_log)
        self._last_log = it
        run = self.wandb_run
        if run is not None:
            run.log({"iteration": it, "progress/env_steps": self._env_steps,
                     "progress/wall_hours": (time.time() - self._tic) / 3600,
                     **flat})
        self.csv_logger.dump(it, flat)
        print("\n" + iteration_summary(it, info, time.time() - self._tic), flush=True)


def rng_state() -> dict[str, Any]:
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), python=random.getstate())


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def wandb_logger(run_dir: Path, wandb_init: dict | None):
    """A WandbLogger when TV2S_WANDB=1 (as MetricLogger), else False. The run id is kept in logs/<exp>/wandb_id,
    so a resumed training continues the same wandb run."""
    if wandb_init is None or os.environ.get("TV2S_WANDB", "0") != "1":
        return False
    from lightning.pytorch.loggers import WandbLogger

    id_file = Path(run_dir).parent / "wandb_id"
    run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
    id_file.write_text(run_id)
    return WandbLogger(project=os.environ.get("WANDB_PROJECT", "tactile-video2sim"), id=run_id, resume="allow",
                       save_dir=str(run_dir), **wandb_init)


def train_lightning(ppo: Any, runner_args: RunnerArgs, run_dir: Path, wandb_init: dict | None,
                    start_iteration: int = 0) -> dict[str, Any]:
    """PPO trained by a Lightning Trainer (one epoch = every remaining iteration)."""
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    csv_logger = MetricLogger(run_dir)                    # progress.csv + TensorBoard, no wandb
    module = PPOLightningModule(ppo, runner_args, run_dir, csv_logger, start_iteration)
    logger = wandb_logger(run_dir, wandb_init)
    trainer = L.Trainer(
        accelerator="gpu" if ppo.device.type == "cuda" else "cpu", devices=1, max_epochs=1, logger=logger,
        enable_checkpointing=False, num_sanity_val_steps=0, default_root_dir=str(run_dir),
        enable_model_summary=True,
    )
    tic = time.time()
    try:
        trainer.fit(module)
    finally:
        if logger:
            logger.experiment.finish()
    return dict(total_time=time.time() - tic, iterations=runner_args.total_iterations - start_iteration)
