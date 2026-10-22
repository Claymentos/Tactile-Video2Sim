#!/usr/bin/env python3
"""Benchmark-Genesis' RL reconstruction (rl_recon) on a per-episode reference: train the residual policy.

rl_recon's recipe (Benchmark-Genesis: reconstruction/rl_recon/{env,ppo,train}.py; env rl_recon, rl_recon_env.py): a
residual on the retargeted reference (base +-3 cm / 0.25 rad, fingers +-0.4 rad), the soft contact-F1 + reach tactile
reward against the glove's contacts, exponential hand kernels and bounded object hinges, the hand-anchored object
assist decaying to 0 over the first half of training. PPO with an ELU MLP, observation normalisation (std floor 0.1),
value clipping and a near-zero initial residual. launch/rl_recon.sh runs it, then scripts/policy/eval_rl_recon.py.

    python scripts/policy/run_ppo_recon.py --exp_name=et_screwdriver --env.data.reference=data/references/et_screwdriver/reference.npz \
        --num_envs=2048 --runner.total_iterations=2000
  starts: --env.train_start=random (anywhere, and in hand: Benchmark-Genesis' recipe) instead of frame 0
  ablations: --env.tactile_reward=False (train.py --no-tactile), --env.glove_obs=False (--no-glove-obs),
             --env.assist_init=0 (no assist)
  a fine-tune (train.py --resume ... --reset-iterations): --resume_checkpoint=<.pt> --reset_iterations=True
  --lightning: the same PPO trained by a Lightning Trainer (wandb logged from the LightningModule with TV2S_WANDB=1)
"""

import sys

from common import parse_cli, split_overrides, train_policy

MAIN_ARGS = {"exp_name", "env_name", "num_envs", "resume_checkpoint", "lightning", "reset_iterations", "device"}


def main(exp_name: str, env_name: str = "rl_recon", num_envs: int = 2048, resume_checkpoint: str | None = None,
         reset_iterations: bool = False, lightning: bool = False, device: str = "cuda", overrides: tuple = ({}, {}, {})) -> None:
    train_policy("recon", exp_name, env_name, num_envs, overrides, resume_checkpoint, device, reset_iterations, lightning)


if __name__ == "__main__":
    script, env_ov, algo_ov, runner_ov = split_overrides(parse_cli(sys.argv[1:]), MAIN_ARGS)
    if "exp_name" not in script:
        raise SystemExit(__doc__)
    main(**script, overrides=(env_ov, algo_ov, runner_ov))
