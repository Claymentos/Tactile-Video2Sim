#!/usr/bin/env python3
"""Benchmark-Genesis' RL reconstruction (rl_recon) on HumanTouch: train / evaluate the bimanual residual policy.

rl_recon's recipe (Benchmark-Genesis: reconstruction/rl_recon/{env,ppo,train}.py) on two floating Wuji
hands: a residual on the retargeted reference (base +-3 cm / 0.25 rad, fingers +-0.4 rad), the soft
contact-F1 + reach tactile reward against the glove's binary contacts, exponential hand kernels and
bounded object hinges, random / in-grasp (RSI) starts, and the hand-anchored object assist decaying to
0 over the first half of training. PPO with an ELU MLP, observation normalisation (std floor 0.1),
value clipping and a near-zero initial residual. Glove cells are carried onto the Wuji with the
`anatomical` mapping by default.

Train:
    python scripts/policy/run_ppo_recon.py --exp_name=recon_v1 --num_envs=2048 --runner.total_iterations=1500
Evaluate every checkpoint and keep the best (model_best.pt), as rl_recon's recon_rsl_eval.py --all:
    python scripts/policy/run_ppo_recon.py --exp_name=recon_v1 --eval --all_checkpoints
Ablations: no tactile reward: --reward_args.TactileContactF1Reward.scale=0;
    --env.glove_obs=False hides the glove contacts from the policy; --env.assist_init=0 disables the assist.

On a per-episode reference (rl_recon_env.py: Benchmark-Genesis' env.py, one or two hands; launch/rl_recon.sh):
    python scripts/policy/run_ppo_recon.py --exp_name=screwdriver_v1 --env_name=rl_recon \
        --env.data.reference=data/references/screwdriver_103227/reference.npz --num_envs=2048 --runner.total_iterations=2000
    python scripts/policy/eval_rl_recon.py --exp_name=screwdriver_v1 --all
  ablations: --env.tactile_reward=False (train.py --no-tactile), --env.glove_obs=False (--no-glove-obs);
  a fine-tune (train.py --resume ... --reset-iterations): --resume_checkpoint=<.pt> --reset_iterations=True

Lightning variant (same PPO, a Lightning Trainer, wandb logged from the LightningModule with TV2S_WANDB=1): --lightning
"""

import sys

from common import evaluate_run, parse_cli, split_overrides, train_policy

MAIN_ARGS = {"exp_name", "env_name", "num_envs", "eval", "checkpoint", "all_checkpoints", "split_mode",
             "resume_checkpoint", "lightning", "reset_iterations", "device"}


def main(exp_name: str, env_name: str = "bimanual_recon", num_envs: int = 2048, eval: bool = False,
         checkpoint: str | None = None, all_checkpoints: bool = False, split_mode: str | None = None,
         resume_checkpoint: str | None = None, reset_iterations: bool = False, lightning: bool = False, device: str = "cuda",
         overrides: tuple = ({}, {}, {})) -> None:
    if eval:
        evaluate_run("recon", exp_name, checkpoint, all_checkpoints, split_mode, overrides[0], device)
    else:
        train_policy("recon", exp_name, env_name, num_envs, overrides, resume_checkpoint, device, reset_iterations, lightning)


if __name__ == "__main__":
    script, env_ov, algo_ov, runner_ov = split_overrides(parse_cli(sys.argv[1:]), MAIN_ARGS)
    if "exp_name" not in script:
        raise SystemExit(__doc__)
    main(**script, overrides=(env_ov, algo_ov, runner_ov))
