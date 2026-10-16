#!/usr/bin/env python3
"""DexTaG on HumanTouch: train / evaluate the bimanual tactile-as-guidance retargeter (teacher).

DexTaG's recipe (github: DexTaG, scripts/policy/run_ppo_single_hand_retargeting.py) on two floating
Wuji hands: relative base / absolute finger actions, the reference motion with 16 future frames and both
simulated glove images as observations (temporal encoder with a Conv2D glove encoder), DexTaG's tracking
+ tactile-map + tactile-gated proximity rewards, the decaying object guide force, perturbation kicks and
domain randomisation. Glove cells are placed on the Wuji with DexTaG's `layout` mapping by default.

Train:
    python scripts/policy/run_ppo_dextag.py --exp_name=dextag_v1 --num_envs=1024 --runner.total_iterations=15000
Evaluate (one env per clip, deterministic, no aids):
    python scripts/policy/run_ppo_dextag.py --exp_name=dextag_v1 --eval [--split_mode=eval] [--all_checkpoints]
Overrides (dot notation): --env.tactile.mapping=anatomical --env.data.split_mode=train
    --reward_args.WeightedGaussianBlurredTactileMapSimilarityReward.scale=0  --algo.lr=3e-4  --runner.save_interval=250
Lightning variant (same PPO, a Lightning Trainer, wandb logged from the LightningModule with TV2S_WANDB=1): --lightning
"""

import sys

from common import evaluate_run, parse_cli, split_overrides, train_policy

MAIN_ARGS = {"exp_name", "env_name", "num_envs", "eval", "checkpoint", "all_checkpoints", "split_mode",
             "resume_checkpoint", "lightning", "device"}


def main(exp_name: str, env_name: str = "bimanual_dextag", num_envs: int = 1024, eval: bool = False,
         checkpoint: str | None = None, all_checkpoints: bool = False, split_mode: str | None = None,
         resume_checkpoint: str | None = None, lightning: bool = False, device: str = "cuda",
         overrides: tuple = ({}, {}, {})) -> None:
    if eval:
        evaluate_run("dextag", exp_name, checkpoint, all_checkpoints, split_mode, overrides[0], device)
    else:
        train_policy("dextag", exp_name, env_name, num_envs, overrides, resume_checkpoint, device,
                     lightning=lightning)


if __name__ == "__main__":
    script, env_ov, algo_ov, runner_ov = split_overrides(parse_cli(sys.argv[1:]), MAIN_ARGS)
    if "exp_name" not in script:
        raise SystemExit(__doc__)
    main(**script, overrides=(env_ov, algo_ov, runner_ov))
