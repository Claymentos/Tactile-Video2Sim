"""Named PPO configurations.

  PPO_RECON   Benchmark-Genesis' rl_recon/ppo.py: ELU MLP (512, 256, 128), observation normalisation with a
              std floor of 0.1, value clipping, near-zero actor output, initial std 0.4, KL thresholds 2x / 0.5x.
"""

import math

from tv2s_agent.algos.config.schema import LearningRateType, PPOArgs
from tv2s_agent.modules.config.schema import ActivationType, MLPConfig

PPO_RECON = PPOArgs(
    policy_backbone=MLPConfig(hidden_dims=(512, 256, 128), activation=ActivationType.ELU),
    critic_backbone=MLPConfig(hidden_dims=(512, 256, 128), activation=ActivationType.ELU),
    lr=5e-4, lr_type=LearningRateType.ADAPTIVE, lr_adaptive_factor=1.5, lr_min=1e-5, lr_max=5e-3,
    kl_high=2.0, kl_low=0.5, target_kl=0.01,
    gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, value_clip=True, value_loss_coef=1.0, entropy_coef=0.0,
    max_grad_norm=1.0, num_epochs=5, num_mini_batches=4, rollout_length=32,
    initial_log_std=math.log(0.4), actor_output_init=1e-3, obs_normalization=True, obs_norm_std_floor=0.1,
)

AlgoArgsRegistry = {"recon": PPO_RECON}
