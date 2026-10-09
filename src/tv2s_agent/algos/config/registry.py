"""Named PPO configurations, one per method.

  PPO_DEXTAG  DexTaG's teacher PPO (algos/config/registry.py: PPO_HAND_IMITATOR_TACTILE_MAP_XARM): temporal
              encoders, the actor without history, the critic with the privileged history and the
              reference glove image.
  PPO_RECON   Benchmark-Genesis' rl_recon/ppo.py: ELU MLP (512, 256, 128), observation normalisation with a
              std floor of 0.1, value clipping, near-zero actor output, initial std 0.4, KL thresholds 2x / 0.5x.
"""

import math

from tv2s_agent.algos.config.schema import LearningRateType, PPOArgs
from tv2s_agent.modules.config.schema import ActivationType, MLPConfig, TemporalEncoderConfig

# Input splits are filled from the env (TemporalEncoderConfig.with_layout).
TEMPORAL_ENCODER_DEXTAG = TemporalEncoderConfig(
    motion_latent_dim=128, tactile_map_latent_dim=128, history_latent_dim=64, future_latent_dim=128,
    mlp_hidden_dims=(512, 512, 256, 128), activation=ActivationType.SWISH, use_layer_norm=True,
)

PPO_DEXTAG = PPOArgs(
    policy_backbone=TEMPORAL_ENCODER_DEXTAG,
    critic_backbone=TEMPORAL_ENCODER_DEXTAG,
    lr=1e-4, lr_type=LearningRateType.ADAPTIVE, lr_adaptive_factor=1.5, lr_min=1e-5, lr_max=1e-2,
    kl_high=1.5, kl_low=0.5, target_kl=0.01,
    gamma=0.98, gae_lambda=0.95, clip_ratio=0.1, value_loss_coef=1.0, entropy_coef=1e-4, max_grad_norm=1.0,
    num_epochs=5, num_mini_batches=4, rollout_length=32,
)

PPO_RECON = PPOArgs(
    policy_backbone=MLPConfig(hidden_dims=(512, 256, 128), activation=ActivationType.ELU),
    critic_backbone=MLPConfig(hidden_dims=(512, 256, 128), activation=ActivationType.ELU),
    lr=5e-4, lr_type=LearningRateType.ADAPTIVE, lr_adaptive_factor=1.5, lr_min=1e-5, lr_max=5e-3,
    kl_high=2.0, kl_low=0.5, target_kl=0.01,
    gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, value_clip=True, value_loss_coef=1.0, entropy_coef=0.0,
    max_grad_norm=1.0, num_epochs=5, num_mini_batches=4, rollout_length=32,
    initial_log_std=math.log(0.4), actor_output_init=1e-3, obs_normalization=True, obs_norm_std_floor=0.1,
)

AlgoArgsRegistry = {"dextag": PPO_DEXTAG, "recon": PPO_RECON}
