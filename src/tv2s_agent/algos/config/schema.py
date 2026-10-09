"""PPO configuration (DexTaG's PPOArgs, plus the switches that reproduce Benchmark-Genesis' rl_recon/ppo.py)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from tv2s_agent.modules.config.schema import MLPConfig, NetworkBackboneConfig


class LearningRateType(str, Enum):
    FIXED = "FIXED"
    ADAPTIVE = "ADAPTIVE"


class PPOArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=(), use_enum_values=True)

    policy_backbone: NetworkBackboneConfig = MLPConfig()
    critic_backbone: NetworkBackboneConfig = MLPConfig()

    lr: float = 3e-4
    value_lr: float | None = None            # None: same as lr
    lr_type: LearningRateType = LearningRateType.ADAPTIVE
    lr_adaptive_factor: float = 1.5
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    kl_high: float = 1.5                     # lr / factor if KL > kl_high * target_kl (DexTaG 1.5, rl_recon 2)
    kl_low: float = 0.5                      # lr * factor if KL < kl_low * target_kl
    target_kl: float = 0.01

    gamma: float = Field(default=0.98, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_ratio: float = 0.2
    value_clip: bool = False                 # rl_recon clips the value update like the policy
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    num_mini_batches: int = 4
    rollout_length: int = 32

    initial_log_std: float = 0.0
    actor_output_init: float | None = None  # rl_recon: 1e-3 (start from the zero residual)
    obs_normalization: bool = False          # rl_recon: running normaliser with a std floor
    obs_norm_std_floor: float = 0.1
    critic_warmup_iters: int = 0
