"""Network configurations (DexTaG's gs_agent/modules/config/schema.py).

The registries hold the architecture (latent sizes, hidden layers); the input splits of the temporal
encoder (`motion_dim`, `proprio_dim`, ...) are filled from the environment's observation layout at
build time (`with_layout`), so they never drift from the observation terms.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict


def frozen_config() -> ConfigDict:
    return ConfigDict(extra="forbid", frozen=True, protected_namespaces=(), use_enum_values=True)


class ActivationType(str, Enum):
    RELU = "RELU"
    ELU = "ELU"
    TANH = "TANH"
    SWISH = "SWISH"


class NetworkBackboneType(str, Enum):
    MLP = "MLP"
    TEMPORAL_ENCODER = "TEMPORAL_ENCODER"


class MLPConfig(BaseModel):
    model_config = frozen_config()
    network_type: NetworkBackboneType = NetworkBackboneType.MLP
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: ActivationType = ActivationType.ELU


class TemporalEncoderConfig(BaseModel):
    """DexTaG's temporal encoder: [motion | proprio | tactile maps | target maps | history | future]."""
    model_config = frozen_config()
    network_type: NetworkBackboneType = NetworkBackboneType.TEMPORAL_ENCODER
    # input splits (filled from the env's obs_layout)
    motion_dim: int = 0
    proprio_dim: int = 0
    tactile_maps: int = 0                 # simulated glove images (one per hand)
    target_tactile_maps: int = 0          # reference glove images (critic)
    tactile_map_h: int = 23
    tactile_map_w: int = 20
    history_timesteps: int = 0
    history_per_step_dim: int = 0
    future_timesteps: int = 0
    future_per_step_dim: int = 0
    # architecture
    motion_latent_dim: int = 128
    tactile_map_latent_dim: int = 128
    history_latent_dim: int = 64
    future_latent_dim: int = 128
    mlp_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: ActivationType = ActivationType.SWISH
    use_layer_norm: bool = True

    def with_layout(self, layout: dict[str, int]) -> "TemporalEncoderConfig":
        return self.model_copy(update={k: v for k, v in layout.items() if k in type(self).model_fields})

    @property
    def input_dim(self) -> int:
        img = self.tactile_map_h * self.tactile_map_w
        return (self.motion_dim + self.proprio_dim + (self.tactile_maps + self.target_tactile_maps) * img
                + self.history_timesteps * self.history_per_step_dim + self.future_timesteps * self.future_per_step_dim)


NetworkBackboneConfig = MLPConfig | TemporalEncoderConfig
