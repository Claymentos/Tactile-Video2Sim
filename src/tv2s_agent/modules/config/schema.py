"""Network configurations (DexTaG's gs_agent/modules/config/schema.py): an MLP backbone."""

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


class MLPConfig(BaseModel):
    model_config = frozen_config()
    network_type: NetworkBackboneType = NetworkBackboneType.MLP
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: ActivationType = ActivationType.ELU


NetworkBackboneConfig = MLPConfig
