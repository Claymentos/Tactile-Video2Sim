"""Network backbones: DexTaG's MLP (gs_agent/modules/models.py) and Benchmark-Genesis' running observation
normaliser (rl_recon/ppo.py)."""

import torch
from torch import nn

from tv2s_agent.modules.config.schema import (
    ActivationType,
    MLPConfig,
    NetworkBackboneConfig,
    NetworkBackboneType,
)


def get_activation(activation: ActivationType | str) -> nn.Module:
    return {"RELU": nn.ReLU, "ELU": nn.ELU, "TANH": nn.Tanh, "SWISH": nn.SiLU}[str(getattr(activation, "value", activation))]()


class RunningNorm(nn.Module):
    """Running mean / variance normaliser with a std floor (rl_recon/ppo.py): features constant over a
    rollout (a glove channel that never fired) would otherwise get var ~1e-12 and land at the clip the
    first time they flip, spiking the KL."""

    def __init__(self, dim: int, clip: float = 5.0, std_floor: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip, self.std_floor = clip, std_floor

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var).clamp_min(self.std_floor)).clamp(-self.clip, self.clip)


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, config: MLPConfig, output_dim: int | None = None) -> None:
        super().__init__()
        layers, d = [], input_dim
        for h in config.hidden_dims:
            layers += [nn.Linear(d, h), get_activation(config.activation)]
            d = h
        if output_dim is not None:
            layers.append(nn.Linear(d, output_dim))
        self._network = nn.Sequential(*layers)
        self.input_dim, self.output_dim = input_dim, output_dim if output_dim is not None else d

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._network(obs)


def create_network(config: NetworkBackboneConfig, input_dim: int, output_dim: int | None = None) -> nn.Module:
    match NetworkBackboneType(config.network_type):
        case NetworkBackboneType.MLP:
            return MLPBackbone(input_dim, config, output_dim)
    raise ValueError(config.network_type)
