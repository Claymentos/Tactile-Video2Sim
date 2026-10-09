"""Network backbones: DexTaG's MLP and temporal encoder (gs_agent/modules/models.py), with the tactile-map
encoder taking one channel per hand (and per reference map), and Benchmark-Genesis' running observation
normaliser (rl_recon/ppo.py)."""

import torch
from torch import nn

from tv2s_agent.modules.config.schema import (
    ActivationType,
    MLPConfig,
    NetworkBackboneConfig,
    NetworkBackboneType,
    TemporalEncoderConfig,
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


class TemporalEncoderBackbone(nn.Module):
    """Target motion -> motion latent (raw motion also passed through); proprioception passed through;
    glove images -> Conv2D latent; history (timesteps x per-step) -> Conv1D latent; future -> Conv1D latent;
    everything concatenated into an MLP."""

    def __init__(self, input_dim: int, config: TemporalEncoderConfig, output_dim: int | None = None) -> None:
        super().__init__()
        c = config
        assert input_dim == c.input_dim, f"input dim {input_dim} != layout {c.input_dim} ({c})"
        self.c = c
        act = lambda: get_activation(c.activation)
        self.motion_encoder = nn.Sequential(nn.Linear(c.motion_dim, 60), act(), nn.Linear(60, c.motion_latent_dim)) \
            if c.motion_dim else None
        n_maps = c.tactile_maps + c.target_tactile_maps
        self.tactile_map_encoder = nn.Sequential(
            nn.Conv2d(n_maps, 16, 4, 2, 1), act(), nn.Conv2d(16, 32, 4, 2, 1), act(), nn.Conv2d(32, 64, 4, 2, 1), act(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(64 * 9, c.tactile_map_latent_dim), act(),
        ) if n_maps else None
        self.history_encoder = self._temporal(c.history_per_step_dim, c.history_timesteps, c.history_latent_dim, act) \
            if c.history_timesteps else None
        self.future_encoder = self._temporal(c.future_per_step_dim, c.future_timesteps, c.future_latent_dim, act) \
            if c.future_timesteps else None
        d = (c.motion_dim + c.proprio_dim + (c.motion_latent_dim if c.motion_dim else 0)
             + (c.tactile_map_latent_dim if n_maps else 0) + (c.history_latent_dim if c.history_timesteps else 0)
             + (c.future_latent_dim if c.future_timesteps else 0))
        layers = []
        for h in c.mlp_hidden_dims:
            layers += [nn.Linear(d, h), act()]
            if h == 256 and c.use_layer_norm:
                layers.append(nn.LayerNorm(h))
            d = h
        if output_dim is not None:
            layers.append(nn.Linear(d, output_dim))
        self.mlp_backbone = nn.Sequential(*layers)
        self.input_dim, self.output_dim = input_dim, output_dim if output_dim is not None else d

    @staticmethod
    def _temporal(per_step: int, steps: int, latent: int, act) -> nn.Module:
        """Per-step linear, then two Conv1d over time, then a linear projection (DexTaG's history/future)."""
        conv = nn.Sequential(nn.Conv1d(60, 40, kernel_size=min(4, steps), stride=2 if steps >= 4 else 1), act(),
                             nn.Conv1d(40, 20, kernel_size=1), act())
        with torch.no_grad():
            n_out = conv(torch.zeros(1, 60, steps)).numel()
        return nn.ModuleDict(dict(pre=nn.Sequential(nn.Linear(per_step, 60), act()), conv=conv,
                                  post=nn.Linear(n_out, latent)))

    @staticmethod
    def _run_temporal(enc: nn.ModuleDict, x: torch.Tensor, steps: int) -> torch.Tensor:
        B = x.shape[0]
        y = enc["pre"](x.reshape(B, steps, -1)).transpose(1, 2)
        return enc["post"](enc["conv"](y).reshape(B, -1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        c, B, i = self.c, obs.shape[0], 0

        def take(n: int) -> torch.Tensor:
            nonlocal i
            x = obs[:, i:i + n]
            i += n
            return x

        motion, proprio = take(c.motion_dim), take(c.proprio_dim)
        feats = [motion, proprio]
        if self.motion_encoder is not None:
            feats.append(self.motion_encoder(motion))
        n_maps = c.tactile_maps + c.target_tactile_maps
        if self.tactile_map_encoder is not None:
            maps = take(n_maps * c.tactile_map_h * c.tactile_map_w).reshape(B, n_maps, c.tactile_map_h, c.tactile_map_w)
            feats.append(self.tactile_map_encoder(maps))
        if self.history_encoder is not None:
            feats.append(self._run_temporal(self.history_encoder, take(c.history_timesteps * c.history_per_step_dim),
                                            c.history_timesteps))
        if self.future_encoder is not None:
            feats.append(self._run_temporal(self.future_encoder, take(c.future_timesteps * c.future_per_step_dim),
                                            c.future_timesteps))
        return self.mlp_backbone(torch.cat(feats, -1))


def create_network(config: NetworkBackboneConfig, input_dim: int, output_dim: int | None = None) -> nn.Module:
    match NetworkBackboneType(config.network_type):
        case NetworkBackboneType.MLP:
            return MLPBackbone(input_dim, config, output_dim)
        case NetworkBackboneType.TEMPORAL_ENCODER:
            return TemporalEncoderBackbone(input_dim, config, output_dim)
    raise ValueError(config.network_type)
