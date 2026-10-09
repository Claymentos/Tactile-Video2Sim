"""Gaussian actor and state-value critic (DexTaG's gs_agent/modules/{policies,critics}.py).

Both take an optional shared input normaliser (Benchmark-Genesis normalises observations; DexTaG
does not)."""

import torch
from torch import nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(self, backbone: nn.Module, action_dim: int, initial_log_std: float = 0.0,
                 output_init_scale: float | None = None, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.mu = nn.Linear(backbone.output_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        if output_init_scale is None:
            nn.init.xavier_uniform_(self.mu.weight)
        else:
            # rl_recon: a near-zero last layer, so training starts from the zero residual (the reference)
            nn.init.uniform_(self.mu.weight, -output_init_scale, output_init_scale)
        nn.init.zeros_(self.mu.bias)
        Normal.set_default_validate_args(False)

    def dist_from_obs(self, obs: torch.Tensor) -> Normal:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        mean = self.mu(self.backbone(obs))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def forward(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist_from_obs(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1, keepdim=True)

    def forward_with_dist_params(self, obs: torch.Tensor):
        dist = self.dist_from_obs(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1, keepdim=True), dist.mean, dist.stddev

    def evaluate_log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.dist_from_obs(obs).log_prob(act).sum(-1, keepdim=True)

    @property
    def action_std(self) -> torch.Tensor:
        return self.log_std.exp()


class StateValueFunction(nn.Module):
    def __init__(self, backbone: nn.Module, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.normalizer = normalizer
        self.v = nn.Linear(backbone.output_dim, 1) if backbone.output_dim != 1 else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.v(self.backbone(obs))
