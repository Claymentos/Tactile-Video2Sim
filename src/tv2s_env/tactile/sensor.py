"""Simulated glove: taxel points on the Wuji links read against the object's signed distance field.

  distance  d_p = SDF_object(taxel p)                         (Benchmark-Genesis' ContactDepthProbe)
  pressure  f_p = kn * max(probe_radius - d_p, 0)             (a penalty spring)
Per glove channel: the smallest distance and the largest pressure of its taxels.
"""

import torch

from tv2s_env.tactile.mapping import TaxelSet


class GloveSensor:
    """One hand's simulated glove for a batch of envs."""

    def __init__(self, taxels: TaxelSet, link_index: dict[str, int], probe_radius: float, kn: float,
                 device: torch.device):
        self.device = device
        self.probe_radius = probe_radius
        self.kn = kn
        self.n_channels = taxels.n_channels
        self.link_ids = torch.as_tensor([link_index[l] for l in taxels.link], device=device)
        self.pos = torch.as_tensor(taxels.pos, dtype=torch.float32, device=device)            # (P, 3)
        self.channel = torch.as_tensor(taxels.channel, dtype=torch.long, device=device)       # (P,)

    def taxel_world(self, link_pos: torch.Tensor, link_rot: torch.Tensor) -> torch.Tensor:
        """link_pos (N, L, 3), link_rot (N, L, 3, 3) -> taxel positions (N, P, 3)."""
        R = link_rot[:, self.link_ids]
        return (R @ self.pos[None, :, :, None])[..., 0] + link_pos[:, self.link_ids]

    def reduce(self, d: torch.Tensor):
        """Per-taxel distances (N, P) -> per-channel (distance, pressure), each (N, C)."""
        f = self.kn * (self.probe_radius - d).clamp(min=0)
        N = d.shape[0]
        idx = self.channel.expand(N, -1)
        dist = torch.full((N, self.n_channels), float("inf"), device=self.device).scatter_reduce(1, idx, d, "amin")
        pressure = torch.zeros((N, self.n_channels), device=self.device).scatter_reduce(1, idx, f, "amax")
        return dist, pressure
