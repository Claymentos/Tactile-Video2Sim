"""Simulated glove: taxel points on the Wuji links read against the object's signed distance field.

The DexTaG Genesis fork's TactileField sensor and TactileVisualizer are not in stock Genesis, so the
tactile reading is computed here, in torch, from the link and object poses:
  distance  d_p = SDF_object(taxel p)                         (Benchmark-Genesis' ContactDepthProbe)
  pressure  f_p = kn * max(probe_radius - d_p, 0)             (a penalty spring, as DexTaG's TactileField)
Per glove channel: the smallest distance and the largest pressure of its taxels (DexTaG's visualizer
shows a pixel as its strongest point). `to_image` lays channels out on the glove grid (HumanTouch: 23x20).
"""

import numpy as np
import torch

from tv2s_data.glove import GloveLayout
from tv2s_env.tactile.mapping import TaxelSet


def cylinder_sdf(p: torch.Tensor, radius: torch.Tensor, half_length: torch.Tensor) -> torch.Tensor:
    """Exact signed distance to a capped cylinder along z, centred at the origin.
    p (N, P, 3) object frame; radius, half_length (N,) -> (N, P)."""
    d_r = torch.linalg.norm(p[..., :2], dim=-1) - radius[:, None]
    d_z = p[..., 2].abs() - half_length[:, None]
    outside = torch.sqrt(d_r.clamp(min=0) ** 2 + d_z.clamp(min=0) ** 2)
    inside = torch.maximum(d_r, d_z).clamp(max=0)
    return outside + inside


class GloveSensor:
    """One hand's simulated glove for a batch of envs."""

    def __init__(self, taxels: TaxelSet, layout: GloveLayout, link_index: dict[str, int],
                 probe_radius: float, kn: float, device: torch.device):
        assert taxels.n_channels == layout.n_channels
        self.device = device
        self.probe_radius = probe_radius
        self.kn = kn
        self.n_channels = layout.n_channels
        self.link_ids = torch.as_tensor([link_index[l] for l in taxels.link], device=device)
        self.pos = torch.as_tensor(taxels.pos, dtype=torch.float32, device=device)            # (P, 3)
        self.channel = torch.as_tensor(taxels.channel, dtype=torch.long, device=device)       # (P,)
        self.rows = torch.as_tensor(layout.rows, device=device)
        self.cols = torch.as_tensor(layout.cols, device=device)
        self.grid_shape = tuple(layout.grid_shape)
        self.region = layout.region
        # per-channel taxel index lists are scatter targets; fingertip masks (row i = finger i)
        self.finger_masks = torch.stack([torch.as_tensor(layout.region == r, device=device)
                                         for r in ("thumb", "index", "middle", "ring", "pinky")])
        # DexTaG's pixel weights: fingertip cells (the distal third of a finger patch) count 3x
        tip = torch.as_tensor((layout.region != "palm") & (layout.u > 2 / 3), device=device)
        w = torch.where(tip, 3.0, 1.0)
        self.channel_weights = w * len(w) / w.sum()

    def taxel_world(self, link_pos: torch.Tensor, link_rot: torch.Tensor) -> torch.Tensor:
        """link_pos (N, L, 3), link_rot (N, L, 3, 3) -> taxel positions (N, P, 3)."""
        R = link_rot[:, self.link_ids]
        return (R @ self.pos[None, :, :, None])[..., 0] + link_pos[:, self.link_ids]

    def read(self, link_pos, link_rot, obj_pos, obj_rot, radius, half_length):
        """Per-channel (distance, pressure), each (N, C)."""
        p = self.taxel_world(link_pos, link_rot)
        p_obj = ((p - obj_pos[:, None])[..., None, :] @ obj_rot[:, None])[..., 0, :]
        return self.reduce(cylinder_sdf(p_obj, radius, half_length))

    def reduce(self, d: torch.Tensor):
        """Per-taxel distances (N, P) -> per-channel (distance, pressure), each (N, C)."""
        f = self.kn * (self.probe_radius - d).clamp(min=0)
        N = d.shape[0]
        idx = self.channel.expand(N, -1)
        dist = torch.full((N, self.n_channels), float("inf"), device=self.device).scatter_reduce(1, idx, d, "amin")
        pressure = torch.zeros((N, self.n_channels), device=self.device).scatter_reduce(1, idx, f, "amax")
        return dist, pressure

    def to_image(self, values: torch.Tensor) -> torch.Tensor:
        """(N, C) channel values -> (N, rows, cols) glove image (HumanTouch: 23 x 20)."""
        img = torch.zeros((values.shape[0],) + self.grid_shape, device=values.device, dtype=values.dtype)
        img[:, self.rows, self.cols] = values
        return img


def fingertip_taxel_centres(taxels: TaxelSet, layout: GloveLayout) -> dict[str, np.ndarray]:
    """Mean link-frame position of the distal taxels of each finger (DexTaG's fingertip tactile centre)."""
    out = {}
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        m = (layout.region[taxels.channel] == f) & (layout.u[taxels.channel] > 2 / 3)
        links = taxels.link[m]
        link = max(set(links), key=list(links).count)
        out[f] = (link, taxels.pos[m][links == link].mean(0))
    return out
