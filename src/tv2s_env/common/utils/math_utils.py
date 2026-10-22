"""Rotation helpers (torch, quaternions wxyz). Subset of DexTaG's math_utils plus Benchmark-Genesis'
rotation-matrix helpers (rl_recon/common.py)."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> axis-angle (..., 3); stable away from pi."""
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return v * (angle / (2 * torch.sin(angle)).clamp_min(1e-6))[..., None]


def rotation_angle(R: torch.Tensor) -> torch.Tensor:
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0)
    return torch.acos(cos)


# ---------------------------------------------------------------- numpy, for reference preprocessing
