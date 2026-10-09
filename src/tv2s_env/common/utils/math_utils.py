"""Rotation helpers (torch, quaternions wxyz). Subset of DexTaG's math_utils plus Benchmark-Genesis'
rotation-matrix helpers (rl_recon/common.py)."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def quat_angle(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rotation angle between two quaternion batches (sign invariant)."""
    dot = (q * r).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(dot)


def matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> axis-angle (..., 3); stable away from pi."""
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return v * (angle / (2 * torch.sin(angle)).clamp_min(1e-6))[..., None]


def rotation_angle(R: torch.Tensor) -> torch.Tensor:
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0)
    return torch.acos(cos)


def axis_angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between two unsigned axes (a symmetric object's axis has no direction)."""
    return torch.acos((a * b).sum(-1).abs().clamp(max=1.0))


# ---------------------------------------------------------------- numpy, for reference preprocessing
def np_velocity(x: np.ndarray, dt: float, sigma: float = 2.0) -> np.ndarray:
    """Finite-difference velocity along axis 0, Gaussian-smoothed (DexTaG / ManipTrans' compute_velocity)."""
    v = np.gradient(x, axis=0) / dt
    return gaussian_filter1d(v, sigma, axis=0, mode="nearest") if sigma > 0 else v


def np_angular_velocity(R: np.ndarray, dt: float, sigma: float = 2.0) -> np.ndarray:
    """(T, ..., 3, 3) rotations -> world angular velocity (T, ..., 3)."""
    shape = R.shape[:-2]
    d = np.einsum("t...ij,t...kj->t...ik", R[1:], R[:-1])
    w = Rotation.from_matrix(d.reshape(-1, 3, 3)).as_rotvec().reshape((shape[0] - 1,) + shape[1:] + (3,)) / dt
    w = np.concatenate([w, w[-1:]], 0)
    return gaussian_filter1d(w, sigma, axis=0, mode="nearest") if sigma > 0 else w


def np_quat_continuity(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs along axis 0 so consecutive ones share a hemisphere."""
    q = q.copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    return q
