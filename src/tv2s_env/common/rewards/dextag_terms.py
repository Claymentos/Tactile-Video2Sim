"""DexTaG's reward terms (gs_env/common/rewards/hand_imitator_terms.py), made bimanual.

Every state tensor carries a hand axis: (B, H, ...) with H = 2 (left, right). A term computes one value
per hand and averages over the hands it applies to: object terms only over hands whose object target is
valid, tactile terms only over hands whose reference glove reads contact.
Tracking terms are DexTaG's (from ManipTrans); the tactile, contact and smoothness terms are DexTaG's
tactile-as-guidance terms.
"""

import torch
import torch.nn.functional as F

from tv2s_env.common.utils.math_utils import axis_angle_between, quat_angle

from .reward_terms import RewardTerm


def hand_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """(B, H) -> (B,), mean over the hands in `mask` (0 where no hand is)."""
    if mask is None:
        return x.mean(-1)
    m = mask.float()
    return (x * m).sum(-1) / m.sum(-1).clamp(min=1.0)


FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")


class WristPositionTrackingReward(RewardTerm):
    required_keys = ("base_pos", "target_wrist_pos")

    def __init__(self, scale: float = 1.0, k: float = 40.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_pos, target_wrist_pos):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * (target_wrist_pos - base_pos).norm(dim=-1)))


class WristRotationTrackingReward(RewardTerm):
    required_keys = ("base_quat", "target_wrist_quat")

    def __init__(self, scale: float = 1.0, k: float = 2.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_quat, target_wrist_quat):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * quat_angle(base_quat, target_wrist_quat)))


class _PerFingerReward(RewardTerm):
    def __init__(self, scale: float = 1.0, name: str | None = None, **kw: float):
        super().__init__(scale, name)
        self.weights = torch.tensor([kw.get(f"{f}_weight", 0.8) for f in FINGER_KEYS])
        self.ks = torch.tensor([kw.get(f"{f}_k", self.default_k) for f in FINGER_KEYS])

    def _weighted(self, err: torch.Tensor) -> torch.Tensor:
        w, k = self.weights.to(err.device), self.ks.to(err.device)
        return hand_mean((w * torch.exp(-k * err)).sum(-1))


class FingertipPositionTrackingReward(_PerFingerReward):
    """Per-finger weighted exp(-k * tip distance), fingers thumb..pinky."""
    required_keys = ("finger_link_pos", "target_fingertip_pos")
    default_k = 100.0

    def _compute(self, finger_link_pos, target_fingertip_pos):  # type: ignore[override]
        return self._weighted((target_fingertip_pos - finger_link_pos).norm(dim=-1))


class FingertipRotationTrackingReward(_PerFingerReward):
    required_keys = ("finger_link_quat", "target_fingertip_quat")
    default_k = 3.0

    def _compute(self, finger_link_quat, target_fingertip_quat):  # type: ignore[override]
        return self._weighted(quat_angle(finger_link_quat, target_fingertip_quat))


class HandDofPositionTrackingReward(RewardTerm):
    required_keys = ("hand_dof_pos", "target_hand_dof_pos")

    def __init__(self, scale: float = 1.0, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, hand_dof_pos, target_hand_dof_pos):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * (target_hand_dof_pos - hand_dof_pos).abs().mean(-1)))


class ObjectPositionTrackingReward(RewardTerm):
    required_keys = ("object_pos", "target_object_pos", "object_valid")

    def __init__(self, scale: float = 1.0, k: float = 40.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, object_pos, target_object_pos, object_valid):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * (target_object_pos - object_pos).norm(dim=-1)), object_valid)


class ObjectRotationTrackingReward(RewardTerm):
    """The proxy cylinder is symmetric about its axis: rotation about it is ignored (DexTaG's
    `rotation_ignore_axis`), only the axis direction is tracked."""
    required_keys = ("object_axis", "target_object_axis", "object_valid")

    def __init__(self, scale: float = 1.0, k: float = 4.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, object_axis, target_object_axis, object_valid):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * axis_angle_between(object_axis, target_object_axis)), object_valid)


def _normalize_and_gamma(x: torch.Tensor, min_max: float, gamma: float) -> torch.Tensor:
    """Per-map peak normalisation (floored at min_max) and gamma, as DexTaG's rendered tactile maps:
    the comparison is on the spatial contact pattern, not the absolute pressure."""
    peak = x.amax(dim=(-2, -1), keepdim=True).clamp(min=min_max)
    x = (x / peak).clamp(0.0, 1.0)
    return x.pow(gamma) if gamma != 1.0 else x


class WeightedGaussianBlurredTactileMapSimilarityReward(RewardTerm):
    """DexTaG's tactile-map reward: normalised + gamma + 3x3 Gaussian blur, weighted L1 restricted to
    the pixels where the reference glove reads contact, gated by the number of such pixels."""
    required_keys = ("tactile_map", "target_tactile_map", "tactile_pixel_weights")

    def __init__(self, scale: float = 1.0, k: float = 4.0, sigma: float = 1.0, gamma: float = 0.5,
                 min_max: float = 0.3, name: str | None = None):
        super().__init__(scale, name)
        self.k, self.gamma, self.min_max = k, gamma, min_max
        ax = torch.arange(-1, 2, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        self._kernel = (kernel / kernel.sum()).reshape(1, 1, 3, 3)

    def similarity(self, cur: torch.Tensor, tgt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """cur, tgt (B, H, h, w) -> (B, H) in [0, 1]."""
        if self._kernel.device != cur.device:
            self._kernel = self._kernel.to(cur.device)
        B, H, h, w = cur.shape
        mask = (tgt > 0).reshape(B, H, -1)
        contact_scale = (mask.sum(-1).float() / 10.0).clamp(0.0, 1.0)
        blur = lambda x: F.conv2d(_normalize_and_gamma(x.reshape(B * H, 1, h, w), self.min_max, self.gamma),
                                  self._kernel, padding=1).reshape(B, H, -1)
        diff = (blur(tgt) - blur(cur)).abs()
        wts = weights.reshape(1, H, -1) * mask
        error = (diff * wts).sum(-1) / wts.sum(-1).clamp(min=1e-6)
        return torch.exp(-self.k * error) * contact_scale

    def _compute(self, tactile_map, target_tactile_map, tactile_pixel_weights):  # type: ignore[override]
        sim = self.similarity(tactile_map, target_tactile_map, tactile_pixel_weights)
        has_contact = (target_tactile_map > 0).flatten(2).any(-1)
        return hand_mean(sim, has_contact)


class TactileGatedFingertipSurfaceProximityReward(RewardTerm):
    """Per finger: exp(-k * max(sdf, 0)) for fingers whose reference glove patch reads contact."""
    required_keys = ("fingertip_to_surface_dist", "target_finger_contact")

    def __init__(self, scale: float = 1.0, k: float = 50.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, fingertip_to_surface_dist, target_finger_contact):  # type: ignore[override]
        prox = torch.exp(-self.k * fingertip_to_surface_dist.clamp(min=0.0))
        return (prox * target_finger_contact.float()).sum(-1).mean(-1)


class SupportContactPenalty(RewardTerm):
    """DexTaG's TableContactPenalty, on each hand's support surface: exp(-k * max(force - threshold, 0))."""
    required_keys = ("support_contact_force",)

    def __init__(self, scale: float = 1.0, k: float = 0.005, threshold: float = 0.0, name: str | None = None):
        super().__init__(scale, name)
        self.k, self.threshold = k, threshold

    def _compute(self, support_contact_force):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * (support_contact_force - self.threshold).clamp(min=0.0)))


class _VelocityChangePenalty(RewardTerm):
    def __init__(self, scale: float = 1.0, k: float = 2.0, threshold: float = 0.05,
                 height_threshold: float = 0.05, name: str | None = None):
        super().__init__(scale, name)
        self.k, self.threshold, self.height_threshold = k, threshold, height_threshold

    def _penalty(self, change, height, valid):
        p = torch.exp(-self.k * (change - self.threshold).clamp(min=0.0)) - 1.0
        return (p * (height > self.height_threshold).float() * valid.float()).sum(-1)


class ObjectVelocityChangePenalty(_VelocityChangePenalty):
    required_keys = ("object_vel_change", "object_height_above_support", "object_valid")

    def _compute(self, object_vel_change, object_height_above_support, object_valid):  # type: ignore[override]
        return self._penalty(object_vel_change, object_height_above_support, object_valid)


class ObjectAngularVelocityChangePenalty(_VelocityChangePenalty):
    required_keys = ("object_ang_vel_change", "object_height_above_support", "object_valid")

    def _compute(self, object_ang_vel_change, object_height_above_support, object_valid):  # type: ignore[override]
        return self._penalty(object_ang_vel_change, object_height_above_support, object_valid)


class AllDOFPowerPenalty(RewardTerm):
    required_keys = ("dof_force", "dof_vel")

    def __init__(self, scale: float = 1.0, k: float = 10.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, dof_force, dof_vel):  # type: ignore[override]
        return hand_mean(torch.exp(-self.k * (dof_force * dof_vel).abs().sum(-1)))
