"""Benchmark-Genesis' rl_recon reward (rl_recon/env.py: ReconEnv.metrics), as bimanual reward terms.

    r = w_tac * r_tactile + w_hand * r_hand + w_obj * (1 - p_position) + w_obj_axis * (1 - p_axis)
        - w_rate * |a_t - a_{t-1}|^2

Every term is bounded and the total stays positive, so an early termination always costs reward.
Hands are averaged; object terms count only where the object target is valid (p = 0 elsewhere, as
rl_recon's `obj_valid`).
"""

import torch

from tv2s_env.common.utils.math_utils import rotation_angle

from .dextag_terms import hand_mean
from .reward_terms import RewardTerm


class TactileContactF1Reward(RewardTerm):
    """(1 - w) * soft contact F1 against the glove + w * reach shaping pulling the glove's contact
    taxels towards the object surface. Soft contact: sigmoid((probe_radius - d) / tau)."""
    required_keys = ("tactile_distance", "target_contact")

    def __init__(self, scale: float = 1.0, probe_radius: float = 0.004, soft_tau: float = 0.0015,
                 reach_weight: float = 0.5, reach_scale: float = 0.02, name: str | None = None):
        super().__init__(scale, name)
        self.r, self.tau, self.w, self.reach_scale = probe_radius, soft_tau, reach_weight, reach_scale

    def _compute(self, tactile_distance, target_contact):  # type: ignore[override]
        y = target_contact.float()
        soft = torch.sigmoid((self.r - tactile_distance) / self.tau)
        f1 = (2 * (soft * y).sum(-1) + 1) / (soft.sum(-1) + y.sum(-1) + 1)
        n_y = y.sum(-1)
        reach = (torch.exp(-(tactile_distance - self.r).clamp(min=0) / self.reach_scale) * y).sum(-1) / n_y.clamp(min=1)
        reach = torch.where(n_y > 0, reach, torch.ones_like(reach))
        return hand_mean((1 - self.w) * f1 + self.w * reach)


class HandTrackingKernelReward(RewardTerm):
    """0.4 exp(-e_palm / s_pos) + 0.2 exp(-e_rot / s_rot) + 0.4 exp(-e_keypoints / s_kp)."""
    required_keys = ("base_pos", "target_wrist_pos", "base_rot", "target_wrist_rot", "keypoints", "target_keypoints")

    def __init__(self, scale: float = 1.0, sigma_pos: float = 0.02, sigma_rot: float = 0.25,
                 sigma_kp: float = 0.03, name: str | None = None):
        super().__init__(scale, name)
        self.s_pos, self.s_rot, self.s_kp = sigma_pos, sigma_rot, sigma_kp

    def _compute(self, base_pos, target_wrist_pos, base_rot, target_wrist_rot, keypoints, target_keypoints):  # type: ignore[override]
        e_pos = (base_pos - target_wrist_pos).norm(dim=-1)
        e_rot = rotation_angle(base_rot.transpose(-1, -2) @ target_wrist_rot)
        e_kp = (keypoints - target_keypoints).norm(dim=-1).mean(-1)
        r = 0.4 * torch.exp(-e_pos / self.s_pos) + 0.2 * torch.exp(-e_rot / self.s_rot) + 0.4 * torch.exp(-e_kp / self.s_kp)
        return hand_mean(r)


class ObjectPositionHingeReward(RewardTerm):
    """1 - p_position, p = 1 - exp(-hinge): the error across the head-camera ray has a small margin,
    the error along it (depth, the weak axis of a video-based object track) a large one."""
    required_keys = ("object_pos", "target_object_pos", "camera_ray", "object_valid")

    def __init__(self, scale: float = 1.0, margin_across: float = 0.02, scale_across: float = 0.05,
                 margin_along: float = 0.10, scale_along: float = 0.10, name: str | None = None):
        super().__init__(scale, name)
        self.ma, self.sa, self.ml, self.sl = margin_across, scale_across, margin_along, scale_along

    def _compute(self, object_pos, target_object_pos, camera_ray, object_valid):  # type: ignore[override]
        err = object_pos - target_object_pos
        along = (err * camera_ray).sum(-1)
        across = (err - along[..., None] * camera_ray).norm(dim=-1)
        hinge = (across - self.ma).clamp(min=0) / self.sa + (along.abs() - self.ml).clamp(min=0) / self.sl
        p = object_valid.float() * (1 - torch.exp(-hinge))
        return hand_mean(1 - p)


class ObjectAxisHingeReward(RewardTerm):
    """1 - p_axis on the object's long axis as seen from the head camera: both axes projected across the
    viewing ray (direction and foreshortening), compared as an angle, 15 deg free. Unsigned, since the
    proxy cylinder has no tip (rl_recon signs it with CoTracker for a screwdriver)."""
    required_keys = ("object_axis", "target_object_axis", "camera_ray", "object_valid")

    def __init__(self, scale: float = 1.0, margin: float = 0.26, axis_scale: float = 0.5, name: str | None = None):
        super().__init__(scale, name)
        self.margin, self.axis_scale = margin, axis_scale

    def _compute(self, object_axis, target_object_axis, camera_ray, object_valid):  # type: ignore[override]
        proj = lambda a: a - (a * camera_ray).sum(-1, keepdim=True) * camera_ray
        a, b = proj(object_axis), proj(target_object_axis)
        err = torch.minimum((a - b).norm(dim=-1), (a + b).norm(dim=-1))
        e_axis = 2 * torch.asin((err / 2).clamp(max=1.0))
        p = object_valid.float() * (1 - torch.exp(-(e_axis - self.margin).clamp(min=0) / self.axis_scale))
        return hand_mean(1 - p)


class ActionRatePenalty(RewardTerm):
    """-|a_t - a_{t-1}|^2 (mean over action dims, clamped at 1)."""
    required_keys = ("action_rate",)

    def _compute(self, action_rate):  # type: ignore[override]
        return -action_rate
