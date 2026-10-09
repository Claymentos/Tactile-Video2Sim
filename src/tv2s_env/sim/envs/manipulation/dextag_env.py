"""DexTaG's retargeter (teacher) task on the bimanual HumanTouch scene.

Follows DexTaG's SingleHandRetargetingEnvTactileMap, per hand:
  - Action (26 per hand): 6 virtual-base deltas (DexTaG: 7 arm joint deltas) followed by 20 absolute
    finger targets normalised to [-1, 1] over the joint limits, with EMA smoothing and a random
    1-step latency drawn from the action history.
  - Observations are selected by name from the config: the reference motion at t+1 and K future frames,
    proprioception, the simulated glove image of both hands (the critic also sees the reference glove
    image, the action/proprio history and the last reward).
  - Rewards: DexTaG's tracking terms plus its tactile-as-guidance terms (tactile-map similarity,
    tactile-gated fingertip proximity), contact and smoothness penalties.
  - Episodes run the whole clip from frame 0; they end on success (clip end) or on failure (fingertip,
    object or support-contact failure). Training aids: the decaying object guide force and random
    perturbation kicks.
"""

import torch

from tv2s_env.common.utils.math_utils import axis_angle_between
from tv2s_env.sim.envs.config.schema import DexTaGEnvArgs
from tv2s_env.sim.envs.manipulation.bimanual_tactile_env import H, BimanualTactileEnv

# reference terms observed for the current and the future frames (B, H, ...) -> flattened
MOTION_TERM_KEYS = {
    "target_hand_dof_pos": "dof_pos", "target_hand_dof_vel": "dof_vel", "target_fingertip_quat": "tip_quat",
    "target_wrist_quat": "wrist_quat", "target_wrist_vel": "wrist_vel", "target_wrist_ang_vel": "wrist_ang_vel",
    "target_object_quat": "obj_quat", "target_object_vel": "obj_vel", "target_object_ang_vel": "obj_ang_vel",
}
DELTA_TERM_KEYS = {"delta_fingertip_pos": ("tip_pos", "finger_link_pos"),
                   "delta_wrist_pos": ("wrist_pos", "base_pos"),
                   "delta_object_pos": ("obj_pos", "object_pos")}


class DexTaGBimanualEnv(BimanualTactileEnv):
    _args: DexTaGEnvArgs

    def __init__(self, args: DexTaGEnvArgs, num_envs: int, show_viewer: bool = False,
                 device: torch.device = torch.device("cuda"), eval_mode: bool = False) -> None:
        super().__init__(args, num_envs, show_viewer, device, eval_mode)

    # ------------------------------------------------------------------ buffers
    def _init_buffers(self) -> None:
        super()._init_buffers()
        a, B, dev = self._args, self._num_envs, self.device
        self.action_dim = H * 26
        self.action_history_buf = torch.zeros(B, a.obs_history_len, self.action_dim, device=dev)
        scale = [a.base_action_scale_pos] * 3 + [a.base_action_scale_rot] * 3
        self._base_scale = torch.tensor(scale, device=dev)
        self.update_buffers()
        self.proprio_dim = self._concat(a.proprio_terms).shape[-1]
        self.motion_dim = self._motion(self.t + 1, a.target_motion_terms).shape[-1]
        self.proprio_history_buf = torch.zeros(B, a.obs_history_len, self.proprio_dim, device=dev)
        self.disable_termination = False

    def _concat(self, terms: list[str]) -> torch.Tensor:
        s = self._args.obs_scales
        return torch.cat([getattr(self, k).reshape(self._num_envs, -1) * s.get(k, 1.0) for k in terms], -1)

    def _motion(self, t: torch.Tensor, terms: list[str]) -> torch.Tensor:
        """Reference motion terms at frames t (B,), relative ones against the current state."""
        s, parts = self._args.obs_scales, []
        for k in terms:
            if k in MOTION_TERM_KEYS:
                x = self.ref_at(MOTION_TERM_KEYS[k], t)
            elif k in DELTA_TERM_KEYS:
                ref_key, cur = DELTA_TERM_KEYS[k]
                x = self.ref_at(ref_key, t) - getattr(self, cur)
            elif k == "object_valid":
                x = (self.ref_at("obj_valid", t) & self.obj_active_t).float()
            else:
                raise KeyError(f"unknown target motion term {k}")
            parts.append(x.reshape(self._num_envs, -1) * s.get(k, 1.0))
        return torch.cat(parts, -1)

    def update_buffers(self) -> None:
        super().update_buffers()
        self.fingertip_to_surface_dist = self.fingertip_to_surface_dist.clamp(max=0.2)

    # ------------------------------------------------------------------ observations
    def obs_layout(self, role: str) -> dict[str, int]:
        """Dims of the slots of TemporalEncoderBackbone for the actor or the critic."""
        a = self._args
        terms = a.actor_obs_terms if role == "actor" else a.critic_obs_terms
        n_img = self.tactile_map.shape[-1] * self.tactile_map.shape[-2]
        layout = dict(motion_dim=self.motion_dim if "target_motion" in terms else 0,
                      proprio_dim=self.proprio_dim + (1 if "last_reward" in terms else 0),
                      tactile_maps=H if a.tactile_map_term in terms else 0,
                      target_tactile_maps=H if "target_tactile_map" in terms else 0,
                      tactile_map_h=self.tactile_map.shape[-2], tactile_map_w=self.tactile_map.shape[-1],
                      history_timesteps=a.obs_history_len if "action_proprio_history_flat" in terms else 0,
                      history_per_step_dim=self.action_dim + self.proprio_dim,
                      future_timesteps=a.obs_future_length if "target_action_flat" in terms else 0,
                      future_per_step_dim=self.motion_dim)
        layout["total"] = (layout["motion_dim"] + layout["proprio_dim"]
                           + (layout["tactile_maps"] + layout["target_tactile_maps"]) * n_img
                           + layout["history_timesteps"] * layout["history_per_step_dim"]
                           + layout["future_timesteps"] * layout["future_per_step_dim"])
        return layout

    def _build_obs(self, terms: list[str]) -> torch.Tensor:
        a, B = self._args, self._num_envs
        parts = []
        proprio = self._concat(a.proprio_terms)
        if "target_motion" in terms:
            parts.append(self._motion(self.t + 1, a.target_motion_terms))
        parts.append(proprio)
        if "last_reward" in terms:
            parts.append(self.last_reward)
        if a.tactile_map_term in terms:
            parts.append(self.tactile_map.reshape(B, -1) * a.obs_scales.get("tactile_map", 1.0))
        if "target_tactile_map" in terms:
            parts.append(self.target_tactile_map.reshape(B, -1) * a.obs_scales.get("tactile_map", 1.0))
        if "action_proprio_history_flat" in terms:
            parts.append(torch.cat([self.action_history_buf, self.proprio_history_buf], -1).reshape(B, -1))
        if "target_action_flat" in terms:
            K, st = a.obs_future_length, a.obs_future_stride
            parts += [self._motion(self.t + 1 + st * (k + 1), a.target_motion_terms) for k in range(K)]
        return torch.cat(parts, -1)

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.proprio_history_buf = torch.cat([self.proprio_history_buf[:, 1:],
                                              self._concat(self._args.proprio_terms)[:, None]], 1)
        return self._build_obs(self._args.actor_obs_terms), self._build_obs(self._args.critic_obs_terms)

    # ------------------------------------------------------------------ actions
    def apply_action(self, action: torch.Tensor) -> None:
        a, B = self._args, self._num_envs
        action = action.detach().to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], action[:, None]], 1)
        lo, hi = a.action_latency_range
        latency = torch.randint(lo, hi + 1, (B,), device=self.device, generator=self.generator)
        exec_action = self.action_history_buf[torch.arange(B, device=self.device), -1 - latency].reshape(B, H, 26)
        targets = []
        for h, hand in enumerate(self.hands):
            base = self.base_dof_pos[:, h] + exec_action[:, h, :6] * self._base_scale
            fingers = (exec_action[:, h, 6:] + 1) / 2 * (hand.q_high - hand.q_low) + hand.q_low
            fingers = a.action_ema * fingers + (1 - a.action_ema) * self.hand_dof_pos[:, h]
            targets.append(torch.cat([base, fingers], -1))
        self._physics_step(targets)

    def _apply_object_aids(self) -> None:
        a, o = self._args, self._args.objects
        if self._eval_mode:
            return
        t1 = self.t + 1
        if a.object_guide_decay_steps > 0 and self.training_step < a.object_guide_decay_steps:
            decay = 1.0 - self.training_step / a.object_guide_decay_steps
            ref_pos = self.ref_at("obj_pos", t1)
            valid = (self.ref_at("obj_valid", t1) & self.obj_active_t).float()
            for h, obj in enumerate(self.objects):
                p, v, m = obj.get_pos(), obj.get_vel(), self.obj_mass[:, h:h + 1]
                f = m * (a.object_guide_kp * (ref_pos[:, h] - p) - a.object_guide_kd * v)
                f[:, 2] += m[:, 0] * 9.81
                self._apply_object_wrench(h, f * (decay * valid[:, h:h + 1]))
        if o.perturbation_force_scale > 0 or o.perturbation_torque_scale > 0:
            c0, c1 = o.perturbation_curriculum
            if self.training_step < c0:
                return
            ramp = min((self.training_step - c0) / max(c1 - c0, 1), 1.0)
            if not hasattr(self, "_kick_prob"):
                lo, hi = torch.log(torch.tensor(o.perturbation_prob_range))
                self._kick_prob = torch.exp(lo + (hi - lo) * torch.rand(self._num_envs, H, device=self.device))
            lifted = self.object_height_above_support > o.perturbation_lift_threshold
            kick = ((torch.rand(self._num_envs, H, device=self.device) < self._kick_prob) & lifted).float()
            for h in range(H):
                m = self.obj_mass[:, h:h + 1] * kick[:, h:h + 1] * ramp
                f = torch.randn(self._num_envs, 3, device=self.device) * m * o.perturbation_force_scale
                tq = torch.randn(self._num_envs, 3, device=self.device) * m * o.perturbation_torque_scale
                self._apply_object_wrench(h, f, tq)

    # ------------------------------------------------------------------ episodes
    def get_terminated(self) -> torch.Tensor:
        a = self._args
        error = ((self.base_lin_vel.norm(dim=-1) > 100) | (self.base_ang_vel.norm(dim=-1) > 200)
                 | (self.object_lin_vel.norm(dim=-1) > 100) | (self.object_ang_vel.norm(dim=-1) > 200)
                 | ~torch.isfinite(self.object_pos).all(-1) | ~torch.isfinite(self.base_pos).all(-1)).any(-1)
        tip_err = (self.finger_link_pos - self.target_fingertip_pos).norm(dim=-1)              # (B, H, 5)
        hand_failure = (tip_err > torch.tensor(a.fingertip_failure_dist, device=self.device)).any(-1).any(-1)
        obj_pos_fail = (self.object_pos - self.target_object_pos).norm(dim=-1) > a.object_pos_failure
        obj_rot_fail = torch.rad2deg(axis_angle_between(self.object_axis, self.target_object_axis)) > a.object_rot_failure_deg
        object_failure = ((obj_pos_fail | obj_rot_fail) & self.object_valid).any(-1)
        s, (c0, c1, c2) = self.training_step, a.support_contact_thresholds
        thr = c0 - (c0 - c1) * min(s / 10000, 1.0) if s < 10000 else c1 - (c1 - c2) * min((s - 10000) / 10000, 1.0)
        support_failure = (self.support_contact_force > thr).any(-1)
        failed = hand_failure | object_failure | support_failure | error
        if self.disable_termination:
            failed = error
        succeeded = (self.t + a.success_lookahead >= self.horizon) & ~failed
        self._extra_info["termination"] = dict(
            any=failed | succeeded, succeeded=succeeded, failed=failed, error=error, hand_failure=hand_failure,
            object_failure=object_failure, support_failure=support_failure)
        self.reset_buf = failed | succeeded
        return self.reset_buf

    def get_truncated(self) -> torch.Tensor:
        """Episodes end on success or failure, never on a time limit (DexTaG)."""
        return torch.zeros_like(self.reset_buf)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        start = torch.zeros(len(envs_idx), dtype=torch.long, device=self.device)
        self._reset_hands_and_objects(envs_idx, start, torch.zeros(len(envs_idx), H, dtype=torch.bool, device=self.device))
        self.t[envs_idx] = start
        self.start[envs_idx] = start
        self.action_history_buf[envs_idx] = 0.0
        if hasattr(self, "proprio_history_buf"):
            self.proprio_history_buf[envs_idx] = 0.0
        self.update_buffers()
