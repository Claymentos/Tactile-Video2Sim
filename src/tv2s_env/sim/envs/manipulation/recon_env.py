"""Benchmark-Genesis' RL reconstruction task (reconstruction/rl_recon/env.py) on the bimanual HumanTouch scene.

Follows ReconEnv, per hand:
  - Action (26 per hand, in [-1, 1]): a residual on the reference at t+1: base x, y, z (x 3 cm), base
    Euler (x 0.25 rad), 20 fingers (x 0.4 rad). The policy starts on the retargeted reference.
  - Observation: phase; finger and base joint state; the reference 1, 2, 4 frames ahead (relative); the
    object in the palm frame, its target and its long axis; the simulated glove (soft contact and
    distance per channel); the reference glove contacts 1 and 5 frames ahead; keypoint errors; the last
    action.
  - Reward: soft contact F1 + reach (tactile), hand kernels, object position / axis hinges, action rate.
  - Episodes: from frame 0, from a random frame before the first grasp, or (reference state
    initialisation) inside a grasp with the object in the reference hand; terminated when the palm,
    the keypoints or the object drift too far, truncated at the clip end.
  - Training aid: a velocity-level controller holding each object at its in-palm pose while the glove
    reports the grasp (DexMachina's virtual object controller anchored to the hand), scaled by
    `assist_scale` which the runner decays to 0.
"""

import torch

from tv2s_env.common.utils.math_utils import matrix_to_rotvec
from tv2s_env.sim.envs.config.schema import ReconEnvArgs
from tv2s_env.sim.envs.manipulation.bimanual_tactile_env import H, BimanualTactileEnv


class ReconBimanualEnv(BimanualTactileEnv):
    _args: ReconEnvArgs

    def __init__(self, args: ReconEnvArgs, num_envs: int, show_viewer: bool = False,
                 device: torch.device = torch.device("cuda"), eval_mode: bool = False,
                 render: dict | None = None) -> None:
        self.assist_scale = 0.0
        self.random_start = not eval_mode
        self.disable_termination = False
        self.in_hand_start = False
        super().__init__(args, num_envs, show_viewer, device, eval_mode, render)

    def _init_buffers(self) -> None:
        super()._init_buffers()
        a, dev = self._args, self.device
        self.action_dim = H * 26
        self.last_action = torch.zeros(self._num_envs, self.action_dim, device=dev)
        self.action_rate = torch.zeros(self._num_envs, device=dev)
        self.action_scale = torch.tensor([a.action_scale_pos] * 3 + [a.action_scale_rot] * 3
                                         + [a.action_scale_finger] * 20, device=dev).repeat(H)
        # first grasp onset of each clip (random starts stop before it), and its grasp frames (RSI)
        g0 = torch.as_tensor(self.grasp[..., 0], device=dev).float()
        g0[~torch.as_tensor(self.obj_active, device=dev)] = float("inf")
        first = g0.min(-1).values
        self.first_grasp = torch.where(torch.isfinite(first), first, torch.zeros_like(first)).long()
        self.holding_any = self.ref["holding"].any(-1)                                   # (n_clips, T)

    # ------------------------------------------------------------------ observations
    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        a, B, t = self._args, self._num_envs, self.t
        T = self.lengths_t[self.env_clip].float()
        parts = [(t.float() / T)[:, None], self.hand_dof_pos.reshape(B, -1), 0.1 * self.hand_dof_vel.reshape(B, -1),
                 0.1 * self.base_dof_vel.reshape(B, -1)]
        for k in a.obs_future:
            tk = t + k
            parts += [(self.ref_at("wrist_pos", tk) - self.base_pos).reshape(B, -1),
                      matrix_to_rotvec(self.base_rot.transpose(-1, -2) @ self.ref_at("wrist_rot", tk)).reshape(B, -1),
                      (self.ref_at("dof_pos", tk) - self.hand_dof_pos).reshape(B, -1)]
        rel = ((self.object_pos - self.base_pos)[..., None, :] @ self.base_rot)[..., 0, :]
        oR_p = self.base_rot.transpose(-1, -2) @ self.object_rot
        parts += [rel.reshape(B, -1), oR_p[..., :2].reshape(B, -1), self.object_lin_vel.reshape(B, -1),
                  0.1 * self.object_ang_vel.reshape(B, -1), (self.ref_at("obj_pos", t + 1) - self.object_pos).reshape(B, -1),
                  self.object_pos[..., 2:].reshape(B, -1) - self.support_z_t,
                  self.ref_at("obj_rot", t + 1)[..., 2].reshape(B, -1), self.object_axis.reshape(B, -1),
                  self.object_valid.float()]
        c = a.tactile
        d = self.tactile_distance
        parts += [torch.sigmoid((c.probe_radius - d) / c.soft_tau).reshape(B, -1), torch.tanh(d / 0.03).reshape(B, -1)]
        glove = 1.0 if a.glove_obs else 0.0
        for k in a.obs_tactile_future:
            parts.append(glove * self.ref_at("contact", t + k).reshape(B, -1))
        parts += [(self.ref_at("keypoints", t + 1) - self.keypoints).reshape(B, -1), self.last_action]
        obs = torch.cat(parts, -1)
        return obs, obs

    # ------------------------------------------------------------------ actions
    def apply_action(self, action: torch.Tensor) -> None:
        action = action.detach().to(self.device).clamp(-1, 1)
        self.action_rate = ((action - self.last_action) ** 2).mean(-1).clamp(max=1.0)
        self.last_action = action.clone()
        t1 = self.t + 1
        ref = torch.cat([self.ref_at("base_q", t1), self.ref_at("dof_pos", t1)], -1)       # (B, H, 26)
        delta = (action * self.action_scale).reshape(self._num_envs, H, 26)
        self._physics_step([ref[:, h] + delta[:, h] for h in range(H)])

    def _apply_object_aids(self) -> None:
        """Hold each object at its in-palm pose while the reference glove reports the grasp."""
        if self.assist_scale <= 0 or self._eval_mode:
            return
        a, t1 = self._args, self.t + 1
        holding = self.ref_at("holding", t1) & self.obj_active_t                              # (B, H)
        if not holding.any():
            return
        anchor = torch.as_tensor(self.pose_in_palm, dtype=torch.float32, device=self.device)[self.env_clip]
        for h, (hand, obj) in enumerate(zip(self.hands, self.objects)):
            st = hand.state()
            palm_p, palm_R = st["palm_pos"], st["palm_rot"]
            p, v, w = obj.get_pos(), obj.get_vel(), obj.get_ang()
            from tv2s_env.common.utils.math_utils import quat_to_rotmat
            R = quat_to_rotmat(obj.get_quat())
            p_target = palm_p + (palm_R @ anchor[:, h, :3, 3:])[..., 0]
            R_target = palm_R @ anchor[:, h, :3, :3]
            k = (self.assist_scale * a.assist_gain * holding[:, h].float())[:, None]
            m = self.obj_mass[:, h:h + 1]
            force = k * m * ((p_target - p) / a.assist_tau - v) / self.dt
            force[:, 2] += self.assist_scale * holding[:, h].float() * m[:, 0] * 9.81
            r2 = self.obj_radius[:, h] ** 2
            l2 = (2 * self.obj_half_length[:, h]) ** 2
            inertia = torch.stack([(3 * r2 + l2) / 12, (3 * r2 + l2) / 12, r2 / 2], -1) * m      # cylinder, object frame
            w_des = matrix_to_rotvec(R_target @ R.transpose(-1, -2)) / a.assist_tau
            dw_obj = ((w_des - w)[:, None] @ R)[:, 0]                                          # object frame
            torque = k * (R @ (inertia * dw_obj)[..., None])[..., 0] / self.dt
            self._apply_object_wrench(h, force, torque)

    def set_training_progress(self, iteration: int, total_iterations: int) -> None:
        """The assist decays linearly to 0 over the first `assist_decay` of training (rl_recon/train.py)."""
        super().set_training_progress(iteration, total_iterations)
        a = self._args
        progress = iteration / max(1.0, a.assist_decay * total_iterations)
        self.assist_scale = a.assist_init * max(0.0, 1.0 - progress)

    # ------------------------------------------------------------------ episodes
    def errors(self) -> dict[str, torch.Tensor]:
        """Per-hand errors that terminate an episode (rl_recon: palm, keypoints, depth-aware object)."""
        e_wrist = (self.base_pos - self.target_wrist_pos).norm(dim=-1)
        e_kp = (self.keypoints - self.target_keypoints).norm(dim=-1).mean(-1)
        err = self.object_pos - self.target_object_pos
        along = (err * self.camera_ray).sum(-1)
        across = (err - along[..., None] * self.camera_ray).norm(dim=-1)
        e_obj = torch.sqrt(across ** 2 + (along / self._args.obj_depth_discount) ** 2) * self.object_valid
        return dict(e_wrist=e_wrist, e_kp=e_kp, e_obj=e_obj)

    def get_terminated(self) -> torch.Tensor:
        a, e = self._args, self.errors()
        bad = ~torch.isfinite(self.object_pos).all(-1).all(-1) | ~torch.isfinite(self.base_pos).all(-1).all(-1)
        drift = ((e["e_wrist"] > a.term_wrist) | (e["e_kp"] > a.term_kp) | (e["e_obj"] > a.term_obj)).any(-1)
        terminated = bad if self.disable_termination else (drift | bad)
        self._extra_info["termination"] = dict(terminated=terminated, drift=drift, error=bad)
        self._terminated = terminated
        return terminated

    def get_truncated(self) -> torch.Tensor:
        truncated = (self.t >= self.horizon) & ~self._terminated
        self._extra_info["termination"]["truncated"] = truncated
        return truncated

    def get_reward(self):
        reward, terms = super().get_reward()
        bad = self._extra_info["termination"]["error"]
        reward = torch.where(bad, torch.zeros_like(reward), reward)
        terms["Total"] = reward
        return reward, terms

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        a, n, dev, gen = self._args, len(envs_idx), self.device, self.generator
        clip = self.env_clip[envs_idx]
        start = torch.zeros(n, dtype=torch.long, device=dev)
        if self.random_start:
            pick = torch.rand(n, device=dev, generator=gen) < a.start_frame_prob
            smax = (self.first_grasp[clip] - 10).clamp(min=0)
            start = torch.where(pick, (torch.rand(n, device=dev, generator=gen) * (smax + 1)).long(), start)
            rsi = torch.rand(n, device=dev, generator=gen) < a.rsi_grasp_prob
            if rsi.any():
                # a random frame where some hand holds its object
                w = self.holding_any[clip].float() + 1e-9
                frames = torch.multinomial(w, 1, generator=gen)[:, 0]
                start = torch.where(rsi & self.holding_any[clip].any(-1), frames, start)
        if self.in_hand_start:
            start = torch.as_tensor(self.grasp[..., 0], device=dev)[clip].masked_fill(
                ~torch.as_tensor(self.obj_active, device=dev)[clip], 10**6).min(-1).values
            start = start + 3
        start = torch.minimum(start, self.lengths_t[clip] - 2)
        self._reset_hands_and_objects(envs_idx, start, self.ref["holding"][clip, start])
        self.t[envs_idx] = start
        self.start[envs_idx] = start
        self.horizon[envs_idx] = self.lengths_t[clip] - 1
        self.last_action[envs_idx] = 0.0
        self.update_buffers()
