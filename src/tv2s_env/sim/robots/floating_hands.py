"""A Wuji hand on a 6-DoF virtual base in Genesis (Benchmark-Genesis' rl_recon hand), one per side.

Joints: prismatic x, y, z, then a fixed rotation R0 (the mean wrist orientation of the hand's clips, so
the Euler angles stay away from gimbal lock), then revolute x, y, z, then the 20 finger joints. Every
joint is PD position-controlled and gravity is compensated; a free joint cannot take position control
on its rotation, hence the virtual base. This replaces DexTaG's xArm7: HumanTouch gives wrist poses,
not arm configurations, and two arms would add a reachability problem the data says nothing about.
"""

import os

import genesis as gs
import numpy as np
import torch

from tv2s_env.common.utils.math_utils import quat_to_rotmat
from tv2s_env.robots.wuji_model import BASE_JOINTS, FINGERS, KEYPOINT_PARTS, WujiHand
from tv2s_env.sim.envs.config.schema import HandArgs


class FloatingWujiHand:
    n_base = 6
    def __init__(self, scene: gs.Scene, side: str, base_rotation: np.ndarray, args: HandArgs, work_dir: str,
                 device: torch.device):
        self.side = side
        self.args = args
        self.device = device
        wuji_dir = args.wuji_dir or os.environ.get(
            "WUJI_RETARGETING_DIR", os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"))
        from tv2s_env.robots.wuji_model import default_urdf

        self.model = WujiHand(side, urdf=default_urdf(side, wuji_dir))
        os.makedirs(work_dir, exist_ok=True)
        urdf = self.model.write_floating_urdf(os.path.join(work_dir, f"wuji_{side}_floating.urdf"),
                                              base_rotation, base_mass=0.01)
        self.base_rotation = torch.as_tensor(base_rotation, dtype=torch.float32, device=device)
        self.entity = scene.add_entity(
            gs.morphs.URDF(file=urdf, fixed=True, merge_fixed_links=True, default_armature=0.01),
            material=gs.materials.Rigid(gravity_compensation=1.0, friction=args.friction),
            surface=gs.surfaces.Default(color=(0.75, 0.75, 0.78) if side == "left" else (0.6, 0.65, 0.75)),
        )

    def post_build(self, num_envs: int) -> None:
        a, e = self.args, self.entity
        self.num_envs = num_envs
        self.base_dofs = [e.get_joint(n).dofs_idx_local[0] for n in BASE_JOINTS]
        self.finger_dofs = [e.get_joint(n).dofs_idx_local[0] for n in self.model.dof_names]
        self.all_dofs = self.base_dofs + self.finger_dofs
        self.link_index = {l.name: l.idx_local for l in e.links}
        self.palm_link = self.link_index[self.model.root]
        # fingertips and keypoints: tip_link is merged into link4, so they ride on it with an offset
        parents, offsets = [], []
        for f in FINGERS:
            for part in KEYPOINT_PARTS:
                name = self.model.link_name(f, part)
                if part == "tip_link":
                    j = self.model.joint_origin(name)
                    parents.append(self.link_index[j["parent"]]), offsets.append(j["origin"][:3, 3])
                else:
                    parents.append(self.link_index[name]), offsets.append(np.zeros(3))
        self.kp_links = torch.as_tensor(parents, device=self.device)
        self.kp_offsets = torch.as_tensor(np.array(offsets), dtype=torch.float32, device=self.device)
        self.tip_kp = [3 * i + 2 for i in range(5)]                     # tips among the 15 keypoints
        self.kp = torch.as_tensor([a.kp_base_pos] * 3 + [a.kp_base_rot] * 3 + [a.kp_finger] * 20,
                                  dtype=torch.float32, device=self.device)
        self.kv = torch.as_tensor([a.kv_base_pos] * 3 + [a.kv_base_rot] * 3 + [a.kv_finger] * 20,
                                  dtype=torch.float32, device=self.device)
        e.set_dofs_kp(self.kp, self.all_dofs)
        e.set_dofs_kv(self.kv, self.all_dofs)
        upper = np.r_[[a.base_force_limit] * 3, [a.base_torque_limit] * 3, self.model.effort * a.finger_effort_scale]
        e.set_dofs_force_range(-upper, upper, self.all_dofs)
        self.q_low = torch.as_tensor(self.model.lower, dtype=torch.float32, device=self.device)
        self.q_high = torch.as_tensor(self.model.upper, dtype=torch.float32, device=self.device)

    def randomize_gains(self, generator: torch.Generator | None = None) -> None:
        """DexTaG's kp/kd domain randomisation (per env, per dof)."""
        (k_lo, k_hi), (d_lo, d_hi) = self.args.kp_range, self.args.kd_range
        if (k_lo, k_hi) == (1.0, 1.0) and (d_lo, d_hi) == (1.0, 1.0):
            return
        n = (self.num_envs, len(self.all_dofs))
        kp = self.kp * (k_lo + (k_hi - k_lo) * torch.rand(n, device=self.device, generator=generator))
        kv = self.kv * (d_lo + (d_hi - d_lo) * torch.rand(n, device=self.device, generator=generator))
        envs = torch.arange(self.num_envs, device=self.device)
        self.entity.set_dofs_kp(kp, self.all_dofs, envs_idx=envs)
        self.entity.set_dofs_kv(kv, self.all_dofs, envs_idx=envs)

    # ------------------------------------------------------------------ state
    def state(self) -> dict[str, torch.Tensor]:
        e = self.entity
        lp, lq = e.get_links_pos(), e.get_links_quat()
        LR = quat_to_rotmat(lq)
        kp = (LR[:, self.kp_links] @ self.kp_offsets[None, :, :, None])[..., 0] + lp[:, self.kp_links]
        return dict(
            link_pos=lp, link_rot=LR, link_quat=lq,
            q=e.get_dofs_position(self.all_dofs), dq=e.get_dofs_velocity(self.all_dofs),
            force=e.get_dofs_control_force(self.all_dofs),
            palm_pos=lp[:, self.palm_link], palm_rot=LR[:, self.palm_link], palm_quat=lq[:, self.palm_link],
            palm_vel=e.get_links_vel()[:, self.palm_link], palm_ang=e.get_links_ang()[:, self.palm_link],
            keypoints=kp, tip_pos=kp[:, self.tip_kp], tip_quat=lq[:, self.kp_links[self.tip_kp]],
        )

    def set_state(self, q: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.entity.set_dofs_position(q, self.all_dofs, envs_idx=envs_idx, zero_velocity=True)
        self.entity.control_dofs_position(q, self.all_dofs, envs_idx=envs_idx)

    def control(self, q_target: torch.Tensor) -> None:
        q_target = q_target.clone()
        q_target[:, 6:] = torch.max(torch.min(q_target[:, 6:], self.q_high), self.q_low)
        self.entity.control_dofs_position(q_target, self.all_dofs)
        self.last_target = q_target
