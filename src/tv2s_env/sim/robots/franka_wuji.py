"""A Wuji hand on a Franka Panda in Genesis: the arm's 7 joints in place of the floating hand's virtual base.

The hand's palm frame (z along the fingers) is fixed on the Panda's flange (`attachment`, z out of the flange),
`mount_offset` out of it and turned about it (`mount_yaw`, else the turn that reaches the clip best): the wrist sits on the flange as on a forearm. The base
stands on the table (z = 0). The demonstration gives palm poses, not arm configurations: `place_arm` solves the
whole clip by damped least-squares IK (warm-started frame to frame, a null-space pull to the Panda's ready posture),
for candidate bases behind the wrist trajectory on the side of the head (where the demonstrator's shoulder is),
and keeps the base that reaches the clip best. Its joint trajectory is the reference the policy acts around.

Every joint is PD position-controlled and gravity is compensated (the Panda's torque controller does the same).
Same interface as FloatingWujiHand (state, set_state, control, randomize_gains), with `n_base` = 7 arm dofs.
"""

import os
import xml.etree.ElementTree as ET

import genesis as gs
import numpy as np
import torch

from tv2s_env.common.utils.math_utils import quat_to_rotmat
from tv2s_env.robots.wuji_model import FINGERS, KEYPOINT_PARTS, WujiHand
from tv2s_env.sim.envs.config.schema import ArmArgs, HandArgs

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_VELOCITY = np.array([2.175] * 4 + [2.61] * 3)              # rad/s, the Panda's datasheet
FLANGE = "attachment"
TABLE_LINKS = ["link3", "link4", "link5", "link6", "link7"]    # the links that can come down to the table
# The bullet URDF's inertias are placeholders (0.1 kg m^2 on every link and on the massless flange: 10-500x the
# real ones): MuJoCo Menagerie's
# franka_emika_panda (same link frames) as mass, centre of mass, ixx iyy izz ixy ixz iyz
PANDA_INERTIA = {
    "link0": (0.629769, (-0.041018, -0.00014, 0.049974), (0.00315, 0.00388, 0.004285, 8.2904e-7, 0.00015, 8.2299e-6)),
    "link1": (4.970684, (0.003875, 0.002081, -0.04762), (0.70337, 0.70661, 0.009117, -0.000139, 0.006772, 0.019169)),
    "link2": (0.646926, (-0.003141, -0.02872, 0.003495), (0.007962, 0.02811, 0.025995, -0.003925, 0.010254, 0.000704)),
    "link3": (3.228604, (0.027518, 0.039252, -0.066502), (0.037242, 0.036155, 0.01083, -0.004761, -0.011396, -0.012805)),
    "link4": (3.587895, (-0.05317, 0.104419, 0.027454), (0.025853, 0.019552, 0.028323, 0.007796, -0.001332, 0.008641)),
    "link5": (1.225946, (-0.011953, 0.041065, -0.038437), (0.035549, 0.029474, 0.008627, -0.002117, -0.004037, 0.000229)),
    "link6": (1.666555, (0.060149, -0.014117, -0.010517), (0.001964, 0.004354, 0.005433, 0.000109, -0.001158, 0.000341)),
    "link7": (0.735522, (0.010517, -0.004252, 0.061597), (0.012516, 0.010027, 0.004815, -0.000428, -0.001196, -0.000741)),
}


def default_panda_urdf() -> str:
    return os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "panda_bullet", "panda_nohand.urdf")


def mount_transform(arm: ArmArgs, yaw: float) -> np.ndarray:
    """The palm's pose in the flange frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    M = np.eye(4)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    M[2, 3] = arm.mount_offset
    return M


def base_transform(xy, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    B = np.eye(4)
    B[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    B[:2, 3] = xy
    return B


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


class PandaKinematics:
    """Numpy FK, Jacobian and DLS IK of the palm mounted on the Panda (WujiHand's URDF parsing is generic)."""

    def __init__(self, urdf: str, mount: np.ndarray) -> None:
        self.chain = WujiHand("panda", urdf=urdf)
        self.joints = [next(j for j in self.chain.joints if j["name"] == n) for n in self.chain.dof_names]
        self.lower, self.upper = self.chain.lower, self.chain.upper
        self.mount = mount

    def fk(self, q: np.ndarray, base: np.ndarray):
        """The palm pose (..., 4, 4) and the Jacobian (..., 6, 7) of q (..., 7) in the world (base (..., 4, 4))."""
        T = self.chain.fk(q, base=base)
        palm = T[FLANGE] @ self.mount
        axes = np.stack([T[j["child"]][..., :3, :3] @ (j["axis"] / np.linalg.norm(j["axis"])) for j in self.joints], -1)
        origins = np.stack([T[j["child"]][..., :3, 3] for j in self.joints], -1)          # (..., 3, 7)
        Jp = np.cross(axes, palm[..., :3, 3, None] - origins, axis=-2)
        return palm, np.concatenate([Jp, axes], -2), T

    def solve(self, base, p_target, R_target, q, rest, iters: int, w_rot: float = 0.5, damping: float = 0.05,
              k_null: float = 0.1):
        """DLS IK from q (..., 7); returns q, the position (m) and rotation (rad) errors."""
        lo, hi = self.lower + 1e-3, self.upper - 1e-3
        eye6 = np.eye(6)
        W = np.r_[np.ones(3), np.full(3, w_rot)]
        for _ in range(iters):
            palm, J, _ = self.fk(q, base)
            e = np.concatenate([p_target - palm[..., :3, 3], _rotvec(R_target @ np.swapaxes(palm[..., :3, :3], -1, -2))], -1)
            Jw, ew = J * W[:, None], e * W
            Jpinv = np.swapaxes(Jw, -1, -2) @ np.linalg.inv(Jw @ np.swapaxes(Jw, -1, -2) + damping ** 2 * eye6)
            null = np.eye(7) - Jpinv @ Jw
            dq = (Jpinv @ ew[..., None])[..., 0] + (null @ (k_null * (rest - q))[..., None])[..., 0]
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        palm, _, _ = self.fk(q, base)
        e_pos = np.linalg.norm(p_target - palm[..., :3, 3], axis=-1)
        e_rot = np.linalg.norm(_rotvec(R_target @ np.swapaxes(palm[..., :3, :3], -1, -2)), axis=-1)
        return q, e_pos, e_rot

    def track(self, base, wrist_pos, wrist_rot, rest, frames=None):
        """The clip (T, 3), (T, 3, 3) solved frame after frame, warm-started; base (..., 4, 4) batches candidates."""
        frames = np.arange(len(wrist_pos)) if frames is None else frames
        q = np.broadcast_to(rest, base.shape[:-2] + (7,)).copy()
        qs, ep, er = [], [], []
        for i, t in enumerate(frames):
            q, e_p, e_r = self.solve(base, wrist_pos[t], wrist_rot[t], q, rest, iters=150 if i == 0 else 20)
            qs.append(q), ep.append(e_p), er.append(e_r)
        return np.stack(qs, -2), np.stack(ep, -1), np.stack(er, -1)       # (..., F, 7), (..., F), (..., F)


def place_arm(arm: ArmArgs, wrist_pos: np.ndarray, wrist_rot: np.ndarray, head_pos: np.ndarray,
              obstacles: np.ndarray) -> dict:
    """The base (4, 4), the mount (4, 4) and the joint reference (T, 7) of one hand's arm. Candidate bases (unless
    arm.base_xy): around the wrist trajectory's centre, base_distance towards the head and base_lateral across,
    facing the centre, at least 15 cm from the objects; candidate mounts (unless arm.mount_yaw): the hand turned by
    0, +-90 or 180 deg on the flange (on pick_up_screwdriver/103227: 5.5 cm / 42 deg worst errors at 0 deg, 1 mm at
    180). Scored on every third frame by position error + rotation error / 50 + limit proximity."""
    rest = np.asarray(arm.rest_q, float)
    centre = wrist_pos[:, :2].mean(0)
    if arm.base_xy is not None:
        xy = np.asarray(arm.base_xy, float)[None]
        yaw = np.array([arm.base_yaw if arm.base_yaw is not None else np.arctan2(*(centre - xy[0])[::-1])])
    else:
        d = head_pos[:, :2].mean(0) - centre
        d = d / max(np.linalg.norm(d), 1e-6)
        lat = np.array([-d[1], d[0]])
        xy = np.array([centre + r * d + s * lat for r in arm.base_distance for s in arm.base_lateral])
        if len(obstacles):
            xy = xy[np.linalg.norm(xy[:, None] - obstacles[None, :, :2], axis=-1).min(1) > 0.15]
        yaw = np.arctan2(centre[1] - xy[:, 1], centre[0] - xy[:, 0])
        if arm.base_yaw is not None:
            yaw[:] = arm.base_yaw
    bases = np.stack([base_transform(p, y) for p, y in zip(xy, yaw)])
    mounts = [arm.mount_yaw] if arm.mount_yaw is not None else [0.0, np.pi / 2, -np.pi / 2, np.pi]
    best = (np.inf, 0, mounts[0])
    for m in mounts:
        kin = PandaKinematics(arm.urdf or default_panda_urdf(), mount_transform(arm, m))
        if len(bases) * len(mounts) == 1:
            break
        q, ep, er = kin.track(bases, wrist_pos, wrist_rot, rest, np.arange(0, len(wrist_pos), 3))
        margin = np.minimum(q - kin.lower, kin.upper - q)
        cost = ep.mean(-1) + er.mean(-1) / 50 + 0.05 * (np.clip(arm.limit_margin - margin, 0, None) / arm.limit_margin).mean((-1, -2))
        k = int(np.argmin(cost))
        if cost[k] < best[0]:
            best = (cost[k], k, m)
    _, k, m = best
    kin = PandaKinematics(arm.urdf or default_panda_urdf(), mount_transform(arm, m))
    q, ep, er = kin.track(bases[k], wrist_pos, wrist_rot, rest)
    return dict(base=bases[k], mount=mount_transform(arm, m), mount_yaw=m, q=q, e_pos=ep, e_rot=er,
                candidates=len(bases) * len(mounts))


def write_arm_urdf(hand: WujiHand, panda_urdf: str, mount: np.ndarray, out_path: str) -> str:
    """The Panda with the hand on its flange (fixed joint `wuji_mount`), mesh paths made absolute, Menagerie's
    inertias."""
    from scipy.spatial.transform import Rotation

    def absolute(root, urdf):
        for mesh in root.iter("mesh"):
            f = mesh.get("filename").replace("package://", "")
            mesh.set("filename", os.path.normpath(os.path.join(os.path.dirname(urdf), f)))

    arm = ET.parse(panda_urdf).getroot()
    absolute(arm, panda_urdf)
    for link in arm.findall("link"):
        if link.get("name") == FLANGE and link.find("inertial") is not None:      # massless, 0.1 kg m^2 merged into link7
            link.remove(link.find("inertial"))
        if link.get("name") in PANDA_INERTIA and link.find("inertial") is not None:
            m, c, (ixx, iyy, izz, ixy, ixz, iyz) = PANDA_INERTIA[link.get("name")]
            inertial = link.find("inertial")
            inertial.find("origin").set("xyz", " ".join(map(str, c)))
            inertial.find("mass").set("value", str(m))
            for k, v in dict(ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz).items():
                inertial.find("inertia").set(k, str(v))
    hand_root = ET.fromstring(ET.tostring(hand.root_el))
    absolute(hand_root, hand.urdf)
    for el in hand_root:
        if el.tag in ("link", "joint"):
            arm.append(el)
    j = ET.SubElement(arm, "joint", name="wuji_mount", type="fixed")
    r, p, y = Rotation.from_matrix(mount[:3, :3]).as_euler("xyz")
    ET.SubElement(j, "origin", xyz=" ".join(map(str, mount[:3, 3])), rpy=f"{r} {p} {y}")
    ET.SubElement(j, "parent", link=FLANGE)
    ET.SubElement(j, "child", link=hand.root)
    ET.ElementTree(arm).write(out_path)
    return out_path


class FrankaWujiHand:
    n_base = 7

    def __init__(self, scene: gs.Scene, side: str, base: np.ndarray, mount: np.ndarray, args: HandArgs, arm: ArmArgs,
                 work_dir: str, device: torch.device):
        from scipy.spatial.transform import Rotation
        from tv2s_env.robots.wuji_model import default_urdf

        self.side, self.args, self.arm, self.device = side, args, arm, device
        wuji_dir = args.wuji_dir or os.environ.get(
            "WUJI_RETARGETING_DIR", os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"))
        self.model = WujiHand(side, urdf=default_urdf(side, wuji_dir))
        self.kin = PandaKinematics(arm.urdf or default_panda_urdf(), mount)
        os.makedirs(work_dir, exist_ok=True)
        urdf = write_arm_urdf(self.model, arm.urdf or default_panda_urdf(), mount,
                              os.path.join(work_dir, f"panda_wuji_{side}.urdf"))
        q = Rotation.from_matrix(base[:3, :3]).as_quat()
        self.entity = scene.add_entity(
            gs.morphs.URDF(file=urdf, fixed=True, merge_fixed_links=True, links_to_keep=[self.model.root],
                           default_armature=0.01, pos=tuple(map(float, base[:3, 3])),
                           quat=(float(q[3]), float(q[0]), float(q[1]), float(q[2]))),
            material=gs.materials.Rigid(gravity_compensation=1.0, friction=args.friction),
            surface=gs.surfaces.Default(color=(0.75, 0.75, 0.78) if side == "left" else (0.6, 0.65, 0.75)),
        )

    def post_build(self, num_envs: int) -> None:
        a, arm, e = self.args, self.arm, self.entity
        self.num_envs = num_envs
        self.base_dofs = [e.get_joint(n).dofs_idx_local[0] for n in ARM_JOINTS]
        self.finger_dofs = [e.get_joint(n).dofs_idx_local[0] for n in self.model.dof_names]
        self.all_dofs = self.base_dofs + self.finger_dofs
        self.link_index = {l.name: l.idx_local for l in e.links}
        self.palm_link = self.link_index[self.model.root]
        self.table_links = [self.link_index[n] for n in TABLE_LINKS]
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
        self.tip_kp = [3 * i + 2 for i in range(5)]
        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.kp = f32(np.r_[arm.kp, [a.kp_finger] * 20])
        self.kv = f32(np.r_[arm.kv, [a.kv_finger] * 20])
        e.set_dofs_kp(self.kp, self.all_dofs)
        e.set_dofs_kv(self.kv, self.all_dofs)
        upper = np.r_[arm.effort, self.model.effort * a.finger_effort_scale]
        e.set_dofs_force_range(-upper, upper, self.all_dofs)
        self.q_low = f32(self.model.lower)
        self.q_high = f32(self.model.upper)
        self.arm_low, self.arm_high = f32(self.kin.lower), f32(self.kin.upper)
        self.arm_effort, self.arm_velocity = f32(arm.effort), f32(ARM_VELOCITY)

    def randomize_gains(self, generator: torch.Generator | None = None) -> None:
        (k_lo, k_hi), (d_lo, d_hi) = self.args.kp_range, self.args.kd_range
        if (k_lo, k_hi) == (1.0, 1.0) and (d_lo, d_hi) == (1.0, 1.0):
            return
        n = (self.num_envs, len(self.all_dofs))
        kp = self.kp * (k_lo + (k_hi - k_lo) * torch.rand(n, device=self.device, generator=generator))
        kv = self.kv * (d_lo + (d_hi - d_lo) * torch.rand(n, device=self.device, generator=generator))
        envs = torch.arange(self.num_envs, device=self.device)
        self.entity.set_dofs_kp(kp, self.all_dofs, envs_idx=envs)
        self.entity.set_dofs_kv(kv, self.all_dofs, envs_idx=envs)

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
            arm_link_z=lp[:, self.table_links, 2],
        )

    def set_state(self, q: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.entity.set_dofs_position(q, self.all_dofs, envs_idx=envs_idx, zero_velocity=True)
        self.entity.control_dofs_position(q, self.all_dofs, envs_idx=envs_idx)

    def control(self, q_target: torch.Tensor, dq_target: torch.Tensor | None = None) -> None:
        """PD to q_target; dq_target (n, 7), the arm reference's velocity, is fed forward: the damping that holds
        the hand still would otherwise brake every motion (kv 200 x 1 rad/s saturates the wrist's 12 N m)."""
        q_target = q_target.clone()
        q_target[:, :7] = torch.max(torch.min(q_target[:, :7], self.arm_high), self.arm_low)
        q_target[:, 7:] = torch.max(torch.min(q_target[:, 7:], self.q_high), self.q_low)
        if dq_target is None:
            self.entity.control_dofs_position(q_target, self.all_dofs)
        else:
            dq = torch.cat([dq_target, torch.zeros_like(q_target[:, 7:])], -1)
            self.entity.control_dofs_position_velocity(q_target, dq, self.all_dofs)
        self.last_target = q_target
