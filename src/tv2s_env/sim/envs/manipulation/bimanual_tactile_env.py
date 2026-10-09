"""Bimanual HumanTouch environment shared by the DexTaG and Benchmark-Genesis (rl_recon) methods.

Scene: two Wuji hands on virtual bases (FloatingWujiHand), one proxy object per hand (a free cylinder,
see tv2s_data/clips.py) and one support plate per hand under its object, plus a ground plane. Each
environment replays one clip: the clips are spread over the environments in balanced blocks, the layout
Genesis uses for heterogeneous entities (each clip has its own cylinder sizes). A hand without an object
in a clip gets a parked object on the ground, far away, and no support.

Named buffers (attributes, shape (B, H, ...) with H = 2 hands, left then right) hold the simulated state,
the reference at the current frame and the simulated / reference glove; reward terms read them by name
(DexTaG's `required_keys`), observations are built from them by the method subclasses.

Time: `self.t` is the reference frame the current state corresponds to. An action targets frame t + 1;
after the physics step t advances and every buffer is compared with the reference at the new t.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np
import torch

from tv2s_data.glove import load_glove_layout
from tv2s_data.humantouch import SIDES
from tv2s_env.common.bases.base_env import BaseEnv
from tv2s_env.common.utils.math_utils import (
    np_angular_velocity,
    np_quat_continuity,
    np_velocity,
    quat_to_rotmat,
)
from tv2s_env.robots.wuji_model import base_coordinates, mean_rotation
from tv2s_env.sim.envs.config.schema import BimanualEnvArgs
from tv2s_env.sim.robots.floating_hands import FloatingWujiHand
from tv2s_env.tactile.mapping import TaxelSet
from tv2s_env.tactile.sensor import GloveSensor

H = len(SIDES)
PARK_POS = np.array([2.0, 0.0])        # where objects of hands without an object wait (xy, on the ground)
DEFAULT_CYLINDER = dict(radius=0.02, length=0.08, mass=0.1)


def select_clip_names(a) -> list[str]:
    """Clips of a DataArgs: the listed ones (or all of index.json), the train / eval split (deterministic
    hash of the episode, DexTaG's split keyed on the trajectory prefix), then max_clips."""
    import hashlib

    clip_dir = Path(a.clip_dir)
    clips = a.clips.split(",") if isinstance(a.clips, str) else a.clips     # a single CLI clip stays a string
    names = list(clips) or [c["name"] for c in json.load(open(clip_dir / "index.json"))["clips"]]
    if a.split_mode != "all":
        salt = f"{a.split_seed}:" if a.split_seed else ""
        bucket = lambda n: int.from_bytes(hashlib.md5((salt + n.split("_")[0]).encode()).digest()[:8], "big") / 2**64
        names = [n for n in names if (bucket(n) < a.train_ratio) == (a.split_mode == "train")]
    if a.max_clips:
        names = names[: a.max_clips]
    if not names:
        raise ValueError(f"no clips selected in {clip_dir} (split {a.split_mode})")
    return names


def balanced_assignment(n_variants: int, n_envs: int) -> np.ndarray:
    """Variant (clip) of every env: Genesis' balanced block assignment for heterogeneous entities."""
    base, extra = divmod(n_envs, n_variants)
    sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
    return np.repeat(np.arange(n_variants), sizes)


class BimanualTactileEnv(BaseEnv):
    def __init__(self, args: BimanualEnvArgs, num_envs: int, show_viewer: bool = False,
                 device: torch.device = torch.device("cuda"), eval_mode: bool = False) -> None:
        super().__init__(device=device)
        self._args = args
        self._eval_mode = eval_mode
        self._num_envs = num_envs
        self.generator = torch.Generator(device=device).manual_seed(args.seed)
        if not gs._initialized:  # noqa: SLF001
            gs.init(backend=gs.gpu if device.type == "cuda" else gs.cpu, precision="32",
                    logging_level="warning", seed=args.seed, performance_mode=True)

        self._load_clips()
        if num_envs < self.n_clips:
            raise ValueError(f"{num_envs} envs for {self.n_clips} clips: need at least one env per clip")
        self.env_clip = torch.as_tensor(balanced_assignment(self.n_clips, num_envs), device=device)
        self._build_scene(show_viewer)
        self._build_sensors()
        self._init_buffers()
        self._init_rewards()
        self.training_step = 0
        self.reset()

    # ================================================================== data
    def _load_clips(self) -> None:
        clip_dir = Path(self._args.data.clip_dir)
        names = select_clip_names(self._args.data)
        clips = [pickle.load(open(clip_dir / f"{n}.pkl", "rb")) for n in names]
        self.clip_names = names
        self.n_clips = len(clips)
        self.hz = float(clips[0]["metadata"]["hz"])
        self.dt = 1.0 / self.hz
        self.lengths = np.array([c["metadata"]["end"] - c["metadata"]["start"] for c in clips])
        T = int(self.lengths.max())
        self.T_max = T
        meta = clips[0]["metadata"]
        self.layouts = [load_glove_layout(meta["tactile_mapping"], s, finger_tip_row=meta["finger_tip_row"])
                        for s in SIDES]

        def pad(x):
            return np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)], 0) if len(x) < T else x

        # base rotation of each hand's virtual base: the mean wrist orientation over its clips
        rots = {s: [] for s in SIDES}
        for c in clips:
            for s in SIDES:
                q = c["hands"][s]["wrist_quat"]
                rots[s].append(_quats_to_rotmats(q))
        self.base_R0 = {s: mean_rotation(np.concatenate(rots[s])) for s in SIDES}

        keys = {k: [] for k in ["wrist_pos", "wrist_rot", "wrist_quat", "wrist_vel", "wrist_ang_vel", "base_q",
                                 "dof_pos", "dof_vel", "tip_pos", "tip_quat", "keypoints", "pressure", "contact",
                                 "obj_pos", "obj_rot", "obj_quat", "obj_vel", "obj_ang_vel", "obj_valid", "holding"]}
        self.obj_spec = np.zeros((self.n_clips, H, 3))          # radius, length, mass
        self.obj_active = np.zeros((self.n_clips, H), bool)
        self.support_z = np.zeros((self.n_clips, H))
        self.grasp = np.zeros((self.n_clips, H, 2), int)
        self.pose_in_palm = np.tile(np.eye(4), (self.n_clips, H, 1, 1))
        head = []
        for i, c in enumerate(clips):
            per = {k: [] for k in keys}
            L = self.lengths[i]
            for h, s in enumerate(SIDES):
                d = c["hands"][s]
                R = _quats_to_rotmats(d["wrist_quat"])
                pos = d["wrist_positions"].astype(np.float64)
                per["wrist_pos"].append(pos), per["wrist_rot"].append(R), per["wrist_quat"].append(d["wrist_quat"])
                per["wrist_vel"].append(np_velocity(pos, self.dt)), per["wrist_ang_vel"].append(np_angular_velocity(R, self.dt))
                per["base_q"].append(base_coordinates(self.base_R0[s], pos, R))
                per["dof_pos"].append(d["dof_positions"]), per["dof_vel"].append(np_velocity(d["dof_positions"], self.dt))
                per["tip_pos"].append(d["fingertip_pos"]), per["tip_quat"].append(np_quat_continuity(d["fingertip_quat"]))
                per["keypoints"].append(d["ref_keypoints"])
                per["pressure"].append(d["tactile_pressure"] / self._args.tactile.glove_scale)
                per["contact"].append(d["tactile_contact"].astype(np.float32))
                o = d["object"]
                if o is None:
                    poses = np.tile(np.eye(4), (L, 1, 1))
                    poses[:, :2, 3] = PARK_POS + np.array([0.0, 0.3 * h])
                    poses[:, 2, 3] = DEFAULT_CYLINDER["length"] / 2
                    valid, holding = np.zeros(L, bool), np.zeros(L, bool)
                    self.obj_spec[i, h] = [DEFAULT_CYLINDER[k] for k in ("radius", "length", "mass")]
                    self.support_z[i, h] = -1.0
                else:
                    poses, valid = o["pose_matrices"].astype(np.float64), o["valid"]
                    holding = np.zeros(L, bool)
                    holding[o["grasp_interval"][0]:o["grasp_interval"][1]] = True
                    self.obj_spec[i, h] = [o["radius"], o["length"], o["mass"]]
                    self.obj_active[i, h] = True
                    t0 = max(1, o["grasp_interval"][0])
                    hand_low = float(d["ref_keypoints"][:t0, :, 2].min()) - 0.003
                    drop = self._args.objects.support_max_drop
                    self.support_z[i, h] = max(min(o["support_z"], hand_low), o["support_z"] - drop)
                    self.grasp[i, h] = o["grasp_interval"]
                    self.pose_in_palm[i, h] = o["pose_in_palm"]
                Ro = poses[:, :3, :3]
                per["obj_pos"].append(poses[:, :3, 3]), per["obj_rot"].append(Ro)
                per["obj_quat"].append(np_quat_continuity(_rotmats_to_quats(Ro)))
                per["obj_vel"].append(np_velocity(poses[:, :3, 3], self.dt)), per["obj_ang_vel"].append(np_angular_velocity(Ro, self.dt))
                per["obj_valid"].append(valid), per["holding"].append(holding)
            for k in keys:
                keys[k].append(pad(np.stack(per[k], 1)))           # (T, H, ...)
            head.append(pad(c["head"]["pos"]))
        f = lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32, device=self.device)
        self.ref = {k: f(v) for k, v in keys.items()}            # (n_clips, T, H, ...)
        self.ref["obj_valid"] = self.ref["obj_valid"] > 0.5
        self.ref["holding"] = self.ref["holding"] > 0.5
        self.ref["head_pos"] = f(head)                           # (n_clips, T, 3)
        self.lengths_t = torch.as_tensor(self.lengths, device=self.device)
        # glove frames past the clip end must not count as contact
        for k in ("pressure", "contact"):
            for i, L in enumerate(self.lengths):
                self.ref[k][i, L:] = 0.0
        print(f"Loaded {self.n_clips} clips ({self.lengths.sum()} frames at {self.hz:g} Hz) from {clip_dir}; "
              f"objects: {int(self.obj_active[:, 0].sum())} left, {int(self.obj_active[:, 1].sum())} right")

    def ref_at(self, key: str, t: torch.Tensor) -> torch.Tensor:
        """Reference `key` of every env's clip at frames t (B,), clamped to the clip."""
        t = torch.minimum(t, self.lengths_t[self.env_clip] - 1).clamp(min=0)
        return self.ref[key][self.env_clip, t]

    # ================================================================== scene
    def _build_scene(self, show_viewer: bool) -> None:
        a = self._args
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=a.substeps, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, enable_joint_limit=True,
                                                  constraint_solver=gs.constraint_solver.Newton,
                                                  max_collision_pairs=200),
            vis_options=gs.options.VisOptions(show_world_frame=False, plane_reflection=False),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        sx, sy, sz = a.objects.support_size
        self.supports = [self.scene.add_entity(gs.morphs.Box(size=(sx, sy, sz), pos=(0.0, 3.0 + h, -1.0), fixed=True),
                                               surface=gs.surfaces.Default(color=(0.55, 0.45, 0.35)))
                         for h in range(H)]
        self.hands = [FloatingWujiHand(self.scene, s, self.base_R0[s], a.hands, a.work_dir, self.device) for s in SIDES]
        self.objects = []
        for h in range(H):
            morphs = [gs.morphs.Cylinder(radius=float(r), height=float(l), pos=(2.0, 0.3 * h, float(l) / 2))
                      for r, l, _ in self.obj_spec[:, h]]
            self.objects.append(self.scene.add_entity(
                morph=morphs if len(morphs) > 1 else morphs[0],
                material=gs.materials.Rigid(rho=500.0, friction=a.objects.friction),
                surface=gs.surfaces.Default(color=(0.75, 0.45, 0.85) if h == 0 else (0.45, 0.75, 0.55)),
            ))
        self.scene.build(n_envs=self._num_envs, env_spacing=(1.0, 1.0))
        for hand in self.hands:
            hand.post_build(self._num_envs)
            if not self._eval_mode:
                hand.randomize_gains(self.generator)

        B, dev = self._num_envs, self.device
        spec = torch.as_tensor(self.obj_spec, dtype=torch.float32, device=dev)[self.env_clip]     # (B, H, 3)
        self.obj_radius, self.obj_half_length = spec[..., 0], 0.5 * spec[..., 1]
        mass = spec[..., 2].clone()
        o = a.objects
        if not self._eval_mode and o.mass_randomization_range > 0:
            mass *= 1 + o.mass_randomization_range * (2 * torch.rand(mass.shape, device=dev, generator=self.generator) - 1)
        self.obj_mass = mass
        envs = torch.arange(B, device=dev)
        for h, obj in enumerate(self.objects):
            obj.set_links_inertial_mass(mass[:, h:h + 1], links_idx_local=[0], envs_idx=envs)
            if not self._eval_mode and o.friction_range != (1.0, 1.0):
                lo, hi = o.friction_range
                obj.set_friction_ratio(lo + (hi - lo) * torch.rand(B, 1, device=dev, generator=self.generator),
                                       links_idx_local=[0], envs_idx=envs)
        self.obj_link = [obj.links[0].idx for obj in self.objects]
        # support plates under each object's rest pose
        self.obj_active_t = torch.as_tensor(self.obj_active, device=dev)[self.env_clip]            # (B, H)
        self.support_z_t = torch.as_tensor(self.support_z, dtype=torch.float32, device=dev)[self.env_clip]
        rest = self.ref["obj_pos"][self.env_clip, 0]                                                # (B, H, 3)
        for h, sup in enumerate(self.supports):
            pos = torch.stack([rest[:, h, 0], rest[:, h, 1], self.support_z_t[:, h] - sz / 2], -1)
            parked = torch.tensor([0.0, 3.0 + h, -1.0], device=dev)
            sup.set_pos(torch.where(self.obj_active_t[:, h:h + 1], pos, parked), envs_idx=envs)

    def _build_sensors(self) -> None:
        t = self._args.tactile
        self.sensors = []
        for h, s in enumerate(SIDES):
            taxels = TaxelSet.load(str(Path(t.assets_dir) / f"{t.mapping}_{s}.npz"))
            self.sensors.append(GloveSensor(taxels, self.layouts[h], self.hands[h].link_index, t.probe_radius, t.kn,
                                            self.device))
        self.tactile_pixel_weights = torch.stack(
            [s.to_image(s.channel_weights[None])[0] for s in self.sensors])                       # (H, 23, 20)
        self.n_channels = self.layouts[0].n_channels

    # ================================================================== buffers
    def _init_buffers(self) -> None:
        B, dev = self._num_envs, self.device
        z = lambda *s: torch.zeros((B,) + s, device=dev)
        self.t = torch.zeros(B, dtype=torch.long, device=dev)
        self.start = torch.zeros(B, dtype=torch.long, device=dev)
        self.horizon = self.lengths_t[self.env_clip] - 1
        self.reset_buf = torch.zeros(B, dtype=torch.bool, device=dev)
        self.last_reward = z(1)
        self.object_vel_change, self.object_ang_vel_change = z(H), z(H)
        self._extra_info: dict[str, Any] = {}

    def _init_rewards(self) -> None:
        import importlib

        module = importlib.import_module(f"tv2s_env.common.rewards.{self._args.method}_terms")
        self._reward_functions = {}
        self._reward_required_keys: set[str] = set()
        for name, kw in self._args.reward_args.items():
            cls = getattr(module, name, None)
            if cls is None:
                raise ValueError(f"reward {name} not in {module.__name__}")
            kw = dict(kw)
            scale = kw.pop("scale") * (self.dt if self._args.reward_scale_by_dt else 1.0)
            term = cls(scale=scale, **kw)
            self._reward_functions[name] = term
            self._reward_required_keys.update(term.required_keys)

    # ================================================================== state
    def update_buffers(self) -> None:
        """Read the simulator and the reference at frame t into the named buffers."""
        st = [hand.state() for hand in self.hands]
        stack = lambda k: torch.stack([s[k] for s in st], 1)
        self.hand_dof_pos, self.hand_dof_vel = stack("q")[..., 6:], stack("dq")[..., 6:]
        self.base_dof_pos, self.base_dof_vel = stack("q")[..., :6], stack("dq")[..., :6]
        self.dof_vel, self.dof_force = stack("dq"), stack("force")
        self.base_pos, self.base_rot, self.base_quat = stack("palm_pos"), stack("palm_rot"), stack("palm_quat")
        self.base_lin_vel, self.base_ang_vel = stack("palm_vel"), stack("palm_ang")
        self.keypoints, self.finger_link_pos, self.finger_link_quat = stack("keypoints"), stack("tip_pos"), stack("tip_quat")
        self.object_pos = torch.stack([o.get_pos() for o in self.objects], 1)
        self.object_quat = torch.stack([o.get_quat() for o in self.objects], 1)
        self.object_rot = quat_to_rotmat(self.object_quat)
        self.object_axis = self.object_rot[..., 2]
        self.object_lin_vel = torch.stack([o.get_vel() for o in self.objects], 1)
        self.object_ang_vel = torch.stack([o.get_ang() for o in self.objects], 1)
        self.object_pos_rel = self.object_pos - self.base_pos
        self.object_height_above_support = self.object_pos[..., 2] - self.support_z_t
        self.support_contact_force = torch.stack(
            [s.get_links_net_contact_force().norm(dim=-1).sum(-1) for s in self.supports], 1)

        # simulated glove
        dist, pressure = [], []
        for h, sensor in enumerate(self.sensors):
            d, p = sensor.read(st[h]["link_pos"], st[h]["link_rot"], self.object_pos[:, h], self.object_rot[:, h],
                               self.obj_radius[:, h], self.obj_half_length[:, h])
            dist.append(d), pressure.append(p)
        self.tactile_distance, self.tactile_pressure = torch.stack(dist, 1), torch.stack(pressure, 1)   # (B, H, C)
        self.tactile_map = torch.stack([s.to_image(self.tactile_pressure[:, h]) for h, s in enumerate(self.sensors)], 1)
        # DexTaG's fingertip-to-surface distance: closest distal taxel of each finger
        self.fingertip_to_surface_dist = torch.stack(
            [torch.stack([torch.where(s.finger_masks[f] & tip, self.tactile_distance[:, h], float("inf")).amin(-1)
                          for f in range(5)], -1)
             for h, (s, tip) in enumerate(zip(self.sensors, self._distal_masks()))], 1)

        # reference at the current frame
        t = self.t
        self.target_wrist_pos, self.target_wrist_rot = self.ref_at("wrist_pos", t), self.ref_at("wrist_rot", t)
        self.target_wrist_quat = self.ref_at("wrist_quat", t)
        self.target_wrist_vel, self.target_wrist_ang_vel = self.ref_at("wrist_vel", t), self.ref_at("wrist_ang_vel", t)
        self.target_hand_dof_pos, self.target_hand_dof_vel = self.ref_at("dof_pos", t), self.ref_at("dof_vel", t)
        self.target_fingertip_pos, self.target_fingertip_quat = self.ref_at("tip_pos", t), self.ref_at("tip_quat", t)
        self.target_keypoints = self.ref_at("keypoints", t)
        self.target_object_pos, self.target_object_rot = self.ref_at("obj_pos", t), self.ref_at("obj_rot", t)
        self.target_object_axis = self.target_object_rot[..., 2]
        self.target_object_quat = self.ref_at("obj_quat", t)
        self.target_object_vel, self.target_object_ang_vel = self.ref_at("obj_vel", t), self.ref_at("obj_ang_vel", t)
        self.object_valid = self.ref_at("obj_valid", t) & self.obj_active_t
        self.target_pressure = self.ref_at("pressure", t)
        self.target_contact = self.ref_at("contact", t) > 0.5
        self.target_tactile_map = torch.stack(
            [s.to_image(self.target_pressure[:, h] * self.target_contact[:, h]) for h, s in enumerate(self.sensors)], 1)
        self.target_finger_contact = torch.stack(
            [torch.stack([(self.target_contact[:, h] & s.finger_masks[f]).any(-1) for f in range(5)], -1)
             for h, s in enumerate(self.sensors)], 1)
        ray = self.target_object_pos - self.ref_at("head_pos", t)[:, None]
        self.camera_ray = ray / ray.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _distal_masks(self) -> list[torch.Tensor]:
        if not hasattr(self, "_distal"):
            self._distal = [torch.as_tensor((l.region != "palm") & (l.u > 2 / 3), device=self.device) for l in self.layouts]
        return self._distal

    # ================================================================== shared metrics
    def common_metrics(self) -> dict[str, torch.Tensor]:
        """Method-independent metrics (B,), logged and evaluated identically for both methods."""
        from tv2s_env.common.rewards.dextag_terms import WeightedGaussianBlurredTactileMapSimilarityReward, hand_mean
        sim_contact = self.tactile_distance < self._args.tactile.probe_radius
        y = self.target_contact
        tp = (sim_contact & y).sum(-1).float()
        n_sim, n_gt = sim_contact.sum(-1).float(), y.sum(-1).float()
        if not hasattr(self, "_map_metric"):
            self._map_metric = WeightedGaussianBlurredTactileMapSimilarityReward()
        map_sim = self._map_metric.similarity(self.tactile_map, self.target_tactile_map, self.tactile_pixel_weights)
        has = n_gt > 0
        valid = self.object_valid
        return dict(
            tac_tp=tp.sum(-1), tac_sim=n_sim.sum(-1), tac_real=n_gt.sum(-1),
            tac_map_similarity=hand_mean(map_sim, has),
            e_wrist=(self.base_pos - self.target_wrist_pos).norm(dim=-1).mean(-1),
            e_tip=(self.finger_link_pos - self.target_fingertip_pos).norm(dim=-1).mean((-1, -2)),
            e_kp=(self.keypoints - self.target_keypoints).norm(dim=-1).mean((-1, -2)),
            e_obj=hand_mean((self.object_pos - self.target_object_pos).norm(dim=-1), valid),
            obj_valid=valid.any(-1).float(),
        )

    # ================================================================== reset helpers
    def _reset_hands_and_objects(self, envs_idx: torch.Tensor, start: torch.Tensor, in_hand: torch.Tensor) -> None:
        """Hands on the reference at `start`; objects at their reference pose there, i.e. at rest before
        the grasp and in the reference palm inside it (`in_hand` (n, H): place it at the in-palm pose)."""
        clip = self.env_clip[envs_idx]
        for h, hand in enumerate(self.hands):
            q = torch.cat([self.ref["base_q"][clip, start, h], self.ref["dof_pos"][clip, start, h]], -1)
            hand.set_state(q, envs_idx)
        pos = self.ref["obj_pos"][clip, start].clone()                                      # (n, H, 3)
        quat = self.ref["obj_quat"][clip, start].clone()
        before = start[:, None] < torch.as_tensor(self.grasp[..., 0], device=self.device)[clip]
        pos[..., 2] += 0.002 * before                                                       # clear the support
        o = self._args.objects
        if not self._eval_mode and o.init_pos_noise_xy > 0:
            noise = (2 * torch.rand(pos[..., :2].shape, device=self.device, generator=self.generator) - 1)
            pos[..., :2] += o.init_pos_noise_xy * noise * (before & ~in_hand)[..., None]
        if not self._eval_mode and o.init_rot_noise_deg > 0:
            from scipy.spatial.transform import Rotation
            n = len(envs_idx) * H
            r = Rotation.from_rotvec(np.radians(o.init_rot_noise_deg) * (2 * np.random.rand(n, 3) - 1))
            qn = torch.as_tensor(r.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float32, device=self.device).reshape(-1, H, 4)
            from tv2s_env.common.utils.math_utils import quat_mul
            quat = torch.where((before & ~in_hand)[..., None], quat_mul(qn, quat), quat)
        for h, obj in enumerate(self.objects):
            obj.set_pos(pos[:, h], envs_idx=envs_idx, zero_velocity=True)
            obj.set_quat(quat[:, h], envs_idx=envs_idx, zero_velocity=True)

    # ================================================================== stepping
    def _physics_step(self, hand_targets: list[torch.Tensor]) -> None:
        prev_v = torch.stack([o.get_vel() for o in self.objects], 1)
        prev_w = torch.stack([o.get_ang() for o in self.objects], 1)
        for hand, target in zip(self.hands, hand_targets):
            hand.control(target)
        self._apply_object_aids()
        self.scene.step()
        self.t += 1
        self.object_vel_change = (torch.stack([o.get_vel() for o in self.objects], 1) - prev_v).norm(dim=-1)
        self.object_ang_vel_change = (torch.stack([o.get_ang() for o in self.objects], 1) - prev_w).norm(dim=-1)
        self.update_buffers()

    def _apply_object_aids(self) -> None:
        """Training aids acting on the objects (method specific)."""

    def set_training_progress(self, iteration: int, total_iterations: int) -> None:
        """Called by the runner before every iteration: drives the curricula and the training aids."""
        self.training_step = iteration

    def _apply_object_wrench(self, h: int, force: torch.Tensor, torque: torch.Tensor | None = None) -> None:
        solver = self.scene.sim.rigid_solver
        solver.apply_links_external_force(force[:, None], self.obj_link[h], ref="link_com")
        if torque is not None:
            solver.apply_links_external_torque(torque[:, None], self.obj_link[h], ref="link_com")

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = {k: getattr(self, k) for k in self._reward_required_keys}
        total = torch.zeros(self.num_envs, device=self.device)
        terms = {}
        for name, fn in self._reward_functions.items():
            r = fn(state)
            total = total + r
            terms[name] = r
        terms["Total"] = total
        self.last_reward[:] = total[:, None]
        return total, terms

    def get_extra_infos(self) -> dict[str, Any]:
        self._extra_info["metrics"] = self.common_metrics()
        return self._extra_info

    def step(self, action: torch.Tensor):
        """apply -> terminated / truncated -> reward -> infos -> reset done envs -> observations."""
        self.apply_action(action)
        terminated = self.get_terminated()
        truncated = self.get_truncated()
        reward, terms = self.get_reward()
        infos = self.get_extra_infos()
        infos["reward_terms"] = terms
        infos["episode_length"] = (self.t - self.start).clone()
        done = (terminated | truncated).nonzero(as_tuple=True)[0]
        if len(done) > 0:
            self.reset_idx(done)
        actor_obs, critic_obs = self.get_observations()
        return actor_obs, critic_obs, reward[:, None], terminated[:, None], truncated[:, None], infos

    def close(self) -> None:
        pass

    @property
    def num_envs(self) -> int:
        return self._num_envs


def _quats_to_rotmats(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)[:, [1, 2, 3, 0]]).as_matrix()


def _rotmats_to_quats(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
