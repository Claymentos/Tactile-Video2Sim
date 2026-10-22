"""Benchmark-Genesis' rl_recon environment (reconstruction/rl_recon/env.py: ReconEnv) on a per-episode reference,
with one or two hands.

The reference (reference.npz, tv2s_data/reference.py) is either Benchmark-Genesis' own (its prepare_reference.py:
one hand, one object) or one written by this pipeline from an episode clip (scripts/rl/prepare_reference.py: the
hands the demonstration uses, after the scene's `hands_used`, and every reconstructed object). The environment
simulates the reference's hands (`active_hands`: all of them, or one side): with one hand, H = 1 and the action,
observation and reward are that hand's, so a Benchmark-Genesis reference gives ReconEnv's problem exactly (same
observation, 681 values on the screwdriver).

ReconEnv term by term, per hand, objects averaged (rl_recon: one hand, one object):
  scene        the table (ground plane, z = 0), each Wuji Hand on a 6-DoF PD virtual base, gravity compensated,
               every object a rigid body (free or fixed) on its collision mesh, convex-decomposed at
               `decompose_threshold`, mass from the reference (Benchmark-Genesis': `object_mass`, 80 g);
               nothing is teleported after a reset.
  action       26 per hand in [-1, 1]: a residual on the reference at t+1 (base x 3 cm, Euler x 0.25 rad,
               fingers x 0.4 rad).
  glove        per taxel the signed distance to the hand's object (its held object now, else next, else last;
               `tactile_objects=all`: the nearest object), from the voxel SDF of its collision mesh; per channel
               the smallest; contact below the probe radius.
  reward       w_tac r_tactile + w_hand r_hand + w_obj (1 - p_position) + w_obj_axis (1 - p_axis) - w_rate rate:
                 r_tactile  (1 - w) soft contact F1 + w reach, against the glove's contacts at t (hands averaged)
                 r_hand     0.4 exp(-e_palm / 2 cm) + 0.2 exp(-e_rot / 0.25) + 0.4 exp(-e_keypoints / 3 cm)
                 p_position 1 - exp(-hinge): the error across the camera ray (2 cm free) and along it (10 cm)
                 p_axis     1 - exp(-hinge) on the long axis: against the object's mask axis in the image where the
                            reference has one (Benchmark-Genesis' mask_axes.py), else its projected reference axis
               object terms over the task objects (the free objects the simulated hands hold), where their
               reference is valid; zero reward when the simulation blows up.
  episodes     training (`train_start`): from frame 0 (or the first frame clear of the resting objects) as
               evaluation, or ("random") from frame 0, from a random frame before the first grasp (`rest_starts`),
               or (reference state initialisation) at a grasp frame with the object at its pose in the reference
               palm, where the glove fires firmly and the object clears the table; to the clip end. The first episodes end at
               random frames (rl_recon/train.py), so the envs spread over the clip. Terminated when a palm, its
               keypoints or a task object drift too far. Evaluation (`eval_start`): from frame 0, or in hand from
               the first in-hand start frame (rl_recon/train.py's two evaluations).
  assist       during the reference's holds (Benchmark-Genesis: from 5 glove channels on, kept while one of the
               next 5 frames has them), a velocity-level controller pulls the held object to its pose in the palm,
               plus gravity compensation, scaled by `assist_scale` (decayed to 0 by the runner, off in evaluation).
  observation  rl_recon's, per hand: joint state, the reference 1, 2, 4 frames ahead (relative), its object in the
               palm frame with its target and long axis, the simulated glove (soft contact, distance), the glove
               contacts 1 and 5 frames ahead, the keypoint errors; the phase first and the last action last.

Additions for this pipeline's references, off on Benchmark-Genesis' (`start_clearance` of the reference): starts
need the reference taxels `start_clearance` away from every resting object (else it is ejected at the reset).
"""

from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np
import torch

from tv2s_data.reference import load_reference
from tv2s_env.common.bases.base_env import BaseEnv
from tv2s_env.common.utils.math_utils import matrix_to_rotvec, quat_to_rotmat, rotation_angle
from tv2s_env.sim.envs.config.schema import RlReconEnvArgs
from tv2s_env.sim.robots.floating_hands import FloatingWujiHand
from tv2s_env.tactile.mapping import TaxelSet
from tv2s_env.tactile.mesh_sdf import MeshDistance
from tv2s_env.tactile.sensor import GloveSensor

HAND_KEYS = ("wrist_pos", "wrist_rot", "qpos", "base_R0", "base_q", "ref_keypoints", "human_kp", "real_contact", "real_raw")
OBJECT_KEYS = ("obj_pos", "obj_rot", "obj_valid", "object_ids", "object_collision", "object_fixed", "object_mass", "obj_rest_z")


def load_taxels(path: str, side: str) -> TaxelSet:
    """The glove's taxels on the Wuji links (Benchmark-Genesis' wuji_taxels_<side>.npz)."""
    z = np.load(path, allow_pickle=False)
    return TaxelSet(side=side, method="benchmark_genesis", link=z["link"].astype(str), pos=z["pos"], normal=z["normal"],
                    channel=z["channel"], n_channels=len(z["raw_ids"]))


class RlReconEnv(BaseEnv):
    def __init__(self, args: RlReconEnvArgs, num_envs: int, show_viewer: bool = False,
                 device: torch.device = torch.device("cuda"), eval_mode: bool = False,
                 render: dict | None = None, camera_res: tuple[int, int] | None = None) -> None:
        super().__init__(device=device)
        self._args = args
        self._eval_mode = eval_mode
        self._render = render
        self._num_envs = num_envs
        self.step_callback = None
        self.disable_termination = False
        self.assist_scale = 0.0
        self.training_step = 0
        self.generator = torch.Generator(device=device).manual_seed(args.seed)
        if not gs._initialized:  # noqa: SLF001
            gs.init(backend=gs.gpu if device.type == "cuda" else gs.cpu, precision="32",
                    logging_level="warning", seed=args.seed, performance_mode=True)
        self._load_reference()
        self._build_scene(show_viewer)
        self._build_sensors()
        self._init_buffers()
        self.reset()
        if not eval_mode:          # rl_recon/train.py: the first episodes end at random frames
            self.horizon[:] = torch.randint(min(20, self.T - 2), self.T, (num_envs,), device=device, generator=self.generator)

    # ================================================================== data
    def _load_reference(self) -> None:
        import trimesh

        a = self._args
        path = Path(a.data.reference)
        r = load_reference(path)
        sides = list(r["sides"])
        keep = [i for i, s in enumerate(sides) if a.active_hands in ("reference", s)]
        if not keep:
            raise ValueError(f"{path}: hands {sides}, none is {a.active_hands}")
        self.sides, self.H = tuple(sides[i] for i in keep), len(keep)
        self.reference_path, self.clip_names, self.n_clips, self.episode = path, [path.parent.name], 1, True
        self.hz = float(r["fps"])
        self.dt = 1.0 / self.hz
        T = min(len(r["qpos"][0]), len(r["obj_pos"][0]), len(r["real_contact"][0]), len(r["cam_pos"]))
        self.T = T
        ref = {k: np.asarray(r[k])[keep][:, :T] if k not in ("base_R0",) else np.asarray(r[k])[keep] for k in HAND_KEYS}
        ref = {k: (np.moveaxis(v, 0, 1) if k not in ("base_R0",) else v) for k, v in ref.items()}   # (T, H, ...)
        self.base_R0 = {s: ref["base_R0"][h] for h, s in enumerate(self.sides)}
        self.taxel_paths = [str(r["taxels"][i]) for i in keep]
        self.human_kp_index = torch.as_tensor(r["human_kp_index"], device=self.device)

        # objects (T, N, ...)
        N = len(r["object_ids"])
        self.object_ids = [str(x) for x in r["object_ids"]]
        self.collision_paths = [str(x) for x in r["object_collision"]]
        self.obj_free = ~np.asarray(r["object_fixed"], bool)
        mass = np.asarray(r["object_mass"], float)
        self.obj_mass_np = np.where(np.isfinite(mass) & (a.object_mass <= 0), mass, a.object_mass or 0.08)
        ref["obj_pos"] = np.moveaxis(r["obj_pos"][:, :T], 0, 1)
        ref["obj_rot"] = np.moveaxis(r["obj_rot"][:, :T], 0, 1)
        ref["obj_valid"] = np.moveaxis(r["obj_valid"][:, :T], 0, 1)
        ref["obj_weight"] = np.moveaxis(r["obj_weight"][:, :T], 0, 1)                    # (T, N)
        ref["obj_margin_scale"] = np.moveaxis(r["obj_margin_scale"][:, :T], 0, 1)
        ref["obj_quat"] = np.stack([_rotmats_to_quats(ref["obj_rot"][:, k]) for k in range(N)], 1)
        ref["head_pos"] = np.asarray(r["cam_pos"])[:T]
        meshes = [trimesh.load(p, force="mesh") for p in self.collision_paths]
        hulls = [np.asarray(m.convex_hull.vertices) for m in meshes]

        # holds of the simulated hands -> per frame: holding, held object, pose in the palm
        holding, held = np.zeros((T, self.H), bool), np.full((T, self.H), -1)
        anchor = np.tile(np.eye(4), (T, self.H, 1, 1))
        self.holds = []
        for (hi, k, h0, h1), R, p in zip(r["holds"], r["hold_R_palm"], r["hold_p_palm"]):
            if hi not in keep or not self.obj_free[k]:
                continue
            h = keep.index(hi)
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = R, p
            holding[h0:h1, h], held[h0:h1, h], anchor[h0:h1, h] = True, k, A
            self.holds.append((h, int(k), int(h0), min(int(h1), T), A))
        ref["holding"], ref["held"], ref["anchor"] = holding, held, anchor
        self.task = np.zeros(N, bool)
        for _, k, _, _, _ in self.holds:
            self.task[k] = True
        if not self.task.any():
            self.task = self.obj_free.copy()
        if not self.task.any():
            raise ValueError(f"{path}: no free object")

        # each hand's object: the one it holds now, else next, else last (its only one in rl_recon)
        focus = np.full((T, self.H), -1)
        for h in range(self.H):
            mine = sorted((h0, h1, k) for hh, k, h0, h1, _ in self.holds if hh == h)
            default = mine[0][2] if mine else int(np.flatnonzero(self.task)[0])
            for t in range(T):
                now = [k for h0, h1, k in mine if h0 <= t < h1]
                nxt = [k for h0, h1, k in mine if h0 > t]
                last = [k for h0, h1, k in mine if h1 <= t]
                focus[t, h] = now[0] if now else nxt[0] if nxt else last[-1] if last else default
        ref["focus"] = focus

        # long axes (object frame: principal axis, towards the end farther from the centre: the shaft tip)
        axes = []
        for m in meshes:
            V = np.asarray(m.vertices)
            ax = np.linalg.eigh((V - V.mean(0)).T @ (V - V.mean(0)))[1][:, -1]
            proj = V @ ax
            axes.append(ax if proj.max() > -proj.min() else -ax)
        self.obj_axis_np = np.array(axes)
        self.hull_inertia = [m.convex_hull.moment_inertia / max(m.convex_hull.volume, 1e-9) for m in meshes]   # per kg
        self.obj_volume = [m.volume if m.is_watertight else m.convex_hull.volume for m in meshes]

        # resting pose lift: the reference's obj_rest_z at frame 0 (rl_recon), else out of the table
        zmin = np.stack([np.einsum("vj,tij->tvi", V, ref["obj_rot"][:, k])[..., 2].min(1) + ref["obj_pos"][:, k, 2]
                         for k, V in enumerate(hulls)], 1)                                   # (T, N)
        lift = np.maximum(0.0, -zmin)
        rest_z = np.asarray(r["obj_rest_z"], float)
        lift[0] = np.where(np.isfinite(rest_z), rest_z, lift[0])
        ref["rest_lift"] = lift

        # start frames: before the first grasp (rl_recon: up to 10 frames before it) or away from every hold
        margin = int(round(a.rest_margin_s * self.hz))
        n_contact = ref["real_contact"].sum(-1)                                               # (T, H)
        if a.rest_starts == "before_first_grasp":
            first = np.flatnonzero((n_contact >= 5).any(-1))
            smax = max(0, (int(first[0]) if len(first) else T // 2) - margin)
            self.rest_frames = np.arange(min(smax, T - 2) + 1)
        else:
            near = np.zeros(T, bool)
            for _, _, h0, h1, _ in self.holds:
                near[max(0, h0 - margin):min(T, h1 + margin)] = True
            self.rest_frames = np.flatnonzero(~near[: T - 1])
        # in-hand starts: the reference's candidates where the held object, at its pose in the reference palm,
        # clears the table and the glove fires at least rsi_min_contact_frac of its peak
        self.rsi_hold = {}
        for t in np.asarray(r["rsi_frames"], int):
            if t >= T - 1 or not self.holds:
                continue
            i = min(range(len(self.holds)), key=lambda j: 0 if self.holds[j][2] <= t < self.holds[j][3]
                    else min(abs(t - self.holds[j][2]), abs(t - self.holds[j][3])))
            h, k, _, _, A = self.holds[i]
            P = np.eye(4)
            P[:3, :3], P[:3, 3] = ref["wrist_rot"][t, h], ref["wrist_pos"][t, h]
            O = P @ A
            low = (hulls[k] @ O[:3, :3].T + O[:3, 3])[:, 2].min() <= a.rsi_min_clearance
            soft = n_contact[t, h] < a.rsi_min_contact_frac * n_contact[:, h].max()
            if not (low or soft):
                self.rsi_hold[int(t)] = i
        self.rsi_frames = np.array(sorted(self.rsi_hold), int)

        # tensors (1, T, ...): ref[key][env_clip, t] reads every env's frame
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)[None]
        self.ref = {k: f(v) for k, v in ref.items() if k not in ("holding", "held", "focus", "obj_valid")}
        self.ref["holding"] = torch.as_tensor(ref["holding"], device=self.device)[None]
        self.ref["obj_valid"] = torch.as_tensor(ref["obj_valid"], device=self.device)[None]
        for k in ("held", "focus"):
            self.ref[k] = torch.as_tensor(ref[k], dtype=torch.long, device=self.device)[None]
        self.ref["contact"] = self.ref["real_contact"]
        axis_ref = self.ref["obj_rot"][0] @ torch.as_tensor(self.obj_axis_np, dtype=torch.float32, device=self.device)[..., None]
        self.ref["axis_img"] = self._image_axis(axis_ref[..., 0], self.ref["obj_pos"][0] - self.ref["head_pos"][0][:, None])[None]
        self.mask_axis = None
        if "mask_axis_2d" in r:                          # Benchmark-Genesis' mask_axes.py: the orientation target
            g = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
            self.mask_axis = g(np.moveaxis(r["mask_axis_2d"][:, :T], 0, 1))                   # (T, N, 2)
            self.mask_valid = torch.as_tensor(np.moveaxis(r["mask_axis_valid"][:, :T], 0, 1), device=self.device)
            self.cam_R, self.cam_t, self.cam_K = g(r["cam_R"][:T]), g(r["cam_t"][:T]), g(r["cam_K"])
            self.C_w, self.shift_w = g(r["C"]), g(r["shift"])
        self.start_clearance = a.start_clearance if a.start_clearance >= 0 else float(r.get("start_clearance", 0.0))
        self.lengths_t = torch.tensor([T], device=self.device)
        held_ids = sorted({self.object_ids[k] for k in np.flatnonzero(self.task)})
        print(f"Reference {path} ({T} frames at {self.hz:g} Hz, {T / self.hz:.1f} s): hands {list(self.sides)}, "
              f"{N} objects (task: {held_ids}), {len(self.holds)} holds, {len(self.rest_frames)} rest / "
              f"{len(self.rsi_frames)} in-hand start frames" + (", mask axis target" if self.mask_axis is not None else ""))

    def ref_at(self, key: str, t: torch.Tensor) -> torch.Tensor:
        """Reference `key` at frames t (B,), clamped to the clip."""
        return self.ref[key][0, t.clamp(0, self.T - 1)]

    # ================================================================== scene
    def _build_scene(self, show_viewer: bool) -> None:
        from scipy.spatial.transform import Rotation

        a = self._args
        vis = dict(shadow=False, ambient_light=(0.45, 0.45, 0.45), background_color=(0.85, 0.88, 0.92)) \
            if self._render is not None else {}
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=a.substeps, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, enable_joint_limit=True,
                                                  constraint_solver=gs.constraint_solver.Newton,
                                                  max_collision_pairs=200),
            vis_options=gs.options.VisOptions(show_world_frame=False, plane_reflection=False, **vis),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.hands = [FloatingWujiHand(self.scene, s, self.base_R0[s], a.hands, a.work_dir, self.device) for s in self.sides]
        self.scene_entities = []
        for k, path in enumerate(self.collision_paths):
            p, R = self.ref["obj_pos"][0, 0, k].cpu().numpy(), self.ref["obj_rot"][0, 0, k].cpu().numpy()
            self.scene_entities.append(self.scene.add_entity(
                gs.morphs.Mesh(file=path, fixed=not self.obj_free[k], convexify=True, decimate=False,
                               decompose_object_error_threshold=a.decompose_threshold, pos=tuple(map(float, p)),
                               quat=tuple(map(float, Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]))),
                material=gs.materials.Rigid(rho=float(self.obj_mass_np[k] / max(self.obj_volume[k], 1e-6)),
                                            friction=a.object_friction),
                surface=gs.surfaces.Default(color=(0.75, 0.45, 0.85) if self.task[k] else (0.55, 0.6, 0.55))))
        if self._render is not None:
            self._add_render_entities()
        self.scene.build(n_envs=self._num_envs, env_spacing=(3.0, 3.0) if self._render is not None else (1.0, 1.0))
        for hand in self.hands:
            hand.post_build(self._num_envs)
        dev = self.device
        self.obj_link = [e.links[0].idx for e in self.scene_entities]
        self.obj_mass = torch.as_tensor([float(e.get_mass()) for e in self.scene_entities], dtype=torch.float32, device=dev)
        self.obj_inertia = torch.as_tensor(np.array(self.hull_inertia), dtype=torch.float32, device=dev)
        self.obj_axis = torch.as_tensor(self.obj_axis_np, dtype=torch.float32, device=dev)
        self.task_t = torch.as_tensor(self.task, device=dev)
        self.env_clip = torch.zeros(self._num_envs, dtype=torch.long, device=dev)

    def _add_render_entities(self) -> None:
        """Videos (scripts/policy/visualize_episode.py): one camera, translucent ghosts of the reference hands and
        objects (no collision), put on the reference before each render."""
        res = tuple(self._render.get("res", (480, 360)))
        self.camera = self.scene.add_camera(res=res, pos=(0.5, -0.5, 0.5), lookat=(0.0, 0.0, 0.0),
                                            fov=float(self._render.get("fov", 45)), env_idx=0, GUI=False, near=0.02)
        self.ghosts = []
        for h, s in enumerate(self.sides):
            urdf = self.hands[h].model.write_floating_urdf(str(Path(self._args.work_dir) / f"wuji_{s}_ghost.urdf"),
                                                           self.base_R0[s], base_mass=0.01)
            self.ghosts.append(self.scene.add_entity(
                gs.morphs.URDF(file=urdf, fixed=True, merge_fixed_links=True, collision=False),
                material=gs.materials.Rigid(gravity_compensation=1.0),
                surface=gs.surfaces.Default(color=(1.0, 0.45, 0.0, 0.45))))
        self.object_ghosts = [None if not free else self.scene.add_entity(
            gs.morphs.Mesh(file=p, fixed=False, collision=False), material=gs.materials.Rigid(gravity_compensation=1.0),
            surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2, 0.35))) for p, free in zip(self.collision_paths, self.obj_free)]

    @torch.no_grad()
    def set_reference_ghosts(self) -> None:
        from tv2s_env.robots.wuji_model import BASE_JOINTS
        envs = torch.arange(self._num_envs, device=self.device)
        if not hasattr(self, "_ghost_dofs"):
            self._ghost_dofs = [[g.get_joint(n).dofs_idx_local[0] for n in BASE_JOINTS + self.hands[h].model.dof_names]
                                for h, g in enumerate(self.ghosts)]
        for h, g in enumerate(self.ghosts):
            q = torch.cat([self.ref_at("base_q", self.t)[:, h], self.ref_at("qpos", self.t)[:, h]], -1)
            g.set_dofs_position(q, self._ghost_dofs[h], envs_idx=envs, zero_velocity=True)
        for k, g in enumerate(self.object_ghosts):
            if g is not None:
                g.set_pos(self.ref_at("obj_pos", self.t)[:, k], envs_idx=envs, zero_velocity=True)
                g.set_quat(self.ref_at("obj_quat", self.t)[:, k], envs_idx=envs, zero_velocity=True)

    def _build_sensors(self) -> None:
        t = self._args.tactile
        self.sensors = [GloveSensor(load_taxels(self.taxel_paths[h], s), self.hands[h].link_index, t.probe_radius, t.kn,
                                    self.device) for h, s in enumerate(self.sides)]
        if len({s.n_channels for s in self.sensors}) > 1:
            raise ValueError(f"the hands' gloves have different channel counts: {[s.n_channels for s in self.sensors]}")
        self.n_channels = self.sensors[0].n_channels
        self.sdf = [MeshDistance(p, self.device, voxel=self._args.sdf_voxel, signed=True) for p in self.collision_paths]
        self._clear_starts()

    @torch.no_grad()
    def _clear_starts(self) -> None:
        """With a start clearance: the frames where the reference taxels are at least that far from every resting
        object (an object placed into a hand at a reset is ejected). Rest starts keep those frames only, and a
        start at frame 0 moves to the first clear frame."""
        T, dev = self.T, self.device
        self.first_start = 0
        if self.start_clearance <= 0:
            return
        clear = torch.ones(T, dtype=torch.bool, device=dev)
        held = self.ref["held"][0]
        for h, (hand, sensor) in enumerate(zip(self.hands, self.sensors)):
            base = np.tile(np.eye(4), (T, 1, 1))
            base[:, :3, :3], base[:, :3, 3] = self.ref["wrist_rot"][0, :, h].cpu().numpy(), self.ref["wrist_pos"][0, :, h].cpu().numpy()
            fk = hand.model.fk(self.ref["qpos"][0, :, h].cpu().numpy(), base=base)
            names = {i: n for n, i in hand.link_index.items()}
            pose = torch.as_tensor(np.stack([fk[names[int(i)]] for i in sensor.link_ids.tolist()], 1), dtype=torch.float32, device=dev)
            p_w = (pose[..., :3, :3] @ sensor.pos[None, :, :, None])[..., 0] + pose[..., :3, 3]       # (T, P, 3)
            for k, sdf in enumerate(self.sdf):
                if not self.obj_free[k]:
                    continue
                d = sdf(((p_w - self.ref["obj_pos"][0, :, k, None])[..., None, :] @ self.ref["obj_rot"][0, :, k, None])[..., 0, :]).amin(-1)
                clear &= ~(~(held == k).any(-1) & (d < self.start_clearance))
        clear = clear.cpu().numpy()
        self.rest_frames = self.rest_frames[clear[self.rest_frames]]
        ok = np.flatnonzero(clear[: T - 1])
        self.first_start = int(ok[0]) if len(ok) else 0
        if not clear[0]:
            print(f"  the reference hands touch a resting object at frame 0: episodes from frame 0 start at frame "
                  f"{self.first_start} ({self.first_start / self.hz:.2f} s); {len(self.rest_frames)} rest start frames left")

    def _init_buffers(self) -> None:
        a, B, dev = self._args, self._num_envs, self.device
        self.t = torch.zeros(B, dtype=torch.long, device=dev)
        self.start = torch.zeros(B, dtype=torch.long, device=dev)
        self.horizon = torch.full((B,), self.T - 1, dtype=torch.long, device=dev)
        self.action_dim = 26 * self.H
        self.last_action = torch.zeros(B, self.action_dim, device=dev)
        self.action_rate = torch.zeros(B, device=dev)
        self.action_scale = torch.tensor([a.action_scale_pos] * 3 + [a.action_scale_rot] * 3
                                         + [a.action_scale_finger] * 20, device=dev).repeat(self.H)
        self.rest_frames_t = torch.as_tensor(self.rest_frames, device=dev)
        self.rsi_frames_t = torch.as_tensor(self.rsi_frames, device=dev)
        self._extra_info: dict[str, Any] = {}
        self._terminated = torch.zeros(B, dtype=torch.bool, device=dev)

    # ================================================================== state
    @staticmethod
    def _image_axis(axis: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
        """rl_recon's projected axis: the unit axis' component across the viewing ray (direction and
        foreshortening, deliberately not normalised)."""
        ray = ray / ray.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return axis - (axis * ray).sum(-1, keepdim=True) * ray

    def _pixel_axis(self, R: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Unit 2D direction (pixels) of the simulated long axes through the real camera of frame t (rl_recon's
        _pixel_axis). R (B, N, 3, 3), p (B, N, 3) -> (B, N, 2)."""
        axis = (R @ self.obj_axis[None, :, :, None])[..., 0]
        pts = torch.stack([p, p + 0.08 * axis], 2)                                       # (B, N, 2, 3)
        tt = t.clamp(0, self.T - 1)
        x = ((pts - self.shift_w) @ self.C_w) @ self.cam_R[tt].transpose(-1, -2)[:, None] + self.cam_t[tt][:, None, None]
        fu, fv, pu, pv = self.cam_K
        q = torch.stack([fu * x[..., 0] / x[..., 2] + pu, fv * x[..., 1] / x[..., 2] + pv], -1)
        d = q[:, :, 1] - q[:, :, 0]
        return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def update_buffers(self) -> None:
        """Read the simulator and the reference at frame t."""
        st = [hand.state() for hand in self.hands]
        stack = lambda k: torch.stack([s[k] for s in st], 1)
        q, dq = stack("q"), stack("dq")
        self.hand_dof_pos, self.hand_dof_vel = q[..., 6:], dq[..., 6:]
        self.base_dof_vel = dq[..., :6]
        self.base_pos, self.base_rot, self.base_quat = stack("palm_pos"), stack("palm_rot"), stack("palm_quat")
        self.keypoints, self.finger_link_pos = stack("keypoints"), stack("tip_pos")
        E = self.scene_entities
        self.all_object_pos = torch.stack([e.get_pos() for e in E], 1)                  # (B, N, 3)
        self.all_object_quat = torch.stack([e.get_quat() for e in E], 1)
        self.all_object_rot = quat_to_rotmat(self.all_object_quat)
        self.all_object_vel = torch.stack([e.get_vel() for e in E], 1)
        self.all_object_ang = torch.stack([e.get_ang() for e in E], 1)
        t, B, H = self.t, self._num_envs, self.H
        focus = self.ref_at("focus", t)                                                    # (B, H)
        self._focus = focus

        # simulated glove: signed distance of every taxel to the hand's object (or to the nearest object)
        dist, pressure = [], []
        for h, sensor in enumerate(self.sensors):
            p_w = sensor.taxel_world(st[h]["link_pos"], st[h]["link_rot"])               # (B, P, 3)
            d = torch.stack([sdf(((p_w - self.all_object_pos[:, k, None])[..., None, :] @ self.all_object_rot[:, k, None])[..., 0, :])
                             for k, sdf in enumerate(self.sdf)], 1)                        # (B, N, P)
            if self._args.tactile_objects == "manipulated":
                d = torch.gather(d, 1, focus[:, h, None, None].expand(-1, 1, d.shape[-1]))[:, 0]
            else:
                d = d.amin(1)
            dd, pp = sensor.reduce(d)
            dist.append(dd), pressure.append(pp)
        self.tactile_distance, self.tactile_pressure = torch.stack(dist, 1), torch.stack(pressure, 1)   # (B, H, C)

        # reference at t
        self.target_wrist_pos, self.target_wrist_rot = self.ref_at("wrist_pos", t), self.ref_at("wrist_rot", t)
        self.target_hand_dof_pos = self.ref_at("qpos", t)
        self.target_keypoints = self.ref_at("ref_keypoints", t)
        self.target_fingertip_pos = self.target_keypoints[..., 2::3, :]
        self.target_contact = self.ref_at("contact", t) > 0.5
        self.target_pressure = self.ref_at("real_raw", t)
        self.target_all_object_pos, self.target_all_object_rot = self.ref_at("obj_pos", t), self.ref_at("obj_rot", t)
        ray = self.target_all_object_pos - self.ref_at("head_pos", t)[:, None]
        self.camera_ray = ray / ray.norm(dim=-1, keepdim=True).clamp(min=1e-6)            # (B, N, 3)

        # each hand's object (visualisation, observations)
        take = lambda x: torch.gather(x, 1, focus.view(B, H, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))
        self.object_pos, self.object_quat, self.object_rot = take(self.all_object_pos), take(self.all_object_quat), take(self.all_object_rot)
        self.object_lin_vel, self.object_ang_vel = take(self.all_object_vel), take(self.all_object_ang)
        self.target_object_pos, self.target_object_rot = take(self.target_all_object_pos), take(self.target_all_object_rot)
        self.object_valid = take(self.ref_at("obj_valid", t))
        self._metrics = self.metrics()

    # ================================================================== metrics and reward
    def metrics(self) -> dict[str, torch.Tensor]:
        """rl_recon's reward terms and errors against frame t (ReconEnv.metrics): per hand then averaged, the
        object terms over the task objects."""
        a, t = self._args, self.t
        e_wrist = (self.base_pos - self.target_wrist_pos).norm(dim=-1)                     # (B, H)
        e_rot = rotation_angle(self.base_rot.transpose(-1, -2) @ self.target_wrist_rot)
        e_kp = (self.keypoints - self.target_keypoints).norm(dim=-1).mean(-1)
        e_human = (self.keypoints - self.ref_at("human_kp", t)[:, :, self.human_kp_index]).norm(dim=-1).mean(-1)
        r_hand = (0.4 * torch.exp(-e_wrist / a.sigma_wrist_pos) + 0.2 * torch.exp(-e_rot / a.sigma_wrist_rot)
                  + 0.4 * torch.exp(-e_kp / a.sigma_kp)).mean(-1)

        c = a.tactile
        d, y = self.tactile_distance, self.target_contact.float()                          # (B, H, C)
        soft = torch.sigmoid((c.probe_radius - d) / c.soft_tau)
        f1_soft = (2 * (soft * y).sum(-1) + 1) / (soft.sum(-1) + y.sum(-1) + 1)
        n_y = y.sum(-1)
        reach = (torch.exp(-(d - c.probe_radius).clamp(min=0) / a.tac_reach_scale) * y).sum(-1) / n_y.clamp(min=1)
        reach = torch.where(n_y > 0, reach, torch.ones_like(reach))
        r_tac = ((1 - a.tac_reach_weight) * f1_soft + a.tac_reach_weight * reach).mean(-1)
        sim_contact = d < c.probe_radius
        tp = (sim_contact & (y > 0)).sum(-1).float()

        task = self.task_t.float()                                                          # (N,)
        valid = self.ref_at("obj_valid", t).float() * task                                  # (B, N)
        # the reference's per-frame weight and margin scale of the object terms (keyframe_object_trajectory: strict
        # where the object's pose is measured, loose while it is only assumed in the hand)
        weight, mscale = self.ref_at("obj_weight", t), self.ref_at("obj_margin_scale", t)
        n_task = task.sum()
        err = self.all_object_pos - self.target_all_object_pos                              # (B, N, 3)
        along = (err * self.camera_ray).sum(-1)
        across = (err - along[..., None] * self.camera_ray).norm(dim=-1)
        e_obj = torch.sqrt(across ** 2 + (along / a.obj_depth_discount) ** 2)
        e_obj_rot = rotation_angle(self.all_object_rot.transpose(-1, -2) @ self.target_all_object_rot)
        axis_img = self._image_axis((self.all_object_rot @ self.obj_axis[None, :, :, None])[..., 0], self.camera_ray)
        e_axis = 2 * torch.asin(((axis_img - self.ref_at("axis_img", t)).norm(dim=-1) / 2).clamp(max=1.0))
        valid_axis = valid * weight
        if self.mask_axis is not None:
            tt = t.clamp(0, self.T - 1)
            e_mask = torch.acos((self._pixel_axis(self.all_object_rot, self.all_object_pos, t) * self.mask_axis[tt]).sum(-1).clamp(-1, 1))
            e_axis = torch.where(self.mask_valid[tt], e_mask, e_axis)
            # the masks' axis is measured in the image itself: it scores the frames the reference trajectory leaves
            # out too (a reference whose object target is set on its measured frames only)
            valid_axis = torch.maximum(valid * weight, self.mask_valid[tt].float() * task)
        hinge_pos = ((across - a.obj_margin_across * mscale).clamp(min=0) / a.obj_scale_across
                     + (along.abs() - a.obj_margin_along * mscale).clamp(min=0) / a.obj_scale_along)
        hinge_rot = ((e_axis - a.obj_axis_margin).clamp(min=0) / a.obj_axis_scale
                     + valid * weight * a.obj_rot_weight * (e_obj_rot - a.obj_rot_margin).clamp(min=0) / 0.5)
        p_pos = ((1 - torch.exp(-hinge_pos)) * valid * weight).sum(-1) / n_task
        p_axis = ((1 - torch.exp(-hinge_rot)) * valid_axis).sum(-1) / n_task
        mean_task = lambda x: (x * task).sum(-1) / n_task
        return dict(r_hand=r_hand, r_tac=r_tac, p_pos=p_pos, p_axis=p_axis, f1_soft=f1_soft.mean(-1),
                    e_wrist_h=e_wrist, e_kp_h=e_kp, e_obj_task=e_obj * valid,
                    e_rot=e_rot.mean(-1), e_human=e_human.mean(-1), e_axis=mean_task(e_axis),
                    e_obj_across=mean_task(across), e_obj_along=mean_task(along.abs()), e_obj_depth=mean_task(e_obj),
                    e_obj_3d=mean_task(err.norm(dim=-1)), e_obj_3d_max=(err.norm(dim=-1) * task).amax(-1),
                    obj_height=mean_task(self.all_object_pos[..., 2] - self.ref["obj_pos"][0, 0, :, 2]),
                    tac_tp=tp.sum(-1), tac_sim=sim_contact.sum((-1, -2)).float(), tac_real=n_y.sum(-1))

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, m = self._args, self._metrics
        w_tac = a.w_tac if a.tactile_reward else 0.0
        terms = {"tactile": w_tac * m["r_tac"], "hand": a.w_hand * m["r_hand"], "object_position": a.w_obj * (1 - m["p_pos"]),
                 "object_axis": a.w_obj_axis * (1 - m["p_axis"]), "action_rate": -a.w_rate * self.action_rate}
        total = sum(terms.values())
        total = torch.where(self._extra_info["termination"]["error"], torch.zeros_like(total), total)
        terms["Total"] = total
        return total, terms

    def common_metrics(self) -> dict[str, torch.Tensor]:
        """What the runner logs and eval_rl_recon.py scores: rl_recon's terms and errors."""
        m = self._metrics
        return dict(
            tac_tp=m["tac_tp"], tac_sim=m["tac_sim"], tac_real=m["tac_real"],
            f1_soft=m["f1_soft"], r_tac=m["r_tac"], r_hand=m["r_hand"], p_pos=m["p_pos"], p_axis=m["p_axis"],
            e_wrist=m["e_wrist_h"].mean(-1), e_rot=m["e_rot"], e_kp=m["e_kp_h"].mean(-1), e_human=m["e_human"],
            e_tip=(self.finger_link_pos - self.target_fingertip_pos).norm(dim=-1).mean((-1, -2)),
            e_obj=m["e_obj_3d"], e_obj_depth=m["e_obj_depth"], e_obj_end=m["e_obj_3d_max"],
            e_obj_across=m["e_obj_across"], e_obj_along=m["e_obj_along"], e_axis=m["e_axis"], obj_height=m["obj_height"],
            obj_valid=torch.ones(self._num_envs, device=self.device),
            assist=torch.full((self._num_envs,), self.assist_scale, device=self.device))

    def get_extra_infos(self) -> dict[str, Any]:
        self._extra_info["metrics"] = self.common_metrics()
        return self._extra_info

    # ================================================================== observation
    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """rl_recon's observation (ReconEnv.observe), hands concatenated per term."""
        a, B, t, H = self._args, self._num_envs, self.t, self.H
        flat = lambda x: x.reshape(B, -1)
        parts = [(t.float() / self.T)[:, None], flat(self.hand_dof_pos), flat(0.1 * self.hand_dof_vel), flat(0.1 * self.base_dof_vel)]
        for k in a.obs_future:
            tk = t + k
            parts += [flat(self.ref_at("wrist_pos", tk) - self.base_pos),
                      flat(matrix_to_rotvec(self.base_rot.transpose(-1, -2) @ self.ref_at("wrist_rot", tk))),
                      flat(self.ref_at("qpos", tk) - self.hand_dof_pos)]
        t1 = t + 1
        gather = lambda x: torch.gather(x, 1, self._focus.view(B, H, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))
        rel = ((self.object_pos - self.base_pos)[..., None, :] @ self.base_rot)[..., 0, :]
        oR_p = self.base_rot.transpose(-1, -2) @ self.object_rot
        axis = (self.object_rot @ self.obj_axis[self._focus][..., None])[..., 0]
        parts += [flat(rel), flat(oR_p[..., :2]), flat(self.object_lin_vel), flat(0.1 * self.object_ang_vel),
                  flat(gather(self.ref_at("obj_pos", t1)) - self.object_pos), flat(self.object_pos[..., 2:]),
                  flat(gather(self.ref_at("axis_img", t1))), flat(axis)]
        c = a.tactile
        d = self.tactile_distance
        parts += [flat(torch.sigmoid((c.probe_radius - d) / c.soft_tau)), flat(torch.tanh(d / 0.03))]
        glove = 1.0 if a.glove_obs else 0.0
        for k in a.obs_tactile_future:
            parts.append(flat(glove * self.ref_at("contact", t + k)))
        parts += [flat(self.ref_at("ref_keypoints", t1) - self.keypoints), self.last_action]
        obs = torch.cat(parts, -1)
        return obs, obs

    # ================================================================== stepping
    def apply_action(self, action: torch.Tensor) -> None:
        action = action.detach().to(self.device).clamp(-1, 1)
        self.action_rate = ((action - self.last_action) ** 2).mean(-1).clamp(max=1.0)
        self.last_action = action.clone()
        t1 = self.t + 1
        ref = torch.cat([self.ref_at("base_q", t1), self.ref_at("qpos", t1)], -1)           # (B, H, 26)
        delta = (action * self.action_scale).reshape(self._num_envs, self.H, 26)
        for h, hand in enumerate(self.hands):
            hand.control(ref[:, h] + delta[:, h])
        if self.assist_scale > 0 and not self._eval_mode:
            self._assist()
        self.scene.step()
        self.t += 1
        self.update_buffers()

    def _assist(self) -> None:
        """rl_recon's hand-anchored virtual object controller (ReconEnv._assist): while a simulated hand holds an
        object in the reference, a velocity-level pull to its pose in the simulated palm (a fraction `gain` of the
        velocity error that closes the pose error in `tau`), plus gravity compensation; scaled by `assist_scale`."""
        a, t1, B = self._args, self.t + 1, self._num_envs
        holding, held = self.ref_at("holding", t1), self.ref_at("held", t1)                # (B, H)
        if not holding.any():
            return
        anchor = self.ref_at("anchor", t1)                                                # (B, H, 4, 4)
        st = [hand.state() for hand in self.hands]
        palm_p = torch.stack([s["palm_pos"] for s in st], 1)
        palm_R = torch.stack([s["palm_rot"] for s in st], 1)
        p_target = palm_p + (palm_R @ anchor[..., :3, 3:])[..., 0]
        R_target = palm_R @ anchor[..., :3, :3]
        solver = self.scene.sim.rigid_solver
        envs = torch.arange(B, device=self.device)
        for k, e in enumerate(self.scene_entities):
            sel = holding & (held == k)
            if not self.obj_free[k] or not sel.any():
                continue
            h = sel.float().argmax(-1)
            on = sel.any(-1).float()[:, None]
            p, v, w = e.get_pos(), e.get_vel(), e.get_ang()
            R = quat_to_rotmat(e.get_quat())
            m = self.obj_mass[k]
            gain = self.assist_scale * a.assist_gain * on
            force = gain * m * ((p_target[envs, h] - p) / a.assist_tau - v) / self.dt
            force[:, 2:] += self.assist_scale * on * m * 9.81
            w_des = matrix_to_rotvec(R_target[envs, h] @ R.transpose(-1, -2)) / a.assist_tau
            I_world = R @ (self.obj_inertia[k] * m) @ R.transpose(-1, -2)
            torque = gain * (I_world @ (w_des - w)[..., None])[..., 0] / self.dt
            solver.apply_links_external_force(force[:, None], [self.obj_link[k]], ref="link_com")
            solver.apply_links_external_torque(torque[:, None], [self.obj_link[k]], ref="link_com")

    def set_training_progress(self, iteration: int, total_iterations: int) -> None:
        """The assist decays linearly to 0 over the first `assist_decay` of training (rl_recon/train.py)."""
        a = self._args
        self.training_step = iteration
        self.assist_scale = a.assist_init * max(0.0, 1.0 - iteration / max(1.0, a.assist_decay * total_iterations))

    def get_terminated(self) -> torch.Tensor:
        a, m = self._args, self._metrics
        bad = ~torch.isfinite(self.all_object_pos).all(-1).all(-1) | ~torch.isfinite(self.base_pos).all(-1).all(-1)
        drift = ((m["e_wrist_h"] > a.term_wrist) | (m["e_kp_h"] > a.term_kp)).any(-1) | (m["e_obj_task"] > a.term_obj).any(-1)
        terminated = bad if self.disable_termination else (drift | bad)
        self._extra_info["termination"] = dict(terminated=terminated, drift=drift, error=bad)
        self._terminated = terminated
        return terminated

    def get_truncated(self) -> torch.Tensor:
        truncated = (self.t >= self.horizon) & ~self._terminated
        self._extra_info["termination"]["truncated"] = truncated
        return truncated

    def step(self, action: torch.Tensor):
        """apply -> terminated / truncated -> reward -> infos -> reset done envs -> observations."""
        self.apply_action(action)
        terminated = self.get_terminated()
        truncated = self.get_truncated()
        reward, terms = self.get_reward()
        infos = self.get_extra_infos()
        infos["reward_terms"] = terms
        infos["episode_length"] = (self.t - self.start).clone()
        if self.step_callback is not None:
            self.step_callback(terminated, truncated, infos)
        done = (terminated | truncated).nonzero(as_tuple=True)[0]
        if len(done) > 0:
            self.reset_idx(done)
        actor_obs, critic_obs = self.get_observations()
        return actor_obs, critic_obs, reward[:, None], terminated[:, None], truncated[:, None], infos

    # ================================================================== reset
    def _reset_hands_and_objects(self, envs_idx: torch.Tensor, start: torch.Tensor, in_hand: torch.Tensor | None = None) -> None:
        """Hands on the reference at `start`. Objects: an in-hand start (`in_hand` (n,) bool) puts the held object at
        its pose in the reference palm (rl_recon: the tactile grasp pose); a resting object goes to its reference
        pose, lifted out of the table (frame 0: rl_recon's obj_rest_z) plus 2 mm."""
        n = len(envs_idx)
        for h, hand in enumerate(self.hands):
            hand.set_state(torch.cat([self.ref_at("base_q", start)[:, h], self.ref_at("qpos", start)[:, h]], -1), envs_idx)
        in_hand = torch.zeros(n, dtype=torch.bool, device=self.device) if in_hand is None else in_hand
        # the in-hand object of each env: the hold of its start frame (rsi_hold)
        hold_obj = torch.full((n,), -1, dtype=torch.long, device=self.device)
        P_in = torch.eye(4, device=self.device).repeat(n, 1, 1)
        for i in torch.nonzero(in_hand).flatten().tolist():
            t = int(start[i])
            j = self.rsi_hold.get(t)
            if j is None:
                continue
            h, k, _, _, A = self.holds[j]
            P = torch.eye(4, device=self.device)
            P[:3, :3], P[:3, 3] = self.ref["wrist_rot"][0, t, h], self.ref["wrist_pos"][0, t, h]
            P_in[i] = P @ torch.as_tensor(A, dtype=torch.float32, device=self.device)
            hold_obj[i] = k
        for k, e in enumerate(self.scene_entities):
            if not self.obj_free[k]:
                continue
            pos = self.ref_at("obj_pos", start)[:, k].clone()
            pos[:, 2] += self.ref_at("rest_lift", start)[:, k] + 0.002
            quat = self.ref_at("obj_quat", start)[:, k].clone()
            mine = hold_obj == k
            if mine.any():
                pos[mine] = P_in[mine, :3, 3]
                quat[mine] = torch.as_tensor(_rotmats_to_quats(P_in[mine, :3, :3].cpu().numpy()), dtype=torch.float32, device=self.device)
            e.set_pos(pos, envs_idx=envs_idx, zero_velocity=True)
            e.set_quat(quat, envs_idx=envs_idx, zero_velocity=True)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        a, n, dev, gen = self._args, len(envs_idx), self.device, self.generator
        start = torch.full((n,), self.first_start, dtype=torch.long, device=dev)
        in_hand = torch.zeros(n, dtype=torch.bool, device=dev)
        if self._eval_mode:
            if a.eval_start == "in_hand" and len(self.rsi_frames_t):
                start[:], in_hand[:] = self.rsi_frames_t[0], True
        elif a.train_start == "random":
            if len(self.rest_frames_t) and a.start_frame_prob > 0:
                pick = torch.rand(n, device=dev, generator=gen) < a.start_frame_prob
                k = torch.randint(0, len(self.rest_frames_t), (n,), device=dev, generator=gen)
                start = torch.where(pick, self.rest_frames_t[k], start)
            if len(self.rsi_frames_t) and a.rsi_grasp_prob > 0:
                in_hand = torch.rand(n, device=dev, generator=gen) < a.rsi_grasp_prob
                k = torch.randint(0, len(self.rsi_frames_t), (n,), device=dev, generator=gen)
                start = torch.where(in_hand, self.rsi_frames_t[k], start)
        self._reset_hands_and_objects(envs_idx, start, in_hand)
        self.t[envs_idx], self.start[envs_idx] = start, start
        horizon = torch.full_like(start, self.T - 1)
        if not self._eval_mode and a.train_start == "random" and a.episode_window_s > 0:
            horizon = torch.minimum(start + int(round(a.episode_window_s * self.hz)), horizon)
        self.horizon[envs_idx] = horizon
        self.last_action[envs_idx] = 0.0
        self.action_rate[envs_idx] = 0.0
        self.update_buffers()

    def close(self) -> None:
        pass

    @property
    def num_envs(self) -> int:
        return self._num_envs


def _rotmats_to_quats(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
