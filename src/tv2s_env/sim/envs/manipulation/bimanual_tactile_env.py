"""Bimanual HumanTouch environment shared by the DexTaG and Benchmark-Genesis (rl_recon) methods.

Scene: two Wuji hands on virtual bases (FloatingWujiHand), one proxy object per hand (a free cylinder,
see tv2s_data/clips.py) and one support plate per hand under its object, plus a ground plane. Each
environment replays one clip: the clips are spread over the environments in balanced blocks, the layout
Genesis uses for heterogeneous entities (each clip has its own cylinder sizes). A hand without an object
in a clip gets a parked object on the ground, far away, and no support.

Named buffers (attributes, shape (B, H, ...) with H = len(self.sides) hands, left then right) hold the simulated state,
the reference at the current frame and the simulated / reference glove; reward terms read them by name
(DexTaG's `required_keys`), observations are built from them by the method subclasses.

Episode clips (scripts/data/build_episode_clip.py, `self.episode`): one clip is a whole demonstration with
its reconstructed objects -- every RecGen mesh as a Genesis body (fixed or free, its mass) on its tracked
trajectory, and per hand an object schedule (the object Gemini says the hand manipulates, frame by frame).
Each hand's "object" buffers are its scheduled object's, so the per-hand reward terms are unchanged; the
glove reads the distance to every object (tactile/mesh_sdf.py grids). Training episodes play the whole
demonstration from frame 0, as evaluation (`train_start=random`: they start anywhere, `episode_manip_weight`
times as often inside a manipulation, and last `episode_window_s`). rl_recon's in-palm assist holds each hand's
scheduled object during the reference's holds (ReconBimanualEnv._assist_episode); DexTaG's object aids
(guide and kicks) assume one object per hand and are off in this mode.

Hands: `active_hands` (BimanualEnvArgs) keeps both hands, one side, or the hands that manipulate an object
("auto"); a hand left out is not built, so H is 1 and the actions, observations and rewards are one hand's.

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

from tv2s_data.clips import cylinder_half_height, settle_after_release, settle_before_grasp
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
from tv2s_env.tactile.mesh_sdf import MeshDistance, collision_mesh
from tv2s_env.tactile.sensor import GloveSensor

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
                 device: torch.device = torch.device("cuda"), eval_mode: bool = False,
                 render: dict | None = None, camera_res: tuple[int, int] | None = None) -> None:
        super().__init__(device=device)
        self._args = args
        self._eval_mode = eval_mode
        self._render = render     # None, or {"res": (w, h)}: one camera per env + reference ghost hands
        self._camera_res = camera_res  # None, or (w, h): ego + front camera per env (visualize_policy.py)
        self.step_callback = None      # called after each step, before done envs are reset (visualisation)
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
        self.sides = self._active_sides(clips)
        self.H = len(self.sides)
        self.clip_names = names
        self.n_clips = len(clips)
        self.hz = float(clips[0]["metadata"]["hz"])
        self.dt = 1.0 / self.hz
        self.lengths = np.array([c["metadata"]["end"] - c["metadata"]["start"] for c in clips])
        T = int(self.lengths.max())
        self.T_max = T
        meta = clips[0]["metadata"]
        self.layouts = [load_glove_layout(meta["tactile_mapping"], s, finger_tip_row=meta["finger_tip_row"])
                        for s in self.sides]
        # glove units per observation unit: the clip's own (EgoTouch: 1, normalised) else the config's
        self.glove_scale = meta.get("glove_scale") or self._args.tactile.glove_scale
        self.clip_taxels = meta.get("taxels")

        def pad(x):
            return np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)], 0) if len(x) < T else x

        # base rotation of each hand's virtual base: the mean wrist orientation over its clips
        rots = {s: [] for s in self.sides}
        for c in clips:
            for s in self.sides:
                q = c["hands"][s]["wrist_quat"]
                rots[s].append(_quats_to_rotmats(q))
        self.base_R0 = {s: mean_rotation(np.concatenate(rots[s])) for s in self.sides}

        keys = {k: [] for k in ["wrist_pos", "wrist_rot", "wrist_quat", "wrist_vel", "wrist_ang_vel", "base_q",
                                 "dof_pos", "dof_vel", "tip_pos", "tip_quat", "keypoints", "pressure", "contact",
                                 "obj_pos", "obj_rot", "obj_quat", "obj_vel", "obj_ang_vel", "obj_valid", "holding"]}
        self.episode = "scene_objects" in clips[0]
        if self.episode:
            return self._load_episode(clips[0], keys, clip_dir)
        self.obj_spec = np.zeros((self.n_clips, self.H, 3))          # radius, length, mass
        self.obj_active = np.zeros((self.n_clips, self.H), bool)
        self.support_z = np.zeros((self.n_clips, self.H))
        self.place_z = np.full((self.n_clips, self.H), -1.0)          # top of the placing support, -1: none
        self.grasp = np.zeros((self.n_clips, self.H, 2), int)
        self.pose_in_palm = np.tile(np.eye(4), (self.n_clips, self.H, 1, 1))
        head = []
        for i, c in enumerate(clips):
            per = {k: [] for k in keys}
            L = self.lengths[i]
            for h, s in enumerate(self.sides):
                d = c["hands"][s]
                R = _quats_to_rotmats(d["wrist_quat"])
                pos = d["wrist_positions"].astype(np.float64)
                per["wrist_pos"].append(pos), per["wrist_rot"].append(R), per["wrist_quat"].append(d["wrist_quat"])
                per["wrist_vel"].append(np_velocity(pos, self.dt)), per["wrist_ang_vel"].append(np_angular_velocity(R, self.dt))
                per["base_q"].append(base_coordinates(self.base_R0[s], pos, R))
                per["dof_pos"].append(d["dof_positions"]), per["dof_vel"].append(np_velocity(d["dof_positions"], self.dt))
                per["tip_pos"].append(d["fingertip_pos"]), per["tip_quat"].append(np_quat_continuity(d["fingertip_quat"]))
                per["keypoints"].append(d["ref_keypoints"])
                per["pressure"].append(d["tactile_pressure"] / self.glove_scale)
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
                    # a stable resting start on that support (a tilted proxy would topple, a lowered one fall),
                    # unless the clip starts with the object already in hand
                    if o["grasp_interval"][0] > 0:
                        poses = settle_before_grasp(poses, t0, o["radius"], o["length"], self.support_z[i, h],
                                                    self._args.objects.grasp_blend_frames)
                    # the surface the object was put down on (table / basket), under its release pose
                    t1 = int(o["grasp_interval"][1])
                    if t1 < L:
                        rel = poses[t1 - 1]
                        obj_low = rel[2, 3] - cylinder_half_height(rel[:3, :3], o["radius"], o["length"])
                        near = slice(max(0, t1 - 3), min(L, t1 + 15))
                        low = float(d["ref_keypoints"][near, :, 2].min()) - 0.003
                        self.place_z[i, h] = max(min(obj_low, low), obj_low - drop)
                        poses = settle_after_release(poses, t1, o["radius"], o["length"], self.place_z[i, h],
                                                     self._args.objects.grasp_blend_frames)
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
        self.min_start = self._clear_starts(keys)
        self.min_start_t = torch.as_tensor(self.min_start, device=self.device)
        # glove frames past the clip end must not count as contact
        for k in ("pressure", "contact"):
            for i, L in enumerate(self.lengths):
                self.ref[k][i, L:] = 0.0
        print(f"Loaded {self.n_clips} clips ({self.lengths.sum()} frames at {self.hz:g} Hz) from {clip_dir}; "
              "objects: " + ", ".join(f"{int(self.obj_active[:, h].sum())} {s}" for h, s in enumerate(self.sides)))

    def _active_sides(self, clips: list[dict]) -> tuple[str, ...]:
        """The hands simulated (`active_hands`): "auto" keeps those with an object in some clip (episode clips:
        an object on the hand's schedule; proxy clips: a hand object)."""
        mode = self._args.active_hands
        if mode in SIDES:
            return (mode,)
        if mode == "both":
            return SIDES
        used = lambda c, s: (np.asarray(c["hands"][s]["object_schedule"]) >= 0).any() if "scene_objects" in c \
            else c["hands"][s]["object"] is not None
        sides = tuple(s for s in SIDES if any(used(c, s) for c in clips))
        if not sides:
            raise ValueError("active_hands=auto: no hand manipulates an object in the selected clips")
        return sides

    def _load_episode(self, c: dict, keys: dict[str, list], clip_dir: Path) -> None:
        """A whole demonstration with its reconstructed objects (module docstring)."""
        if self.n_clips != 1:
            raise ValueError("episode clips are trained one at a time (data.clips: one episode clip)")
        L, dt = int(self.lengths[0]), self.dt
        # simulated on their solid collision mesh (mesh_sdf.collision_mesh: a RecGen bottle is a hollow shell)
        self.scene_objects = [{**o, "mesh": str(collision_mesh(o["mesh"]))} for o in c["scene_objects"]]
        N = len(self.scene_objects)
        all_pos = np.stack([o["poses"][:, :3, 3] for o in self.scene_objects], 1).astype(np.float64)   # (T, N, 3)
        all_rot = np.stack([o["poses"][:, :3, :3] for o in self.scene_objects], 1).astype(np.float64)
        sched = np.stack([c["hands"][s]["object_schedule"] for s in self.sides], 1)                       # (T, H)
        per = {k: [] for k in keys}
        held = np.zeros((L, self.H), bool)                          # the holds of build_episode_clip.py (glove contact)
        for k, o in enumerate(self.scene_objects):
            for s, a, b in o.get("holds", []):
                if s not in self.sides:
                    continue
                h = self.sides.index(s)
                held[a:b, h] |= sched[a:b, h] == k
        self.obj_spec = np.tile([DEFAULT_CYLINDER[k] for k in ("radius", "length", "mass")], (1, self.H, 1)).astype(float)
        self.obj_active = (sched >= 0).any(0)[None]
        self.support_z, self.place_z = np.zeros((1, self.H)), np.full((1, self.H), -1.0)
        self.grasp = np.zeros((1, self.H, 2), int)
        self.pose_in_palm = np.tile(np.eye(4), (1, self.H, 1, 1))
        for h, s in enumerate(self.sides):
            d = c["hands"][s]
            R = _quats_to_rotmats(d["wrist_quat"])
            pos = d["wrist_positions"].astype(np.float64)
            per["wrist_pos"].append(pos), per["wrist_rot"].append(R), per["wrist_quat"].append(d["wrist_quat"])
            per["wrist_vel"].append(np_velocity(pos, dt)), per["wrist_ang_vel"].append(np_angular_velocity(R, dt))
            per["base_q"].append(base_coordinates(self.base_R0[s], pos, R))
            per["dof_pos"].append(d["dof_positions"]), per["dof_vel"].append(np_velocity(d["dof_positions"], dt))
            per["tip_pos"].append(d["fingertip_pos"]), per["tip_quat"].append(np_quat_continuity(d["fingertip_quat"]))
            per["keypoints"].append(d["ref_keypoints"])
            per["pressure"].append(d["tactile_pressure"] / self.glove_scale)
            per["contact"].append(d["tactile_contact"].astype(np.float32))
            idx = sched[:, h]
            i0, valid = idx.clip(min=0), idx >= 0
            p, Ro = all_pos[np.arange(L), i0], all_rot[np.arange(L), i0]
            per["obj_pos"].append(p), per["obj_rot"].append(Ro)
            per["obj_quat"].append(np_quat_continuity(_rotmats_to_quats(Ro)))
            per["obj_vel"].append(np_velocity(p, dt)), per["obj_ang_vel"].append(np_angular_velocity(Ro, dt))
            per["obj_valid"].append(valid), per["holding"].append(held[:, h])
            if valid.any():
                on = np.flatnonzero(valid)
                self.grasp[0, h] = (on[0], on[-1] + 1)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.ref = {k: f(np.stack(v, 1))[None] for k, v in per.items()}                               # (1, T, H, ...)
        self.ref["obj_valid"] = self.ref["obj_valid"] > 0.5
        self.ref["holding"] = self.ref["holding"] > 0.5
        self.ref["head_pos"] = f(c["head"]["pos"])[None]
        self.ref["all_obj_pos"] = f(all_pos)[None]                                                     # (1, T, N, 3)
        self.ref["all_obj_quat"] = f(np.stack([np_quat_continuity(_rotmats_to_quats(all_rot[:, k])) for k in range(N)], 1))[None]
        self.ref["schedule"] = torch.as_tensor(sched, device=self.device)[None]                       # (1, T, H)
        manip = (sched >= 0).any(1)
        self.start_weight = f(1.0 + (self._args.episode_manip_weight - 1.0) * manip)
        self.lengths_t = torch.as_tensor(self.lengths, device=self.device)
        self.min_start = np.zeros(1, int)
        self.min_start_t = torch.as_tensor(self.min_start, device=self.device)
        self.tactile_available = bool(c["metadata"].get("tactile_available", True))
        print(f"Loaded episode {c['metadata']['name']} ({L} frames at {self.hz:g} Hz, {L / self.hz:.1f} s) from {clip_dir}: "
              f"{N} objects ({sum(o['fixed'] for o in self.scene_objects)} fixed), manipulation on {manip.mean():.0%} of "
              f"the frames" + ("" if self.tactile_available else ", no tactile reference"))

    def _episode_start(self, envs_idx: torch.Tensor) -> torch.Tensor:
        """Episode mode: frame 0 (evaluation, train_start=first), else anywhere (manipulations weighted)."""
        n = len(envs_idx)
        if self._eval_mode or self._args.train_start == "first":
            return torch.zeros(n, dtype=torch.long, device=self.device)
        w = self.start_weight[: int(self.lengths[0]) - 2]
        return torch.multinomial(w, n, replacement=True, generator=self.generator)

    def _episode_horizon(self, envs_idx: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        end = self.lengths_t[self.env_clip[envs_idx]] - 1
        if self._eval_mode or self._args.train_start == "first":
            return end
        return torch.minimum(start + int(round(self._args.episode_window_s * self.hz)), end)

    def _clear_starts(self, keys: dict[str, list]) -> np.ndarray:
        """First frame of each clip where no reference hand overlaps an object resting before its grasp
        (keypoints at least `start_clearance` from the cylinder): episodes start there at the earliest,
        an object placed inside a hand would be ejected at the reset. If the hand never clears the
        object before the grasp, the episode starts once the object is in the palm."""
        clear = self._args.objects.start_clearance
        out = np.zeros(self.n_clips, int)
        for i in range(self.n_clips):
            kp = keys["keypoints"][i].reshape(self.T_max, -1, 3)                     # both hands
            for h in range(self.H):
                t0 = int(self.grasp[i, h, 0])
                if not self.obj_active[i, h] or t0 == 0:
                    continue
                p, R = keys["obj_pos"][i][:t0, h], keys["obj_rot"][i][:t0, h]
                local = np.einsum("tkj,tji->tki", kp[:t0] - p[:, None], R)
                r, hl = self.obj_spec[i, h, 0], self.obj_spec[i, h, 1] / 2
                d = np.stack([np.linalg.norm(local[..., :2], axis=-1) - r, np.abs(local[..., 2]) - hl], -1)
                sdf = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)
                ok = np.nonzero(sdf.min(-1) > clear)[0]
                first = int(ok[0]) if len(ok) else t0 + self._args.objects.grasp_blend_frames
                out[i] = max(out[i], first)
            out[i] = min(out[i], self.lengths[i] - 10)
        late = [(self.clip_names[i], int(out[i])) for i in np.nonzero(out)[0]]
        if late:
            print(f"{len(late)} clips start after frame 0 (a hand overlaps a resting object before): {late}")
        return out

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
                                                  max_collision_pairs=200,
                                                  batch_links_info=True),   # per-env object masses
            vis_options=gs.options.VisOptions(show_world_frame=False, plane_reflection=False,
                                              **(dict(shadow=False, ambient_light=(0.45, 0.45, 0.45),
                                                      background_color=(0.85, 0.88, 0.92))
                                                 if self._render is not None else {})),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        sx, sy, sz = a.objects.support_size
        self.supports = [self.scene.add_entity(gs.morphs.Box(size=(sx, sy, sz), pos=(0.0, 3.0 + h, -1.0), fixed=True),
                                               surface=gs.surfaces.Default(color=(0.55, 0.45, 0.35)))
                         for h in range(self.H)]
        self.place_supports = [self.scene.add_entity(gs.morphs.Box(size=(sx, sy, sz), pos=(0.0, 5.0 + h, -1.0), fixed=True),
                                                     surface=gs.surfaces.Default(color=(0.45, 0.50, 0.60)))
                               for h in range(self.H)]
        self.hands = [FloatingWujiHand(self.scene, s, self.base_R0[s], a.hands, a.work_dir, self.device) for s in self.sides]
        self.objects = []
        if self.episode:                          # the reconstructed objects, at their first reference pose
            from scipy.spatial.transform import Rotation

            self.scene_entities = []
            for k, o in enumerate(self.scene_objects):
                P = o["poses"][0]
                q = Rotation.from_matrix(P[:3, :3]).as_quat()[[3, 0, 1, 2]]
                self.scene_entities.append(self.scene.add_entity(
                    gs.morphs.Mesh(file=o["mesh"], fixed=bool(o["fixed"]), pos=tuple(map(float, P[:3, 3])),
                                   quat=tuple(map(float, q))),
                    material=gs.materials.Rigid(rho=500.0, friction=a.objects.friction),
                    surface=gs.surfaces.Default(color=(0.75, 0.45, 0.85) if k % 2 == 0 else (0.45, 0.75, 0.55))))
        else:
            for h in range(self.H):
                morphs = [gs.morphs.Cylinder(radius=float(r), height=float(l), pos=(2.0, 0.3 * h, float(l) / 2))
                          for r, l, _ in self.obj_spec[:, h]]
                self.objects.append(self.scene.add_entity(
                    morph=morphs if len(morphs) > 1 else morphs[0],
                    material=gs.materials.Rigid(rho=500.0, friction=a.objects.friction),
                    surface=gs.surfaces.Default(color=(0.75, 0.45, 0.85) if h == 0 else (0.45, 0.75, 0.55)),
                ))
        if self._render is not None:
            self._add_render_entities()
        # visualize_policy.py: per environment an egocentric camera (at the recorded head) and a front camera
        self.cameras = []
        if self._camera_res is not None:
            self.cameras = [tuple(self.scene.add_camera(res=self._camera_res, fov=fov, env_idx=i, GUI=False,
                                                        near=0.02, debug=True)   # debug: draws markers
                                  for fov in (70, 45)) for i in range(self._num_envs)]
        # videos: envs far apart, so that a camera only sees its own env's hands and objects
        spacing = 5.0 if self.cameras else 3.0 if self._render is not None else 1.0
        self.scene.build(n_envs=self._num_envs, env_spacing=(spacing, spacing))
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
        if self.episode:                          # the movable objects' masses (the fixed ones do not need one)
            import trimesh

            N = len(self.scene_objects)
            self.scene_obj_mass = torch.zeros(B, N, device=dev)
            self.scene_obj_inertia = torch.zeros(N, 3, 3, device=dev)        # per kg, object frame (convex hull)
            self.scene_obj_link = [e.links[0].idx for e in self.scene_entities]
            self.scene_obj_free = torch.as_tensor([not o["fixed"] for o in self.scene_objects], device=dev)
            for k, (e, o) in enumerate(zip(self.scene_entities, self.scene_objects)):
                if not o["fixed"]:
                    m = torch.full((B, 1), float(o["mass"]), device=dev)
                    if not self._eval_mode and a.objects.mass_randomization_range > 0:
                        m *= 1 + a.objects.mass_randomization_range * (2 * torch.rand(m.shape, device=dev, generator=self.generator) - 1)
                    e.set_links_inertial_mass(m, links_idx_local=[0], envs_idx=envs)
                    self.scene_obj_mass[:, k] = m[:, 0]
                    hull = trimesh.load(o["mesh"], force="mesh").convex_hull
                    self.scene_obj_inertia[k] = torch.as_tensor(hull.moment_inertia / max(hull.volume, 1e-9),
                                                                dtype=torch.float32, device=dev)
        # support plates under each object's rest pose
        self.obj_active_t = torch.as_tensor(self.obj_active, device=dev)[self.env_clip]            # (B, H)
        self.support_z_t = torch.as_tensor(self.support_z, dtype=torch.float32, device=dev)[self.env_clip]
        rest = self.ref["obj_pos"][self.env_clip, 0]                                                # (B, H, 3)
        for h, sup in enumerate(self.supports):
            pos = torch.stack([rest[:, h, 0], rest[:, h, 1], self.support_z_t[:, h] - sz / 2], -1)
            parked = torch.tensor([0.0, 3.0 + h, -1.0], device=dev)
            sup.set_pos(torch.where(self.obj_active_t[:, h:h + 1], pos, parked), envs_idx=envs)
        # placing plates under each object's settled pose after the release (where the human put it down)
        self.place_z_t = torch.as_tensor(self.place_z, dtype=torch.float32, device=dev)[self.env_clip]
        end = self.ref["obj_pos"][self.env_clip, self.lengths_t[self.env_clip] - 1]                 # (B, H, 3)
        for h, sup in enumerate(self.place_supports):
            pos = torch.stack([end[:, h, 0], end[:, h, 1], self.place_z_t[:, h] - sz / 2], -1)
            parked = torch.tensor([0.0, 5.0 + h, -1.0], device=dev)
            sup.set_pos(torch.where(self.place_z_t[:, h:h + 1] > 0, pos, parked), envs_idx=envs)

    # ================================================================== rendering (videos only)
    def _add_render_entities(self) -> None:
        """One camera per env, and per hand a non-colliding translucent copy of the hand that is put on the
        reference pose before each render (the 'ghost'). Only for small eval envs (videos)."""
        res = tuple(self._render.get("res", (480, 360)))
        fov = float(self._render.get("fov", 45))
        # a single camera moved over every env (envs are 3 m apart, see _build_scene): per-env cameras
        # (env_idx) rendered some envs black in Genesis 1.3.3
        self.camera = self.scene.add_camera(res=res, pos=(0.5, -0.5, 0.5), lookat=(0.0, 0.0, 0.0), fov=fov,
                                            env_idx=0, GUI=False, near=0.02)
        self.ghosts = []
        for h, s in enumerate(self.sides):
            hand = self.hands[h]
            urdf = hand.model.write_floating_urdf(str(Path(self._args.work_dir) / f"wuji_{s}_ghost.urdf"),
                                                  self.base_R0[s], base_mass=0.01)
            self.ghosts.append(self.scene.add_entity(
                gs.morphs.URDF(file=urdf, fixed=True, merge_fixed_links=True, collision=False),
                material=gs.materials.Rigid(gravity_compensation=1.0),
                surface=gs.surfaces.Default(color=(1.0, 0.45, 0.0, 0.45)),
            ))
        self.object_ghosts = []                   # episode mode: the reference objects, translucent
        if self.episode:
            for o in self.scene_objects:
                if not o["fixed"]:
                    self.object_ghosts.append(self.scene.add_entity(
                        gs.morphs.Mesh(file=o["mesh"], fixed=False, collision=False),
                        material=gs.materials.Rigid(gravity_compensation=1.0),
                        surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2, 0.35))))
                else:
                    self.object_ghosts.append(None)

    @torch.no_grad()
    def set_reference_ghosts(self) -> None:
        """Put the ghost hands (and, in episode mode, the ghost objects) on the reference at frame t."""
        from tv2s_env.robots.wuji_model import BASE_JOINTS
        envs = torch.arange(self._num_envs, device=self.device)
        t = torch.minimum(self.t, self.lengths_t[self.env_clip] - 1)
        if not hasattr(self, "_ghost_dofs"):
            self._ghost_dofs = [[g.get_joint(n).dofs_idx_local[0] for n in BASE_JOINTS + self.hands[i].model.dof_names]
                                for i, g in enumerate(self.ghosts)]
        for h, ghost in enumerate(self.ghosts):
            q = torch.cat([self.ref["base_q"][self.env_clip, t, h], self.ref["dof_pos"][self.env_clip, t, h]], -1)
            ghost.set_dofs_position(q, self._ghost_dofs[h], envs_idx=envs, zero_velocity=True)
        for k, g in enumerate(getattr(self, "object_ghosts", [])):
            if g is not None:
                g.set_pos(self.ref["all_obj_pos"][self.env_clip, t, k], envs_idx=envs, zero_velocity=True)
                g.set_quat(self.ref["all_obj_quat"][self.env_clip, t, k], envs_idx=envs, zero_velocity=True)

    @torch.no_grad()
    def render_frames(self) -> np.ndarray:
        """(B, h, w, 3) uint8: every env's camera, looking at its hands, the ghosts on the reference."""
        self.set_reference_ghosts()
        palms = self.base_pos.cpu().numpy()                                            # (B, H, 3)
        centre = palms.mean(1)
        spread = np.linalg.norm(palms[:, 0] - palms[:, -1], axis=-1)                   # frame both hands (0: one hand)
        view = np.array([0.0, -0.85, 0.53])
        offsets = np.asarray(self.scene.envs_offset)
        offsets = offsets - offsets[0]                     # the camera's poses are relative to env 0
        frames = []
        for i in range(self._num_envs):
            c = centre[i] + offsets[i]
            self.camera.set_pose(pos=c + view * (0.45 + 1.1 * spread[i]), lookat=c)
            rgb = self.camera.render(rgb=True, force_render=True)[0]
            frames.append(np.asarray(rgb.cpu() if torch.is_tensor(rgb) else rgb))
        return np.stack(frames)

    def _build_sensors(self) -> None:
        t = self._args.tactile
        self.sensors = []
        for h, s in enumerate(self.sides):
            taxels = TaxelSet.load(str(Path(t.assets_dir) / f"{t.mapping}_{s}.npz"))
            if taxels.n_channels != self.layouts[h].n_channels and self.clip_taxels:   # another glove (EgoTouch)
                taxels = TaxelSet.load(str(Path(t.assets_dir) / f"{self.clip_taxels}_{s}.npz"))
            self.sensors.append(GloveSensor(taxels, self.layouts[h], self.hands[h].link_index, t.probe_radius, t.kn,
                                            self.device))
        if self.episode:
            self.mesh_distances = [MeshDistance(o["mesh"], self.device) for o in self.scene_objects]
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
        self.object_vel_change, self.object_ang_vel_change = z(self.H), z(self.H)
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
        if self.episode:                          # every object; each hand's is its scheduled one
            E = self.scene_entities
            self.all_object_pos = torch.stack([e.get_pos() for e in E], 1)                       # (B, N, 3)
            self.all_object_quat = torch.stack([e.get_quat() for e in E], 1)
            self.all_object_rot = quat_to_rotmat(self.all_object_quat)
            vel, ang = torch.stack([e.get_vel() for e in E], 1), torch.stack([e.get_ang() for e in E], 1)
            self.all_object_vel, self.all_object_ang = vel, ang
            idx = self.ref_at("schedule", self.t).clamp(min=0)                                    # (B, H)
            take = lambda x: torch.gather(x, 1, idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))
            self.object_pos, self.object_quat = take(self.all_object_pos), take(self.all_object_quat)
            self.object_lin_vel, self.object_ang_vel = take(vel), take(ang)
        else:
            self.object_pos = torch.stack([o.get_pos() for o in self.objects], 1)
            self.object_quat = torch.stack([o.get_quat() for o in self.objects], 1)
            self.object_lin_vel = torch.stack([o.get_vel() for o in self.objects], 1)
            self.object_ang_vel = torch.stack([o.get_ang() for o in self.objects], 1)
        self.object_rot = quat_to_rotmat(self.object_quat)
        self.object_axis = self.object_rot[..., 2]
        self.object_pos_rel = self.object_pos - self.base_pos
        self.object_height_above_support = self.object_pos[..., 2] - self.support_z_t
        self.support_contact_force = torch.stack(
            [s.get_links_net_contact_force().norm(dim=-1).sum(-1) for s in self.supports], 1)

        # simulated glove
        dist, pressure = [], []
        for h, sensor in enumerate(self.sensors):
            if self.episode:                      # the nearest of all the objects' surfaces
                p_w = sensor.taxel_world(st[h]["link_pos"], st[h]["link_rot"])                     # (B, P, 3)
                d = torch.stack([md(((p_w - self.all_object_pos[:, k, None])[..., None, :] @ self.all_object_rot[:, k, None])[..., 0, :])
                                 for k, md in enumerate(self.mesh_distances)]).amin(0)
                d, p = sensor.reduce(d)
            else:
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
        e_obj_end = ((self.object_pos - self.target_object_pos).norm(dim=-1) * self.obj_active_t).amax(-1)
        extra = {}
        if self.episode:                          # every free object against its reference (not each hand's)
            ref = self.ref["all_obj_pos"][self.env_clip, torch.minimum(self.t, self.lengths_t[self.env_clip] - 1)]
            e_all = (self.all_object_pos - ref).norm(dim=-1) * self.scene_obj_free
            e_obj_end = e_all.amax(-1)
            extra["e_obj_all"] = e_all.sum(-1) / self.scene_obj_free.sum().clamp(min=1)
        return dict(**extra,
            tac_tp=tp.sum(-1), tac_sim=n_sim.sum(-1), tac_real=n_gt.sum(-1),
            tac_map_similarity=hand_mean(map_sim, has),
            e_wrist=(self.base_pos - self.target_wrist_pos).norm(dim=-1).mean(-1),
            e_tip=(self.finger_link_pos - self.target_fingertip_pos).norm(dim=-1).mean((-1, -2)),
            e_kp=(self.keypoints - self.target_keypoints).norm(dim=-1).mean((-1, -2)),
            e_obj=hand_mean((self.object_pos - self.target_object_pos).norm(dim=-1), valid),
            # worst object against its full reference (in the palm, then settled on the placing support),
            # whatever the validity: judged at the end of an episode
            e_obj_end=e_obj_end,
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
        if self.episode:                          # every movable object on its reference pose at `start`
            for k, (e, o) in enumerate(zip(self.scene_entities, self.scene_objects)):
                if not o["fixed"]:
                    e.set_pos(self.ref["all_obj_pos"][clip, start, k], envs_idx=envs_idx, zero_velocity=True)
                    e.set_quat(self.ref["all_obj_quat"][clip, start, k], envs_idx=envs_idx, zero_velocity=True)
            return
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
            n = len(envs_idx) * self.H
            r = Rotation.from_rotvec(np.radians(o.init_rot_noise_deg) * (2 * np.random.rand(n, 3) - 1))
            qn = torch.as_tensor(r.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float32, device=self.device).reshape(-1, self.H, 4)
            from tv2s_env.common.utils.math_utils import quat_mul
            quat = torch.where((before & ~in_hand)[..., None], quat_mul(qn, quat), quat)
        for h, obj in enumerate(self.objects):
            obj.set_pos(pos[:, h], envs_idx=envs_idx, zero_velocity=True)
            obj.set_quat(quat[:, h], envs_idx=envs_idx, zero_velocity=True)

    # ================================================================== stepping
    def _physics_step(self, hand_targets: list[torch.Tensor]) -> None:
        prev_v, prev_w = self.object_lin_vel.clone(), self.object_ang_vel.clone()
        for hand, target in zip(self.hands, hand_targets):
            hand.control(target)
        self._apply_object_aids()
        self.scene.step()
        self.t += 1
        self.update_buffers()
        self.object_vel_change = (self.object_lin_vel - prev_v).norm(dim=-1)
        self.object_ang_vel_change = (self.object_ang_vel - prev_w).norm(dim=-1)

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
        if self.step_callback is not None:
            self.step_callback(terminated, truncated, infos)
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
