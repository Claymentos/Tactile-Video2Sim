"""Per-episode RL reference (reference.npz): everything the rl_recon environment simulates and compares with.

The format is Benchmark-Genesis' (reconstruction/rl_recon/prepare_reference.py), extended to several hands and
objects. Two writers produce it (scripts/rl/prepare_reference.py):
  - Benchmark-Genesis' own chain on its reconstruction of a video (one hand, one object), unchanged;
  - `clip_reference` here, from an episode clip of this pipeline (scripts/data/build_episode_clip.py): the hands
    the demonstration uses (the scene's `hands_used`, semantic_parsing.py), every reconstructed object.
`load_reference` reads both into one layout, hands on axis 0 of the hand arrays, objects on axis 0 of the
object arrays:

  fps, sides (H,)                         hands, left before right
  wrist_pos (H,T,3), wrist_rot (H,T,3,3)  retargeted Wuji palm pose
  qpos (H,T,20), base_R0 (H,3,3), base_q (H,T,6)   finger joints, virtual base (wuji_model.BASE_JOINTS)
  ref_keypoints (H,T,15,3)                Wuji FK of the reference: PIP, DIP, tip of each finger
  human_kp (H,T,21,3), human_kp_index (15,)   GT human joints, and those the Wuji keypoints correspond to
  real_contact (H,T,C) bool, real_raw (H,T,C)  the glove's channels: contacts, readings
  taxels (H,) paths                       the glove's taxels on the Wuji links (TaxelSet, or Benchmark-Genesis')
  object_ids (N,), object_collision (N,) paths, object_fixed (N,), object_mass (N,) kg (NaN: the env's)
  obj_pos (N,T,3), obj_rot (N,T,3,3), obj_valid (N,T)   object trajectories (reference targets)
  obj_rest_z (N,)                         lift putting the frame-0 object on the table (NaN: from its mesh)
  cam_pos (T,3)                           camera centre (the object error is split along / across its ray)
  holds (M,4) int [hand, object, start, end), hold_R_palm (M,3,3), hold_p_palm (M,3)
                                          when each hand holds which object, at which pose in its palm (the
                                          assist's anchor and the in-hand starts)
  rsi_frames (K,)                         candidate in-hand start frames (inside holds)
  mask_axis_2d (N,T,2), mask_axis_valid (N,T), cam_R (T,3,3), cam_t (T,3), cam_K (4,), C (3,3), shift (3,)
                                          optional: the object's long axis in each frame from its masks
                                          (Benchmark-Genesis' mask_axes.py) and the world -> camera transform
  start_clearance                         starts need the reference taxels this far from resting objects (m)

Benchmark-Genesis' holds are its assist's: the glove reports the grasp from `HOLD_ONSET` channels on and keeps it
while one of the next `HOLD_WINDOW` frames does, anchored at its tactile grasp pose (grasp_R_palm, grasp_p_palm).
"""

from pathlib import Path

import numpy as np

HOLD_ONSET = 5                 # rl_recon EnvConfig.assist_onset
HOLD_WINDOW = 5                # rl_recon ReconEnv.gt_holding: a grasp frame within the next 5
HUMAN_KP = np.array([4 * f + k for f in range(5) for k in (2, 3, 4)])   # OpenPose joints of the Wuji keypoints


def contact_holds(real_contact: np.ndarray) -> list[tuple[int, int]]:
    """rl_recon's assist intervals for one hand (T, C): from the first frame with >= HOLD_ONSET channels in contact
    to the first frame without one within the next HOLD_WINDOW."""
    grasp = real_contact.sum(1) >= HOLD_ONSET
    T = len(grasp)
    holding = np.array([grasp[t:t + HOLD_WINDOW].any() for t in range(T)])
    out, t = [], 0
    while t < T:
        if grasp[t]:
            end = t
            while end < T and holding[end]:
                end += 1
            out.append((t, end))
            t = end
        t += 1
    return out


def load_reference(path: str | Path) -> dict:
    """reference.npz (either writer) -> the layout of the module docstring."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if "sides" in d:                                   # clip_reference: already in the layout
        d["sides"] = [str(s) for s in d["sides"]]
        return d
    # Benchmark-Genesis: one hand, one object
    T = len(d["qpos"])
    H = lambda k: d[k][None]
    out = dict(fps=float(d["fps"]), sides=[str(d["side"])], wrist_pos=H("wrist_pos"), wrist_rot=H("wrist_rot"),
               qpos=H("qpos"), base_R0=H("base_R0"), base_q=H("base_q"), ref_keypoints=H("ref_keypoints"),
               human_kp=H("human_kp"), human_kp_index=d["human_kp_index"], real_contact=H("real_contact"),
               real_raw=H("real_raw").astype(np.float32), taxels=np.array([str(d["taxels"])]),
               object_ids=np.array(["object"]), object_collision=np.array([str(d["object_collision"])]),
               object_fixed=np.array([False]), object_mass=np.array([np.nan]),
               obj_pos=d["obj_pos"][None], obj_rot=d["obj_rot"][None], obj_valid=d["obj_valid"][None],
               obj_rest_z=np.array([float(d["obj_rest_z"])]), cam_pos=d["cam_pos"], start_clearance=0.0)
    holds, R, p = [], [], []
    if "grasp_R_palm" in d:
        for a, b in contact_holds(d["real_contact"]):
            holds.append((0, 0, a, b)), R.append(d["grasp_R_palm"]), p.append(d["grasp_p_palm"])
        out["rsi_frames"] = d["grasp_frames"]
    else:
        out["rsi_frames"] = np.zeros(0, int)
    out["holds"] = np.array(holds, int).reshape(-1, 4)
    out["hold_R_palm"] = np.array(R, float).reshape(-1, 3, 3)
    out["hold_p_palm"] = np.array(p, float).reshape(-1, 3)
    if "mask_axis_2d" in d:
        out.update(mask_axis_2d=d["mask_axis_2d"][None], mask_axis_valid=d["mask_axis_valid"][None],
                   **{k: d[k] for k in ("cam_R", "cam_t", "cam_K", "C", "shift")})
    assert all(len(out[k][0]) == T for k in ("obj_pos", "real_contact"))
    return out


def clip_reference(clip: dict, sides: list[str], start_clearance: float = 0.005, collision_faces: int = 4000,
                   hand_table_clearance: float = 0.008) -> dict:
    """reference.npz contents from an episode clip (build_episode_clip.py), for the hands `sides`. As Benchmark-Genesis'
    prepare_reference.py (--clearance-mode global), each hand is lifted so that its lowest GT joint over the clip clears
    the table by `hand_table_clearance` (< 0: off): hand placement and table plane come from different depth
    estimates, and a hand under the table cannot reach an object lying on it."""
    from scipy.spatial.transform import Rotation

    from tv2s_env.robots.wuji_model import base_coordinates, mean_rotation
    from tv2s_env.tactile.mesh_sdf import collision_mesh

    meta = clip["metadata"]
    T = int(meta["end"] - meta["start"])
    rot = lambda q: Rotation.from_quat(np.asarray(q, float)[:, [1, 2, 3, 0]]).as_matrix()
    hands = {k: [] for k in ("wrist_pos", "wrist_rot", "qpos", "base_R0", "base_q", "ref_keypoints", "human_kp",
                             "real_contact", "real_raw", "taxels")}
    for s in sides:
        h = clip["hands"][s]
        R, pos = rot(h["wrist_quat"])[:T], np.asarray(h["wrist_positions"], float)[:T]
        human, ref_kp = np.asarray(h["human_keypoints"], float)[:T], np.asarray(h["ref_keypoints"], float)[:T]
        dz = max(0.0, hand_table_clearance - float(np.nanmin(human[..., 2]))) if hand_table_clearance >= 0 else 0.0
        if dz > 0:
            print(f"  {s} hand lifted {dz * 100:.1f} cm to clear the table")
        pos, human, ref_kp = pos + [0, 0, dz], human + [0, 0, dz], ref_kp + [0, 0, dz]
        R0 = mean_rotation(R)
        hands["wrist_pos"].append(pos), hands["wrist_rot"].append(R), hands["base_R0"].append(R0)
        hands["base_q"].append(base_coordinates(R0, pos, R)), hands["qpos"].append(np.asarray(h["dof_positions"])[:T])
        hands["ref_keypoints"].append(ref_kp), hands["human_kp"].append(human)
        hands["real_contact"].append(np.asarray(h["tactile_contact"], bool)[:T])
        hands["real_raw"].append(np.asarray(h["tactile_pressure"], np.float32)[:T])
        hands["taxels"].append(str(Path(__file__).resolve().parents[2] / "assets" / "tactile" / f"{meta['taxels']}_{s}.npz"))
    objs = clip["scene_objects"]
    poses = np.stack([np.asarray(o["poses"], float)[:T] for o in objs])                      # (N, T, 4, 4)
    holds, hR, hp, rsi = [], [], [], []
    for k, o in enumerate(objs):
        for s, a, b in o.get("holds", []):
            if s not in sides or o["fixed"]:
                continue
            h = sides.index(s)
            m = (a + b) // 2
            P = np.eye(4)
            P[:3, :3], P[:3, 3] = hands["wrist_rot"][h][m], hands["wrist_pos"][h][m]
            A = np.linalg.inv(P) @ poses[k, m]
            holds.append((h, k, a, b)), hR.append(A[:3, :3]), hp.append(A[:3, 3])
            rsi += list(range(a, min(b, T - 1)))
    out = {k: np.stack(v) if k != "taxels" else np.array(v) for k, v in hands.items()}
    out.update(fps=float(meta["hz"]), sides=np.array(sides), human_kp_index=HUMAN_KP,
               object_ids=np.array([o["id"] for o in objs]),
               object_collision=np.array([str(collision_mesh(o["mesh"], collision_faces)) for o in objs]),
               object_fixed=np.array([bool(o["fixed"]) for o in objs]),
               object_mass=np.array([float(o["mass"]) for o in objs]),
               obj_pos=poses[..., :3, 3], obj_rot=poses[..., :3, :3], obj_valid=np.ones(poses.shape[:2], bool),
               obj_rest_z=np.full(len(objs), np.nan), cam_pos=np.asarray(clip["head"]["pos"], float)[:T],
               holds=np.array(holds, int).reshape(-1, 4), hold_R_palm=np.array(hR, float).reshape(-1, 3, 3),
               hold_p_palm=np.array(hp, float).reshape(-1, 3), rsi_frames=np.unique(np.array(rsi, int)),
               glove_layout=str(meta["tactile_mapping"]), start_clearance=float(start_clearance),
               clip=str(meta["name"]))
    return out
