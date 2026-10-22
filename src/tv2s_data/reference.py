"""Per-episode RL reference (reference.npz): everything the rl_recon environment simulates and compares with.

The file is Benchmark-Genesis' (one hand, one object), written by scripts/rl/prepare_reference.py
(tv2s_data/recon_reference.py). `load_reference` reads it into the layout the environment (rl_recon_env.py) takes,
hands on axis 0 of the hand arrays, objects on axis 0 of the object arrays:

  fps, sides (H,)                         hands, left before right
  wrist_pos (H,T,3), wrist_rot (H,T,3,3)  retargeted Wuji palm pose
  qpos (H,T,20), base_R0 (H,3,3), base_q (H,T,6)   finger joints, virtual base (wuji_model.BASE_JOINTS)
  ref_keypoints (H,T,15,3)                Wuji FK of the reference: PIP, DIP, tip of each finger
  human_kp (H,T,21,3), human_kp_index (15,)   GT human joints, and those the Wuji keypoints correspond to
  real_contact (H,T,C) bool, real_raw (H,T,C)  the glove's channels: contacts, readings
  taxels (H,) paths                       the glove's taxels on the Wuji links (assets/tactile/wuji_taxels_<side>.npz)
  object_ids (N,), object_collision (N,) paths, object_fixed (N,), object_mass (N,) kg (NaN: the env's)
  obj_pos (N,T,3), obj_rot (N,T,3,3), obj_valid (N,T)   object trajectories (reference targets)
  obj_rest_z (N,)                         lift putting the frame-0 object on the table
  obj_weight (N,T), obj_margin_scale (N,T)  per frame, the weight of the object terms and a scale of their position
                                          margins (keyframe_object_trajectory's measured target; 1 when absent)
  cam_pos (T,3)                           camera centre (the object error is split along / across its ray)
  holds (M,4) int [hand, object, start, end), hold_R_palm (M,3,3), hold_p_palm (M,3)
                                          when each hand holds which object, at which pose in its palm (the
                                          assist's anchor and the in-hand starts)
  rsi_frames (K,)                         candidate in-hand start frames (inside holds)
  mask_axis_2d (N,T,2), mask_axis_valid (N,T), cam_R (T,3,3), cam_t (T,3), cam_K (4,), C (3,3), shift (3,)
                                          optional: the object's long axis in each frame from its masks
                                          (Benchmark-Genesis' mask_axes.py) and the world -> camera transform
  start_clearance                         starts need the reference taxels this far from resting objects (m; 0)

Benchmark-Genesis' holds are its assist's: the glove reports the grasp from `HOLD_ONSET` channels on and keeps it
while one of the next `HOLD_WINDOW` frames does, anchored at its tactile grasp pose (grasp_R_palm, grasp_p_palm).
"""

from pathlib import Path

import numpy as np

HOLD_ONSET = 5                 # rl_recon EnvConfig.assist_onset
HOLD_WINDOW = 5                # rl_recon ReconEnv.gt_holding: a grasp frame within the next 5


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
    """reference.npz -> the layout of the module docstring."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    T = len(d["qpos"])
    taxels = Path(str(d["taxels"]))
    if not taxels.exists():                        # written by Benchmark-Genesis' checkout: the same file, here
        taxels = Path(__file__).resolve().parents[2] / "assets" / "tactile" / taxels.name
    H = lambda k: d[k][None]
    out = dict(fps=float(d["fps"]), sides=[str(d["side"])], wrist_pos=H("wrist_pos"), wrist_rot=H("wrist_rot"),
               qpos=H("qpos"), base_R0=H("base_R0"), base_q=H("base_q"), ref_keypoints=H("ref_keypoints"),
               human_kp=H("human_kp"), human_kp_index=d["human_kp_index"], real_contact=H("real_contact"),
               real_raw=H("real_raw").astype(np.float32), taxels=np.array([str(taxels)]),
               object_ids=np.array(["object"]), object_collision=np.array([str(d["object_collision"])]),
               object_fixed=np.array([False]), object_mass=np.array([np.nan]),
               obj_pos=d["obj_pos"][None], obj_rot=d["obj_rot"][None], obj_valid=d["obj_valid"][None],
               obj_weight=d.get("obj_weight", np.ones(T))[None].astype(np.float32),
               obj_margin_scale=d.get("obj_margin_scale", np.ones(T))[None].astype(np.float32),
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
