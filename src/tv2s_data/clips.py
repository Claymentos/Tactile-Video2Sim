"""Clip windows, grasps, and the HumanTouch simulation clips with a proxy object per grasping hand.

A clip is one manipulation cycle, padded by `pad_s`, either from the dataset's semantic annotations (a
`reach_grasp_*` segment plus the `transport_*` segment that follows it, or a `grasp_lift_basket` segment:
`segment_episode`) or from a Gemini scene parsing of the head-camera video (`scene_windows`). Stage 1
(tv2s_data/cuts.py) cuts the episodes with these windows; stage 2 (`sim_clip`) turns a HumanTouch cut into
a simulation clip. Both hands are kept in every clip; a hand "grasps" when its glove reports a sustained
contact (`grasp_interval`).

HumanTouch has no object pose or mesh, so the object a hand holds is inferred from the glove and the
retargeted Wuji reference (DexTaG and Benchmark-Genesis both need an object trajectory: DexTaG records it,
Benchmark-Genesis reconstructs it from video). The proxy is a cylinder:
  - at the firm-grasp frame, the contacting regions of the glove pick contact points on the reference
    Wuji hand (fingertips of contacting fingers, the palmar centre if the palm fires);
  - its axis runs across the palm (index knuckle -> little knuckle), its centre sits between the palm
    and the contacting fingertips (power grasp) or among the fingertips (pinch), its radius is the
    contact points' distance to the axis;
  - it is rigid in the palm while the glove holds it, and left where it was released after (its target
    is invalid shortly after the release). Before the grasp it rests where the grasp started; the
    environment makes that rest physical (`settle_before_grasp`: the nearest stable pose, on its
    support, blended into the grasp over a few frames), and likewise puts a placing support under the
    release pose (`settle_after_release`), where a successful episode leaves the object;
  - the surface it rests on (`support_z`) is put under it at the grasp onset. The two hands get separate
    supports: across the dataset the right hand's lowest fingertip is ~25 cm above the left's, which a
    single table cannot explain.
Pass a real object (mesh + poses) per clip by replacing `object` in the clip file (same keys).
"""

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial.transform import Rotation, Slerp

from tv2s_data.glove import GloveLayout
from tv2s_data.humantouch import SIDES

FINGERS_GLOVE = ("thumb", "index", "middle", "ring", "pinky")
FINGERS_WUJI = ("thumb", "index", "middle", "ring", "little")


@dataclass
class ClipConfig:
    pad_s: float = 0.5
    max_len_s: float = 120.0                # a cap, not a cut: the goal is the whole demonstration
    min_len_s: float = 1.5
    # tactile cleaning / contact
    baseline_window_s: float = 10.0
    contact_threshold: float = 8.0          # cleaned raw units (~2x the residual noise)
    grasp_min_channels: int = 15            # contacting channels that make a grasp
    grasp_min_regions: int = 2              # ... spread over at least this many regions
    grasp_min_s: float = 0.3
    grasp_gap_s: float = 0.2                # contact gaps bridged inside a grasp
    firm_window_s: float = 0.5              # firm frame: max contact within this after the onset
    release_margin_s: float = 0.2           # object target kept valid this long after the release
    # proxy cylinder
    radius_range: tuple[float, float] = (0.012, 0.045)
    length_range: tuple[float, float] = (0.06, 0.16)
    length_margin: float = 0.04
    mass: float = 0.1
    basket_mass: float = 0.3
    clip_types: tuple[str, ...] = ("reach_grasp", "grasp_lift_basket")
    follow_types: tuple[str, ...] = ("transport",)


def clip_config(overrides: list[str]) -> ClipConfig:
    """ClipConfig with command-line overrides KEY=VALUE (tuples as comma lists)."""
    cfg = ClipConfig()
    fields = {f.name for f in dataclasses.fields(cfg)}
    upd = {}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown ClipConfig field {key}: {sorted(fields)}")
        current = getattr(cfg, key)
        if isinstance(current, tuple):
            parts = [p for p in value.strip("()[]").split(",") if p]
            upd[key] = tuple(type(current[0])(p) if current else p for p in parts)
        else:
            upd[key] = type(current)(value)
    return dataclasses.replace(cfg, **upd)


@dataclass
class HandGeometry:
    """Reference Wuji hand at every frame of a clip (world frame)."""
    palm_pos: np.ndarray            # (T, 3)
    palm_rot: np.ndarray            # (T, 3, 3)
    tip_pos: np.ndarray             # (T, 5, 3)  thumb..little tip_link origins
    tip_quat: np.ndarray            # (T, 5, 4)  wxyz, link4 orientation (tip_link is fixed on it)
    knuckles: np.ndarray            # (T, 4, 3)  index..little link2 origins (MCP)
    keypoints: np.ndarray           # (T, 15, 3) PIP, DIP, tip per finger (Benchmark-Genesis' keypoints)
    palm_normal_local: np.ndarray   # (3,) palmar normal in the palm frame
    extras: dict = field(default_factory=dict)


def wuji_reference_geometry(hand, qpos: np.ndarray, palm_pos: np.ndarray, palm_quat: np.ndarray) -> HandGeometry:
    """Forward kinematics of the retargeted reference (tv2s_env.robots.wuji_model.WujiHand)."""
    T = len(qpos)
    base = np.tile(np.eye(4), (T, 1, 1))
    base[:, :3, :3] = Rotation.from_quat(palm_quat[:, [1, 2, 3, 0]]).as_matrix()
    base[:, :3, 3] = palm_pos
    fk = hand.fk(qpos, base)
    tip = np.stack([fk[hand.link_name(f, "tip_link")][:, :3, 3] for f in FINGERS_WUJI], 1)
    tip_rot = np.stack([fk[hand.link_name(f, "link4")][:, :3, :3] for f in FINGERS_WUJI], 1)
    tip_quat = Rotation.from_matrix(tip_rot.reshape(-1, 3, 3)).as_quat().reshape(T, 5, 4)[..., [3, 0, 1, 2]]
    knuckles = np.stack([fk[hand.link_name(f, "link2")][:, :3, 3] for f in FINGERS_WUJI[1:]], 1)
    kp = np.stack([fk[hand.link_name(f, p)][:, :3, 3] for f in FINGERS_WUJI for p in ("link3", "link4", "tip_link")], 1)
    # palmar normal from the flat hand: across the knuckles, signed towards where the fingers curl
    T0 = hand.fk(np.zeros(len(hand.dof_names)))
    o = T0[hand.link_name("palm", "")][:3, 3]
    i_mcp, l_mcp = T0[hand.link_name("index", "link1")][:3, 3], T0[hand.link_name("little", "link1")][:3, 3]
    n = np.cross(i_mcp - o, l_mcp - o)
    n /= np.linalg.norm(n)
    q_curl = np.zeros(len(hand.dof_names))
    for f in range(2, 6):
        q_curl[hand.dof_names.index(f"{hand.side}_finger{f}_joint1")] = 0.3
    tip_curl = hand.fk(q_curl)[hand.link_name("middle", "tip_link")][:3, 3]
    n *= np.sign(n @ (tip_curl - T0[hand.link_name("middle", "tip_link")][:3, 3]))
    R_palm0 = T0[hand.link_name("palm", "")][:3, :3]
    return HandGeometry(palm_pos=palm_pos.astype(np.float64), palm_rot=base[:, :3, :3], tip_pos=tip,
                        tip_quat=tip_quat, knuckles=knuckles, keypoints=kp, palm_normal_local=R_palm0.T @ n)


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def grasp_interval(contact: np.ndarray, layout: GloveLayout, hz: float, cfg: ClipConfig) -> tuple[int, int] | None:
    """Longest sustained grasp [t0, t1) of one hand in a clip, from its binary channel contacts (T, C)."""
    n = contact.sum(1)
    regions = np.stack([contact[:, layout.region == r].sum(1) >= 2 for r in ("palm",) + FINGERS_GLOVE], 1).sum(1)
    grasp = (n >= cfg.grasp_min_channels) & (regions >= cfg.grasp_min_regions)
    gap = max(1, int(round(cfg.grasp_gap_s * hz)))
    # pad so that closing does not erode the clip ends
    grasp = binary_closing(np.pad(grasp, gap), structure=np.ones(2 * gap + 1))[gap:-gap]
    grasp = binary_opening(grasp, structure=np.ones(3))
    runs = [(a, b) for a, b in _intervals(grasp) if (b - a) >= cfg.grasp_min_s * hz]
    if not runs:
        return None
    return max(runs, key=lambda ab: ab[1] - ab[0])


def proxy_cylinder(geom: HandGeometry, contact_t: np.ndarray, layout: GloveLayout, t: int,
                   cfg: ClipConfig) -> tuple[np.ndarray, float, float]:
    """Proxy cylinder at frame t from the contacting glove regions: world pose (4,4), radius, length.
    The cylinder axis is the pose's z axis (Genesis' Cylinder convention)."""
    R_palm, p_palm = geom.palm_rot[t], geom.palm_pos[t]
    normal = R_palm @ geom.palm_normal_local
    knuckles = geom.knuckles[t]
    axis = knuckles[0] - knuckles[3]                                  # index -> little knuckle
    axis -= (axis @ normal) * normal
    axis /= np.linalg.norm(axis)
    active = {r: int(contact_t[layout.region == r].sum()) for r in ("palm",) + FINGERS_GLOVE}
    tips = [geom.tip_pos[t, i] for i, f in enumerate(FINGERS_GLOVE) if active[f] >= 2]
    palm_centre = 0.5 * (p_palm + knuckles.mean(0)) + 0.01 * normal  # palmar side of the palm
    palm_on = active["palm"] >= 4
    if not tips:                                                       # palm-only contact (pushing, carrying)
        tips = [palm_centre + 0.04 * normal]
    tips = np.asarray(tips)
    if palm_on:
        centre = 0.5 * (palm_centre + tips.mean(0))
        points = np.vstack([tips, palm_centre[None]])
    else:
        centre = tips.mean(0)
        points = tips
    rel = points - centre
    radial = rel - (rel @ axis)[:, None] * axis
    radius = float(np.clip(np.median(np.linalg.norm(radial, axis=1)), *cfg.radius_range))
    along = rel @ axis
    length = float(np.clip(np.ptp(along) + cfg.length_margin, *cfg.length_range))
    x = normal - (normal @ axis) * axis
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, np.cross(axis, x), axis], 1)
    pose[:3, 3] = centre
    return pose, radius, length


def cylinder_half_height(rot: np.ndarray, radius: float, length: float) -> float:
    """Half extent along world z of a cylinder whose axis is rot[:, 2]."""
    az = abs(rot[2, 2])
    return radius * np.sqrt(max(0.0, 1.0 - az * az)) + 0.5 * length * az


def stable_rest_pose(pose: np.ndarray, radius: float, length: float, support_top: float) -> np.ndarray:
    """The resting pose nearest to `pose` of a cylinder on a horizontal support whose top is at
    `support_top`: lying on its side if its axis is closer to horizontal, standing on an end otherwise
    (smallest rotation that gets there, centre kept in xy)."""
    axis = pose[:3, 2]
    target = np.array([0.0, 0.0, np.sign(axis[2]) or 1.0])
    if abs(axis[2]) < np.sqrt(0.5):                      # lying: drop the axis' vertical component
        target = np.array([axis[0], axis[1], 0.0])
        target /= np.linalg.norm(target)
    rot = Rotation.align_vectors(target[None], axis[None])[0].as_matrix() @ pose[:3, :3]
    out = pose.copy()
    out[:3, :3] = rot
    out[2, 3] = support_top + cylinder_half_height(rot, radius, length)
    return out


def settle_before_grasp(poses: np.ndarray, t0: int, radius: float, length: float, support_top: float,
                        blend: int) -> np.ndarray:
    """Object trajectory with a physical start: at rest on the support (stable_rest_pose of the grasp
    onset pose) before t0, then blended into the in-palm trajectory over `blend` frames (the hand lifts
    it into the grasp)."""
    out = poses.copy()
    rest = stable_rest_pose(poses[t0], radius, length, support_top)
    out[:t0] = rest
    for j in range(min(blend, len(poses) - t0)):
        a = (j + 1) / (blend + 1)
        key = Rotation.from_matrix(np.stack([rest[:3, :3], poses[t0 + j, :3, :3]]))
        out[t0 + j, :3, :3] = Slerp([0.0, 1.0], key)(a).as_matrix()
        out[t0 + j, :3, 3] = (1 - a) * rest[:3, 3] + a * poses[t0 + j, :3, 3]
    return out


def settle_after_release(poses: np.ndarray, t1: int, radius: float, length: float, support_top: float,
                         blend: int) -> np.ndarray:
    """Object trajectory with a physical end: from the release pose (frame t1 - 1), blended over `blend`
    frames onto its nearest stable pose on the placing support, then at rest there."""
    out = poses.copy()
    release = poses[t1 - 1]
    rest = stable_rest_pose(release, radius, length, support_top)
    out[t1:] = rest
    for j in range(min(blend, len(poses) - t1)):
        a = (j + 1) / (blend + 1)
        key = Rotation.from_matrix(np.stack([release[:3, :3], rest[:3, :3]]))
        out[t1 + j, :3, :3] = Slerp([0.0, 1.0], key)(a).as_matrix()
        out[t1 + j, :3, 3] = (1 - a) * release[:3, 3] + a * rest[:3, 3]
    return out


def object_trajectory(geom: HandGeometry, t0: int, t1: int, t_firm: int, pose_firm: np.ndarray,
                      T: int, release_margin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid in the palm on [t0, t1), at rest before, left at the release pose after.
    Returns poses (T,4,4), valid (T,) and the in-palm pose (4,4)."""
    palm = np.tile(np.eye(4), (T, 1, 1))
    palm[:, :3, :3], palm[:, :3, 3] = geom.palm_rot, geom.palm_pos
    rel = np.linalg.inv(palm[t_firm]) @ pose_firm
    poses = palm @ rel
    poses[:t0] = poses[t0]
    poses[t1:] = poses[t1 - 1]
    valid = np.zeros(T, bool)
    valid[: min(T, t1 + release_margin)] = True
    return poses, valid, rel


def segment_episode(annotations: list[dict], n_frames: int, frame_index: np.ndarray, hz: float,
                    cfg: ClipConfig) -> list[dict]:
    """Clip windows (in resampled frames) from the 60 Hz semantic annotations."""
    to_res = lambda f60: int(np.searchsorted(frame_index, f60))           # first resampled frame >= f60
    clips = []
    for i, a in enumerate(annotations):
        if not any(a["type"].startswith(p) for p in cfg.clip_types):
            continue
        start, end, types = a["start_frame_60"], a["end_frame_60_exclusive"], [a["type"]]
        if i + 1 < len(annotations) and any(annotations[i + 1]["type"].startswith(p) for p in cfg.follow_types):
            end = annotations[i + 1]["end_frame_60_exclusive"]
            types.append(annotations[i + 1]["type"])
        s = max(0, to_res(start) - int(cfg.pad_s * hz))
        e = min(n_frames, to_res(end) + int(cfg.pad_s * hz))
        e = min(e, s + int(cfg.max_len_s * hz))
        if e - s >= cfg.min_len_s * hz:
            clips.append(dict(start=s, end=e, types=types, captions=[a.get("caption", "")]))
    return clips


def scene_windows(scene: dict, n_frames: int, frame_index: np.ndarray, hz: float, src_fps: float,
                  cfg: ClipConfig) -> list[dict]:
    """Clip windows (in resampled frames) from a Gemini scene file (scripts/reconstruction/semantic_parsing.py)."""
    to_res = lambda t_s: int(np.searchsorted(frame_index, t_s * src_fps))   # first resampled frame >= t_s
    clips = []
    for c in scene["clips"]:
        s = max(0, to_res(c["start_s"]) - int(cfg.pad_s * hz))
        e = min(n_frames, to_res(c["end_s"]) + int(cfg.pad_s * hz))
        e = min(e, s + int(cfg.max_len_s * hz))
        if e - s >= cfg.min_len_s * hz:
            clips.append(dict(start=s, end=e, types=[c[f"{side}_hand"]["action"] for side in SIDES],
                              captions=[c["caption"]], scene=c))
    return clips


def sim_clip(cut: dict, arrays: dict, retarget: dict, geoms: dict[str, HandGeometry], head_pose: np.ndarray,
             layouts: dict[str, GloveLayout], cfg: ClipConfig) -> dict | None:
    """Simulation clip of one HumanTouch cut (tv2s_data/cuts.py, 60 Hz video frames): the retargeted
    reference hands on the cut's window (retarget_humantouch.py rate, `retarget["frame_index"]` gives
    each frame's 60 Hz frame), the cut's glove at those frames, and per grasping hand the proxy object.
    `geoms`: wuji_reference_geometry of the whole episode per hand; `head_pose`: the episode's 60 Hz head
    pose. None if no hand grasps in the window (nothing to manipulate)."""
    m = cut["metadata"]
    hz, frame_index = float(retarget["hz"]), retarget["frame_index"]
    s, e = (int(np.searchsorted(frame_index, f)) for f in (m["start"], m["end"]))
    if e - s < 2:
        return None
    T, sl, rows = e - s, slice(s, e), frame_index[s:e] - m["start"]     # rows: the cut's arrays at these frames
    clip = dict(metadata=dict(episode=m["episode"], clip=m["clip"], name=m["name"], cut=m["name"],
                              start=s, end=e, source_frames=frame_index[sl].astype(np.int64), hz=hz,
                              types=m["types"], captions=m["captions"], scene=m["scene"],
                              tactile_mapping=m["tactile_mapping"], finger_tip_row=m["finger_tip_row"]),
                head=dict(pos=head_pose[frame_index[sl], :3], quat=head_pose[frame_index[sl]][:, [6, 3, 4, 5]]),
                hands={})
    for side in SIDES:
        g = geoms[side]
        sub = HandGeometry(palm_pos=g.palm_pos[sl], palm_rot=g.palm_rot[sl], tip_pos=g.tip_pos[sl],
                           tip_quat=g.tip_quat[sl], knuckles=g.knuckles[sl], keypoints=g.keypoints[sl],
                           palm_normal_local=g.palm_normal_local)
        pressure, contact = arrays[f"{side}_pressure"][rows], arrays[f"{side}_contact"][rows]
        sem = cut["hands"][side]["semantic"]
        hand = dict(
            wrist_positions=sub.palm_pos.astype(np.float32),
            wrist_quat=Rotation.from_matrix(sub.palm_rot).as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
            dof_positions=retarget[f"{side}_qpos"][sl].astype(np.float32),
            human_keypoints=retarget[f"{side}_keypoints"][sl].astype(np.float32),
            fingertip_pos=sub.tip_pos.astype(np.float32),
            fingertip_quat=sub.tip_quat.astype(np.float32),
            ref_keypoints=sub.keypoints.astype(np.float32),
            tactile_pressure=pressure,
            tactile_contact=contact,
            object=None,
            # scene cuts: what Gemini says this hand does (action, object, from, to) and the object's
            # attributes (description, material, shape, size); None for annotation cuts
            semantic=sem,
        )
        interval = grasp_interval(contact, layouts[side], hz, cfg)
        if interval is not None:
            t0, t1 = interval
            win = slice(t0, min(t1, t0 + max(1, int(cfg.firm_window_s * hz))))
            t_firm = t0 + int(np.argmax(contact[win].sum(1)))
            pose, radius, length = proxy_cylinder(sub, contact[t_firm], layouts[side], t_firm, cfg)
            poses, valid, rel = object_trajectory(sub, t0, t1, t_firm, pose, T, int(round(cfg.release_margin_s * hz)))
            support_z = float(poses[t0, 2, 3] - cylinder_half_height(poses[t0, :3, :3], radius, length))
            basket = any(t.startswith("grasp_lift_basket") for t in m["types"]) or bool(sem and sem["object_is_bin"])
            hand["object"] = dict(
                type="cylinder", radius=radius, length=length,
                mass=cfg.basket_mass if basket else cfg.mass,
                pose_matrices=poses.astype(np.float32), valid=valid,
                grasp_interval=(int(t0), int(t1)), firm_frame=int(t_firm),
                pose_in_palm=rel.astype(np.float32), support_z=support_z,
            )
        clip["hands"][side] = hand
    return clip if any(h["object"] is not None for h in clip["hands"].values()) else None
