#!/usr/bin/env python3
"""One simulation clip covering a whole episode, with the reconstructed objects (the RL reconstructs the full demo).

Gathers, in the simulation frame of the episode's scene (assemble_scene.py: the table at z = 0):
  - the reference hands at the retargeting rate (30 Hz): the dataset's mocap (--hands mocap, HumanTouch /
    EgoTouch: <data>/retarget/<key>.npz) or HaWoR's (--hands hawor: perception/<key>/hawor/retarget.npz), carried by
    T_sim_world (after camera.npz's hand_correction: EgoTouch's second glove); the Wuji reference geometry (fingertips, keypoints) by forward kinematics;
  - the glove over the whole episode at those frames (tv2s_data/cuts.py sources: HumanTouch's cleaned
    pressure and contacts, EgoTouch's normalised taxels; metadata["glove_scale"] tells the environment its
    unit). TouchScale's glove has no taxel mapping yet: its tactile arrays are zeros and
    metadata["tactile_available"] is False (train without the tactile reward);
  - every reconstructed object: mesh, fixed or not, mass, and its pose at every frame (sim_trajectories.npz,
    interpolated from the sampled frames: positions linear, rotations slerped). With --objects anchored (the
    default), a manipulated object follows the hand while it is held, as Benchmark-Genesis'
    lock_object_to_hand.py does where the tracker cannot see the object (the hand hides it exactly then):
    a hold is a glove contact run of the hand that overlaps a clip giving it the object (the whole run: the
    clip edges are coarse);
    during a hold the pose is palm(t) * offset, the offset the median pose relative to the palm around the
    hold's start and end (where the tracker still sees the object); between holds the object rests at
    the median tracked pose of that rest, lifted to stand on the table, and short blends join the two
    (--rests tracked). By default (--rests released) it only moves while held, as in a physical simulation
    (anchor()): during each hold it sits at the pose in the palm that best explains the glove's contacts
    (tv2s_data/grasp_pose.py, Benchmark-Genesis' tactile_grasp_pose.py: the mocap hands are trusted, the
    video only pulls weakly), before the first hold where that hold grasps it, after each hold where the
    hand released it (dropped on the table, or on a container's bottom, where Gemini's clip moves it);
  - per hand an object schedule (T,): the index of the object the hand manipulates (the scene's clips: the
    object Gemini names for that hand over the clip's time), -1 elsewhere. The environment shows each hand
    its scheduled object as "its" object, so the per-hand reward terms apply unchanged.

Writes <out>/<name>.pkl and <out>/index.json (BimanualTactileEnv's clip folder; clip_dir=<out>).

    python scripts/data/build_episode_clip.py --perception data/humantouch/perception/episode_000002 --hands mocap --data data/humantouch --out data/episodes/ht_ep2
    python scripts/data/build_episode_clip.py --perception data/touchscale/perception/<id> --hands hawor --out data/episodes/ts_29029
    python scripts/data/build_episode_clip.py --perception data/egotouch/perception/<key> --hands mocap --data data/egotouch --out data/episodes/et_<name>
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.clips import ClipConfig, wuji_reference_geometry  # noqa: E402
from tv2s_data.humantouch import SIDES  # noqa: E402
from tv2s_data.scene import support_id  # noqa: E402
from tv2s_env.robots.wuji_model import WujiHand  # noqa: E402

CONTAINER_FLOOR = 0.006                # m: an object put into a container rests this high (its bottom)
HUMANTOUCH_MAPPING = Path.home() / "HumanTouch/X001/meta/humantouch/tactile_mapping.json"


def interp_poses(poses: np.ndarray, t_src: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    ok = np.isfinite(poses).all((1, 2))
    t, P = t_src[ok], poses[ok]
    q = np.clip(t_out, t[0], t[-1])
    out = np.tile(np.eye(4), (len(t_out), 1, 1))
    out[:, :3, :3] = Slerp(t, Rotation.from_matrix(P[:, :3, :3]))(q).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(q, t, P[:, k, 3])
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill the False runs shorter than `gap` between True runs."""
    out = mask.copy()
    runs = _runs(mask)
    for (a0, b0), (a1, _) in zip(runs, runs[1:]):
        if a1 - b0 <= gap:
            out[b0:a1] = True
    return out


def _settle(pose: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """The pose lifted so that the mesh does not go through the table (z = 0)."""
    out = pose.copy()
    out[..., 2, 3] -= np.minimum((verts @ np.swapaxes(pose[..., :3, :3], -1, -2) + pose[..., None, :3, 3])[..., 2].min(-1), 0.0)
    return out


def _median_pose(P: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(P[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(P[:, :3, 3], 0)
    return out


def _drop(pose: np.ndarray, verts: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """The pose moved down (or up) until the mesh stands at height `floor` (the table: 0)."""
    out = pose.copy()
    out[2, 3] -= (verts @ pose[:3, :3].T + pose[:3, 3])[:, 2].min() - floor
    return out


def rigid_chain(P: np.ndarray, holds: list, palms: dict, verts: np.ndarray, ref: int | None) -> np.ndarray:
    """The object moving only while held, rigidly with the palm, from one trusted video pose: the tracked pose at
    frame `ref` (RecGen's reference frame, where the mesh is exactly in the camera frame) carried forwards and
    backwards through the holds (a hold grasps the object where it rests at the onset, it rests where the hand
    released it). Without `ref`, the median tracked pose before the first hold starts the chain."""
    T = len(P)
    ok = np.isfinite(P).all((1, 2))
    out = np.repeat(np.eye(4)[None], T, 0)
    if ref is not None and ok[ref]:
        k = next((i for i, (s, a, b) in enumerate(holds) if a <= ref < b), None)
        if k is not None:                                     # trusted inside a hold: that hold's offset
            s, a, b = holds[k]
            out[a:b] = palms[s][a:b] @ (np.linalg.inv(palms[s][ref]) @ P[ref])
            after, before = _settle(out[b - 1], verts), _settle(out[a], verts)
            nxt, prv, lo, hi = k + 1, k - 1, b, a
        else:
            after = before = _settle(P[ref], verts)
            nxt = next((i for i, h in enumerate(holds) if h[1] > ref), len(holds))
            prv = nxt - 1
            lo, hi = (holds[prv][2] if prv >= 0 else 0), (holds[nxt][1] if nxt < len(holds) else T)
    else:
        first = holds[0][1] if holds else T
        seg = np.flatnonzero(ok[:first]) if ok[:first].any() else np.flatnonzero(ok)
        after = before = _settle(_median_pose(P[seg]), verts) if len(seg) else np.eye(4)
        nxt, prv, lo, hi = 0, -1, 0, (holds[0][1] if holds else T)
    for s, a, b in holds[nxt:]:                               # forwards
        out[lo:a] = after
        out[a:b] = palms[s][a:b] @ (np.linalg.inv(palms[s][a]) @ after)
        after, lo = _settle(out[b - 1], verts), b
    out[lo:] = after
    for s, a, b in reversed(holds[:prv + 1]):                 # backwards
        out[b:hi] = before
        out[a:b] = palms[s][a:b] @ (np.linalg.inv(palms[s][b - 1]) @ before)
        before, hi = _settle(out[a], verts), a
    out[:hi] = before
    return out


def place_container(pose: np.ndarray, verts: np.ndarray, points: np.ndarray, inset: float = 0.03) -> np.ndarray:
    """Smallest horizontal shift of a container (pose (4, 4), mesh vertices) that puts `points` (n, 3: where the
    hands took objects out of it and put them back) inside its footprint, inset by `inset` -- the hands (mocap)
    are trusted over the container's video position (monocular depth). Along the footprint's principal axes."""
    w = verts @ pose[:3, :3].T + pose[:3, 3]
    xy = w[:, :2] - w[:, :2].mean(0)
    _, _, axes = np.linalg.svd(xy[:: max(1, len(xy) // 5000)], full_matrices=False)
    fp, pts = xy @ axes.T, (points[:, :2] - w[:, :2].mean(0)) @ axes.T
    shift = np.zeros(2)
    for i in range(2):
        lo = pts[:, i].max() - (fp[:, i].max() - inset)       # shift >= lo keeps the farthest point inside
        hi = pts[:, i].min() - (fp[:, i].min() + inset)       # shift <= hi keeps the nearest one
        shift[i] = np.clip(0.0, lo, hi) if lo <= hi else 0.5 * (lo + hi)
    out = pose.copy()
    out[:2, 3] += shift @ axes
    return out


def anchor(P: np.ndarray, holds: list, palms: dict, verts: np.ndarray, blend: int, around: int, rests: str = "released",
           ref: int | None = None, offsets: list | None = None, drops: list | None = None):
    """Held object on the hand, still between holds (module docstring). P (T, 4, 4) tracked poses.
    rests="released": the object only moves while held, as in a physical simulation. Each hold carries it at one
    pose in the palm, `offsets` (one per hold: fit_grasp's, the pose that explains the glove) or else
    rigid_chain's (the video's); it rests where the hand released it (on the table when `drops[i][1]`), and
    before the first hold where that hold grasps it (on the table when `drops[0][0]`; a drop is a floor height,
    None: left where it is, lifted above the table if needed); between a release and
    the next grasp, if the two disagree, it slides over the last `blend` frames. No hold: rigid_chain.
    rests="tracked": the in-palm offset is the median tracked pose relative to the palm around the hold's
    edges and each rest the median tracked pose there, blended.
    Every pose is lifted above the table. Returns the poses and the median distance to the tracked ones."""
    T = len(P)
    ok = np.isfinite(P).all((1, 2))
    v = verts[:: max(1, len(verts) // 2000)]
    holds = sorted(holds, key=lambda h: h[1])
    if rests == "released":
        out = rigid_chain(P, holds, palms, v, ref)
        if holds:
            if offsets is None:
                offsets = [np.linalg.inv(palms[s][a]) @ out[a] for s, a, b in holds]
            drops = drops or [(None, None)] * len(holds)
            s, a, _ = holds[0]
            rest = palms[s][a] @ offsets[0]
            rest = _drop(rest, v, drops[0][0]) if drops[0][0] is not None else _settle(rest, v)
            out[:a] = rest
            for i, ((s, a, b), off) in enumerate(zip(holds, offsets)):
                if a > 0:                                     # slide from the last rest to this grasp
                    grasp = palms[s][a] @ off
                    lo = max(holds[i - 1][2] if i else 0, a - blend)
                    for t in range(lo, a):
                        w = (t - lo + 1) / (a - lo + 1)
                        out[t, :3, 3] = (1 - w) * out[lo, :3, 3] + w * grasp[:3, 3]
                        out[t, :3, :3] = Slerp([0, 1], Rotation.from_matrix([out[lo, :3, :3], grasp[:3, :3]]))(w).as_matrix()
                out[a:b] = palms[s][a:b] @ off
                rest = _drop(out[b - 1], v, drops[i][1]) if drops[i][1] is not None else _settle(out[b - 1], v)
                out[b:] = rest
    else:
        out = P.copy()
        held = np.zeros(T, bool)
        for s, a, b in holds:
            Tp = palms[s]
            win = np.r_[np.arange(max(0, a - around), min(T, a + around)), np.arange(max(0, b - around), min(T, b + around))]
            win = win[ok[win]]
            if len(win) == 0:
                continue
            out[a:b] = Tp[a:b] @ _median_pose(np.linalg.inv(Tp[win]) @ P[win])
            held[a:b] = True
        for a, b in _runs(~held):                             # rests: one still pose, standing on the table
            seg = np.arange(a, b)[ok[a:b]]
            if len(seg):
                out[a:b] = _settle(_median_pose(P[seg]), v)
        for t0 in np.flatnonzero(np.diff(held.astype(int)) != 0) + 1:   # blend across each transition
            lo, hi = max(0, t0 - blend), min(T - 1, t0 + blend)
            for t in range(lo + 1, hi):
                w = (t - lo) / (hi - lo)
                out[t, :3, 3] = (1 - w) * out[lo, :3, 3] + w * out[hi, :3, 3]
                out[t, :3, :3] = Slerp([0, 1], Rotation.from_matrix([out[lo, :3, :3], out[hi, :3, :3]]))(w).as_matrix()
    out = _settle(out, v)
    dev = np.linalg.norm(out[ok, :3, 3] - P[ok, :3, 3], axis=1)
    return out, float(np.median(dev)) if len(dev) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (sim_scene.json from assemble_scene.py)")
    ap.add_argument("--hands", choices=("mocap", "hawor"), required=True)
    ap.add_argument("--data", default=None, help="HumanTouch prepared folder (episodes/, retarget/) for --hands mocap")
    ap.add_argument("--root", default=str(Path.home() / "HumanTouch/X001"), help="HumanTouch folder (glove mapping)")
    ap.add_argument("--ts-root", default=str(Path.home() / "TouchScale"), help="TouchScale folder (glove contacts)")
    ap.add_argument("--et-root", default=str(Path.home() / "EgoTouch"), help="EgoTouch folder (the episode list)")
    ap.add_argument("--out", required=True, help="clip folder to write")
    ap.add_argument("--objects", choices=("anchored", "tracked"), default="anchored",
                    help="anchored: held objects follow the hand, rests are still (docstring); tracked: as tracked")
    ap.add_argument("--blend", type=int, default=4, help="frames blended at the hold / rest transitions (--rests tracked)")
    ap.add_argument("--no-grasp-fit", dest="grasp_fit", action="store_false",
                    help="--rests released: carry the object at the video's pose in the palm instead of the one that "
                         "explains the glove (tv2s_data/grasp_pose.py; HumanTouch and EgoTouch, TouchScale has no taxel mapping)")
    ap.add_argument("--taxels", default=None, help="tactile mapping of the Wuji taxels (assets/tactile/<name>_<side>.npz; "
                                                   "default mano, EgoTouch mano_egotouch)")
    ap.add_argument("--rests", choices=("released", "tracked"), default="released",
                    help="anchored objects: rest where the hand released them (default) or at their tracked pose (anchor())")
    ap.add_argument("--name", default=None, help="clip name (default: <prefix>_episode)")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    sc = json.loads((root / "sim_scene.json").read_text())
    scene = json.loads(Path(meta["scene"]).read_text())
    traj = np.load(root / "sim_trajectories.npz")
    T_sim_world = np.asarray(sc["T_sim_world"])
    cam_npz = np.load(root / "camera.npz")
    # each hand in the simulation frame: register_camera.py's rigid correction of its mocap (EgoTouch's second
    # glove; identity otherwise) then T_sim_world
    T_sim_hand = {s: T_sim_world @ (cam_npz["hand_correction"][k] if "hand_correction" in cam_npz.files else np.eye(4))
                  for k, s in enumerate(SIDES)}
    key, dataset = meta["key"], meta["dataset"]

    rpath = Path(args.data) / "retarget" / f"{key}.npz" if args.hands == "mocap" else root / "hawor" / "retarget.npz"
    ret = dict(np.load(rpath))
    fi = ret["frame_index"]                                   # video frames of the reference frames
    hz = float(ret["hz"])
    t_ref = fi / meta["video_fps"]
    T = len(fi)

    # glove over the whole episode at the reference frames
    tactile_available = dataset in ("humantouch", "egotouch")
    glove_scale, mapping = None, HUMANTOUCH_MAPPING
    args.taxels = args.taxels or ("mano_egotouch" if dataset == "egotouch" else "mano")
    if dataset == "egotouch":
        from tv2s_data.cuts import EGOTOUCH_GRASP_MIN_CHANNELS, egotouch_sources
        from tv2s_data.egotouch import LAYOUT_JSON

        src = next(egotouch_sources(Path(args.data), Path(args.et_root), key, ClipConfig()))
        glove = {s: (src.tactile[s]["pressure"][fi], src.contact[s][fi]) for s in SIDES}
        touching = {s: glove[s][1].sum(1) >= EGOTOUCH_GRASP_MIN_CHANNELS for s in SIDES}
        glove_scale, mapping = 1.0, LAYOUT_JSON                  # normalised pressure: observed as it is
    elif dataset == "humantouch":
        from tv2s_data.cuts import humantouch_sources

        cfg = ClipConfig()
        src = next(humantouch_sources(Path(args.data), Path(args.root), {int(key.split("_")[1])}, cfg))
        glove = {s: (src.tactile[s]["pressure"][fi], src.contact[s][fi]) for s in SIDES}
        touching = {s: glove[s][1].sum(1) >= cfg.grasp_min_channels for s in SIDES}
        mapping = Path(args.root) / "meta/humantouch/tactile_mapping.json"
    else:
        from tv2s_data import touchscale as ts
        from tv2s_data.glove import load_glove_layout

        n_ch = load_glove_layout(HUMANTOUCH_MAPPING, "left").n_channels
        glove = {s: (np.zeros((T, n_ch), np.float32), np.zeros((T, n_ch), bool)) for s in SIDES}
        ep = ts.select_episodes(Path(args.ts_root), key)[0]
        counts = ts.contact_counts(ep, ts.frame_times(ep)[np.clip(fi, 0, None)])
        touching = {s: counts[s] >= ts.CONTACT_MIN_TAXELS for s in SIDES}

    # the recorded camera at every reference frame, in the simulation frame (visualisation)
    T_sim_cam = T_sim_world @ interp_poses(np.load(root / "camera.npz")["T_world_cam"], np.asarray(meta["times_s"]), t_ref)
    # head (cameras of the visualisation): HumanTouch's head pose, else the camera position
    if dataset == "humantouch":
        head = np.load(Path(args.data) / "episodes" / f"{key}.npz")["head_pose"][fi]
        head_pos = head[:, :3] @ T_sim_world[:3, :3].T + T_sim_world[:3, 3]
    else:
        Twc = interp_poses(np.load(root / "camera.npz")["T_world_cam"], np.asarray(meta["times_s"]), t_ref)
        head_pos = (T_sim_world @ Twc)[:, :3, 3]

    # palms in the simulation frame (the anchors of the held objects)
    palms = {}
    for s in SIDES:
        Tp = np.tile(np.eye(4), (T, 1, 1))
        Tp[:, :3, :3] = T_sim_hand[s][:3, :3] @ Rotation.from_quat(ret[f"{s}_palm_quat"][:, [1, 2, 3, 0]]).as_matrix()
        Tp[:, :3, 3] = ret[f"{s}_palm_pos"] @ T_sim_hand[s][:3, :3].T + T_sim_hand[s][:3, 3]
        palms[s] = Tp

    # objects: poses at the reference frames, in the simulation frame
    objects = []
    ids = [o["id"] for o in sc["objects"]]
    for o in sc["objects"]:
        poses = interp_poses(traj[o["id"]], np.asarray(meta["times_s"]), t_ref) if not o["fixed"] else \
            np.repeat(traj[o["id"]][:1], T, 0)
        objects.append(dict(id=o["id"], mesh=o["mesh"], fixed=o["fixed"], mass=o["mass"], material=o["material"],
                            poses=poses.astype(np.float32), tracked_poses=poses.astype(np.float32)))

    # which object each hand manipulates, frame by frame (the scene's clips)
    schedule = {s: np.full(T, -1, int) for s in SIDES}
    for c in scene["clips"]:
        for s in SIDES:
            oid = c[f"{s}_hand"]["object"]
            if oid in ids:
                schedule[s][(t_ref >= c["start_s"]) & (t_ref <= c["end_s"])] = ids.index(oid)
    # the holds: each glove contact run (gaps < 0.2 s closed, >= 0.3 s) that overlaps a clip of the hand, whole --
    # Gemini's clip edges are coarse (the can's grasp starts 1 s before its clip, while the hand is still down);
    # the schedule is extended over it
    hold_runs = {s: [] for s in SIDES}
    for s in SIDES:
        on = _close(touching[s], int(round(0.2 * hz)))
        for a, b in _runs(on):
            ks = schedule[s][a:b][schedule[s][a:b] >= 0]
            if b - a >= int(round(0.3 * hz)) and len(ks):
                k = int(np.bincount(ks).argmax())
                hold_runs[s].append((k, a, b))
                schedule[s][a:b] = k

    if args.objects == "anchored":
        import trimesh

        from tv2s_data.grasp_pose import fit_grasp
        from tv2s_env.tactile.mapping import TaxelSet

        recgen = json.loads((root / "objects.json").read_text()) if (root / "objects.json").exists() else {}
        fit = args.rests == "released" and args.grasp_fit and tactile_available
        containers = {x["id"] for x in scene["objects"] if x.get("is_container") and x["id"] in ids}
        support = support_id(scene)
        hands_fk = {s: WujiHand(s) for s in SIDES} if fit else {}
        taxels = {s: TaxelSet.load(str(REPO / "assets" / "tactile" / f"{args.taxels}_{s}.npz")) for s in SIDES} if fit else {}
        hz_ = hz
        for k, o in enumerate(objects):
            if o["fixed"]:
                continue
            holds, moves, hold_clips = [], [], []             # (side, start, end) frames the hand holds it
            holds = [(s, a, b) for s in SIDES for kk, a, b in hold_runs[s] if kk == k]
            holds.sort(key=lambda h: h[1])
            for s, a, b in holds:                             # the clip's from / to (Gemini) of each hold
                mid = t_ref[(a + b) // 2]
                c = next((c for c in scene["clips"] if c["start_s"] - 0.5 <= mid <= c["end_s"] + 0.5
                          and c[f"{s}_hand"]["object"] == o["id"]), None)
                floor = lambda where: 0.0 if where in ("table", support) else CONTAINER_FLOOR if where in containers else None
                moves.append((floor(c[f"{s}_hand"].get("from")), floor(c[f"{s}_hand"].get("to"))) if c else (None, None))
                hold_clips.append(c[f"{s}_hand"] if c else None)
            verts = np.asarray(trimesh.load(o["mesh"], force="mesh").vertices)
            ref = None
            if o["id"] in recgen and recgen[o["id"]].get("reference_time_s") is not None:
                ref = int(np.argmin(np.abs(t_ref - recgen[o["id"]]["reference_time_s"])))
            offsets, fits = None, []
            if fit and holds:
                video = rigid_chain(o["tracked_poses"].astype(np.float64), holds, palms, verts[:: max(1, len(verts) // 2000)], ref)
                offsets = []
                for s, a, b in holds:
                    m = (a + b) // 2
                    g = fit_grasp(hands_fk[s], taxels[s], ret[f"{s}_qpos"][a:b], palms[s][a:b], glove[s][1][a:b], verts,
                                  np.linalg.inv(palms[s][m]) @ video[m], floors=moves[len(offsets)])
                    offsets.append(g.offset)
                    fits.append(g)
            o["poses"], dev = anchor(o["tracked_poses"].astype(np.float64), holds, palms, verts, args.blend, int(round(0.3 * hz_)),
                                     args.rests, ref, offsets, moves)
            o["poses"] = o["poses"].astype(np.float32)
            o["holds"] = [(s, int(a), int(b)) for s, a, b in holds]
            o["hold_clips"] = hold_clips
            o["grasp_fits"] = [dict(f1=g.f1, precision=g.precision, recall=g.recall, penetration_mm=g.penetration_mm,
                                    video_cm=g.video_cm) for g in fits]
            print(f"  {o['id']:24s} {len(holds)} holds ({', '.join(f'{s} {a / hz_:.1f}-{b / hz_:.1f} s' for s, a, b in holds)}), "
                  f"anchored vs tracked: median {dev * 100:.1f} cm")
            for (s, a, b), g in zip(holds, fits):
                print(f"      grasp {s} {a / hz_:.1f} s: glove F1 {g.f1:.2f} (P {g.precision:.2f} R {g.recall:.2f}), "
                      f"hand penetration {g.penetration_mm:.1f} mm, {g.video_cm:.1f} cm from the video")

    if args.objects == "anchored" and args.rests == "released":
        # containers where the hands take objects out and put them back (Gemini's clips): placed around those
        # points (the anchored objects' poses at the grasps from / releases into them)
        for k, o in enumerate(objects):
            if o["fixed"] or o.get("holds") or not next((x for x in scene["objects"] if x["id"] == o["id"]), {}).get("is_container"):
                continue
            pts = []
            for q in objects:
                for (s, a, b), c in zip(q.get("holds", []), q.get("hold_clips", [])):
                    if c and c.get("from") == o["id"]:
                        pts.append(q["poses"][a, :3, 3])
                    if c and c.get("to") == o["id"]:
                        pts.append(q["poses"][b - 1, :3, 3])
            if pts:
                verts = np.asarray(trimesh.load(o["mesh"], force="mesh").vertices)
                new = place_container(o["poses"][0].astype(np.float64), verts[:: max(1, len(verts) // 20000)], np.array(pts))
                shift = new[:3, 3] - o["poses"][0, :3, 3]
                o["poses"] = o["poses"].copy()
                o["poses"][:, :3, 3] += shift.astype(np.float32)
                print(f"  {o['id']:24s} container: shifted {np.round(shift[:2] * 100, 1)} cm to hold the {len(pts)} grasps / releases in it")

    default_name = {"humantouch": lambda: f"ep{int(key.split('_')[1]):06d}_episode",
                    "egotouch": lambda: f"{key.split('__')[-1].replace('_', '')}_episode"}.get(dataset, lambda: f"{key[:8]}_episode")()
    clip = dict(metadata=dict(name=args.name or default_name, episode=key, dataset=dataset, clip=0, start=0, end=T,
                              hz=hz, source_frames=fi.astype(np.int64), types=["episode"], captions=[scene.get("scene", "")],
                              scene=None, episode_mode=True, hands_source=args.hands, tactile_available=tactile_available,
                              tactile_mapping=str(mapping), glove_scale=glove_scale, taxels=args.taxels,
                              finger_tip_row=22, clips=scene["clips"], T_sim_cam=T_sim_cam.astype(np.float32),
                              camera_intrinsics=meta["intrinsics"], camera_size=meta["size"], video=meta["video"],
                              video_fps=meta["video_fps"], undistort=bool(meta.get("undistorted", False))),
                head=dict(pos=head_pos.astype(np.float32)), hands={}, scene_objects=objects)
    for s in SIDES:
        R = T_sim_hand[s][:3, :3] @ Rotation.from_quat(ret[f"{s}_palm_quat"][:, [1, 2, 3, 0]]).as_matrix()
        p = ret[f"{s}_palm_pos"] @ T_sim_hand[s][:3, :3].T + T_sim_hand[s][:3, 3]
        quat = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
        g = wuji_reference_geometry(WujiHand(s), ret[f"{s}_qpos"], p, quat)
        kp_h = ret[f"{s}_keypoints"] @ T_sim_hand[s][:3, :3].T + T_sim_hand[s][:3, 3]
        clip["hands"][s] = dict(
            wrist_positions=p.astype(np.float32), wrist_quat=quat.astype(np.float32),
            dof_positions=ret[f"{s}_qpos"].astype(np.float32), human_keypoints=kp_h.astype(np.float32),
            fingertip_pos=g.tip_pos.astype(np.float32), fingertip_quat=g.tip_quat.astype(np.float32),
            ref_keypoints=g.keypoints.astype(np.float32), tactile_pressure=glove[s][0].astype(np.float32),
            tactile_contact=glove[s][1], object=None, object_schedule=schedule[s], semantic=None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = clip["metadata"]["name"]
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(clip, f)
    (out / "index.json").write_text(json.dumps(dict(clips=[dict(name=name, episode=key, frames=T)]), indent=1))
    low = min(float(g) for g in [clip["hands"][s]["ref_keypoints"][..., 2].min() for s in SIDES])
    print(f"{name}: {T} frames @ {hz:g} Hz ({T / hz:.1f} s), hands from {args.hands} (lowest keypoint {low * 100:+.1f} cm "
          f"above the table), {len(objects)} objects ({sum(o['fixed'] for o in objects)} fixed), schedule: "
          + ", ".join(f"{s} {[ids[i] for i in sorted(set(schedule[s]) - {-1})]}" for s in SIDES)
          + ("" if tactile_available else ", no tactile") + f" -> {out / name}.pkl")


if __name__ == "__main__":
    main()
