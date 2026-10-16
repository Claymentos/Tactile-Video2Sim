#!/usr/bin/env python3
"""Perception: place the video camera in the mocap world, fix the depth scale, find the table.

For recordings with a hand mocap and a tracked head (HumanTouch): the camera pose of every sampled frame is
T_world_cam(t) = T_world_head(t) @ X, with X the camera-from-head offset. The recorded calibration gives X
up to an error that puts the projected hands 10-15 cm off the gloves, so X (6 parameters, shared by the
whole episode) is refined: the mocap joints of each hand, projected, must fall inside that hand's SAM3
mask (segment_objects_sam3.py --hands) on every frame -- a robust least-squares fit on the masks' distance
transforms. Masks are assigned to hands per frame (the two masks closest to each hand's projection).
A per-frame PnP on detected 2D joints is not used: WiLoR misses these gloved hands.

EgoTouch (tv2s_data/egotouch.py): the hands (Rokoko) and the chest tracker (Vive) live in two different
worlds and the camera has no calibration at all, so T_world_cam(t) = W @ T_vive_chest(t) @ X with W (Vive ->
Rokoko) and X both fitted (12 parameters), and the focal length (the principal point at the image centre).
They are found from scratch: a static camera first (many random starts), then the chest-driven one
started from it; the glove masks are the dataset's (track 0 left, 1 right: no assignment). Without a chest
track the camera stays static. The two Rokoko gloves are not placed consistently with each other (the
resting hand can come out ~0.5 m off the table): the camera is fitted on the hand seen the most, then the
other hand gets its own rigid correction (6 parameters, about its mean position), and all are refined
together (camera.npz: hand_correction, applied by build_episode_clip.py). A coverage term (mask pixels
near the projected bones) keeps the hands their size: joints inside a mask alone let a far camera fit. --hands-only stops after this fit (the depth is estimated next, with the
fitted field of view); the full run then starts from that camera.

Then, with the camera placed:
  - depth scale: the mocap is metric, the monocular depth (MoGe) is not exactly: the median ratio of the
    joints' camera depth to the depth map under them (hands are ~1.5 cm in front of their joints:
    --hand-thickness) rescales every depth map and the RecGen meshes (reconstruct stage reads it);
  - table: the SAM3 mask of the scene's support surface ('table' by default), lifted with the scaled depth
    and carried into the world over all frames, fitted with a RANSAC plane: its normal, height and extent.

Writes <perception>/camera.npz: T_world_cam (T, 4, 4), X (4, 4), X0 (the calibration), W (4, 4),
hand_correction (2, 4, 4) (left, right: identity but for EgoTouch), depth_scale,
table_normal (3,), table_point (3,), table_extent (2, 3) (corners of its bounding rectangle in the plane),
reproj_px (T, 2) median distance of each hand's projected joints to its mask, and a check video
register_overlay.mp4 (joints before / after over the frames with the hand masks). A fitted focal length
is written to frames.json and intrinsics.yaml.

Run in the `genesis` env.
    python scripts/reconstruction/register_camera.py --perception data/humantouch/perception/episode_000002 --dataset humantouch --data data/humantouch
    python scripts/reconstruction/register_camera.py --perception data/egotouch/perception/<key> --dataset egotouch --data data/egotouch [--hands-only]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_objects_sam3 import segmentation_masks  # noqa: E402

SIDES = ("left", "right")
DS = 4                                         # distance transforms at 1/4 resolution
# MediaPipe hand bones (the coverage term: the projected skeleton must fill its mask)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def pose(p: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, p
    return m


def vec_pose(v: np.ndarray) -> np.ndarray:
    """(6,) rotation vector + translation -> (4, 4)."""
    return pose(v[3:6], Rotation.from_rotvec(v[:3]).as_matrix())


def pose_vec(m: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(m[:3, :3]).as_rotvec(), m[:3, 3]]


def humantouch_mocap(data: Path, root: Path, key: str, frame_index: np.ndarray) -> dict:
    """Head poses (T, 4, 4), hand joints (side -> (T, J, 3)) in the mocap world at the sampled frames,
    and the calibrated camera-from-head offset."""
    from tv2s_data.humantouch import camera_from_head, load_episode_meta

    e = int(key.split("_")[1])
    ep = np.load(data / "episodes" / f"{key}.npz")
    head = ep["head_pose"][frame_index]
    T_head = np.stack([pose(h[:3], Rotation.from_quat(h[3:7]).as_matrix()) for h in head])
    cal = load_episode_meta(root, e)["camera_calibration"]
    if str(ep["world"]) != "zup":
        raise SystemExit(f"{key}: extracted before the z-up conversion, re-run extract_humantouch.py --overwrite")
    X0 = camera_from_head(next(v for k, v in cal.items() if "extrinsics" in k))
    joints = {s: ep[f"{s}_skeleton"][frame_index].astype(np.float64) for s in SIDES}
    valid = {s: ep[f"{s}_skeleton_valid"][frame_index].reshape(len(frame_index), -1).mean(-1) > 0.5 for s in SIDES}
    return dict(T_head=T_head, X0=X0, joints=joints, valid=valid, moving=True)


def egotouch_mocap(data: Path, key: str, frame_index: np.ndarray, min_chest: float = 0.5) -> dict:
    """The Rokoko hands and the chest tracker (Vive world, gaps interpolated) at the sampled frames; the chest
    is dropped (static camera, T_head = identity) when tracked on less than `min_chest` of the frames."""
    ep = np.load(data / "episodes" / f"{key}.npz")
    fi = np.clip(frame_index, 0, len(ep["time_s"]) - 1)
    joints = {s: np.nan_to_num(ep[f"{s}_skeleton"][fi].astype(np.float64)) for s in SIDES}
    valid = {s: ep[f"{s}_skeleton_valid"][fi] | (np.isfinite(ep[f"{s}_skeleton"][fi]).all((1, 2))
                                                  & (np.abs(joints[s]).sum((1, 2)) > 0)) for s in SIDES}
    ok = ep["chest_valid"]
    moving = ok.mean() >= min_chest
    T_head = np.tile(np.eye(4), (len(fi), 1, 1))
    if moving:
        p, q, t = ep["chest_pose"][ok, :3], ep["chest_pose"][ok, 3:], np.flatnonzero(ok)
        tq = np.clip(fi, t[0], t[-1])
        T_head[:, :3, :3] = Slerp(t, Rotation.from_quat(q))(tq).as_matrix()
        T_head[:, :3, 3] = np.stack([np.interp(tq, t, p[:, k]) for k in range(3)], 1)
    return dict(T_head=T_head, X0=None, joints=joints, valid=valid, moving=moving)


def project(T_world_cam: np.ndarray, pts: np.ndarray, K: dict) -> tuple[np.ndarray, np.ndarray]:
    """(T, J, 3) world points -> (T, J, 2) pixels and (T, J) camera depth."""
    T_cw = np.linalg.inv(T_world_cam)
    q = np.einsum("tij,tkj->tki", T_cw[:, :3, :3], pts) + T_cw[:, None, :3, 3]
    z = q[..., 2]
    uv = np.stack([K["fx"] * q[..., 0] / z + K["cx"], K["fy"] * q[..., 1] / z + K["cy"]], -1)
    return uv, z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--dataset", choices=("humantouch", "egotouch"), required=True, help="where the mocap and head poses are")
    ap.add_argument("--data", required=True, help="prepared folder (episodes/)")
    ap.add_argument("--root", default=str(Path.home() / "HumanTouch" / "X001"), help="HumanTouch folder (calibration)")
    ap.add_argument("--support", default="auto", help="scene object the hands work on (plane fit); auto: "
                                                      "tv2s_data.scene.support_id")
    ap.add_argument("--rotation-only", action="store_true",
                    help="refine the calibration's rotation only (its translation is trusted): the masks constrain "
                         "the distance along the optical axis poorly")
    ap.add_argument("--gravity-weight", type=float, default=None,
                    help="px per degree of the support surface off horizontal (0: hands only). Default 2; 0 for EgoTouch, "
                         "whose Rokoko world is not gravity-aligned (assemble_scene.py takes the table as horizontal)")
    ap.add_argument("--hand-thickness", type=float, default=0.015, help="joints behind the visible hand surface (m)")
    ap.add_argument("--hands-only", action="store_true", help="EgoTouch: fit the camera and the focal length on the "
                                                               "hands only (before the depth), then stop")
    ap.add_argument("--fit-focal", action="store_true", help="fit the focal length (EgoTouch: always with --hands-only)")
    ap.add_argument("--starts", type=int, default=40, help="EgoTouch: random starts of the static camera")
    ap.add_argument("--coverage", type=float, default=None,
                    help="weight of the coverage term (mask pixels farther than a finger's radius from the projected "
                         "skeleton): without it a far / narrow camera fits as well, its hands drawn smaller inside the "
                         "masks. Default 1 for EgoTouch (no calibration), 0 for HumanTouch")
    ap.add_argument("--focal-prior", type=float, default=0.15,
                    help="fitted focal: relative spread of its prior (the intrinsics of frames.json; 0: free). The masks "
                         "constrain it poorly: joints inside a mask cost nothing, so a far, narrow camera fits too")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    K0, (W, H) = dict(meta["intrinsics"]), meta["size"]
    frame_index = np.asarray(meta["frame_index"])
    if args.dataset == "humantouch":
        mocap = humantouch_mocap(Path(args.data), Path(args.root).expanduser(), meta["key"], frame_index)
    else:
        mocap = egotouch_mocap(Path(args.data), meta["key"], frame_index)
    T = len(frame_index)
    ego = args.dataset == "egotouch"
    # per-hand rigid correction of the mocap (EgoTouch's second glove), about the hand's mean position
    hand_centre = {s: mocap["joints"][s][mocap["valid"][s]].reshape(-1, 3).mean(0) if mocap["valid"][s].any()
                   else np.zeros(3) for s in SIDES}
    corr = {s: np.eye(4) for s in SIDES}

    def hand_pose(h: np.ndarray, s: str) -> np.ndarray:
        R = Rotation.from_rotvec(h[:3]).as_matrix()
        return pose(hand_centre[s] + h[3:6] - R @ hand_centre[s], R)

    def J(s: str) -> np.ndarray:
        return mocap["joints"][s] @ corr[s][:3, :3].T + corr[s][:3, 3]
    if args.gravity_weight is None:
        args.gravity_weight = 0.0 if ego else 2.0
    if args.support == "auto":
        from tv2s_data.scene import support_id

        args.support = support_id(json.loads(Path(meta["scene"]).read_text()))
    fit_focal = args.fit_focal or (ego and args.hands_only)
    use_depth = not args.hands_only

    # hand masks -> distance transforms, assigned to hands per frame with the calibration's projection
    tracks = segmentation_masks(root / "sam3" / "hands.npz")                       # (N, T, H, W)
    labelled = str(np.load(root / "sam3" / "hands.npz")["prompt"]).startswith("dataset glove masks")
    cover_w = args.coverage if args.coverage is not None else (1.0 if args.dataset == "egotouch" else 0.0)
    rng_px = np.random.default_rng(0)
    mask_px = [[None] * T for _ in tracks]                                         # 40 pixels of every mask
    for n, tr in enumerate(tracks):
        for t in range(T):
            yy, xx = np.nonzero(tr[t][::2, ::2])
            if len(xx):
                k = rng_px.choice(len(xx), min(40, len(xx)), replace=False)
                mask_px[n][t] = np.stack([xx[k] * 2.0, yy[k] * 2.0], 1)
    bone_a, bone_b = np.array([b[0] for b in BONES]), np.array([b[1] for b in BONES])

    def coverage(uv: np.ndarray, z: np.ndarray, px: np.ndarray, f: float) -> np.ndarray:
        """Distance (px) of mask pixels to the projected bones beyond a finger's radius (1.1 cm)."""
        A, B = uv[bone_a], uv[bone_b]
        AB = B - A
        tt = np.clip(((px[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-6)[None], 0, 1)
        d = np.linalg.norm(px[:, None] - (A[None] + tt[..., None] * AB[None]), axis=-1).min(1)
        return np.maximum(d - f * 0.011 / max(float(np.median(z)), 0.05), 0.0)
    small = tracks[:, :, DS // 2::DS, DS // 2::DS]
    dts = np.stack([[distance_transform_edt(~m) * DS if m.any() else np.full(m.shape, np.inf) for m in tr]
                    for tr in small]).astype(np.float32)                               # (N, T, h, w) px

    def sample(dt: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Distance (px) to a mask at full-resolution pixels uv (J, 2); outside the image: 200 px."""
        uv = np.nan_to_num(uv, nan=-1e4, posinf=-1e4, neginf=-1e4)
        d = map_coordinates(np.where(np.isfinite(dt), dt, 500), [uv[:, 1] / DS - 0.5, uv[:, 0] / DS - 0.5],
                            order=1, mode="nearest")
        out = (uv[:, 0] < 0) | (uv[:, 0] >= W) | (uv[:, 1] < 0) | (uv[:, 1] >= H)
        return np.where(out, 200.0, d)

    # the camera from the parameters: HumanTouch x (X's correction); EgoTouch x, w (W's) and log focal
    camera0 = None
    if ego and not args.hands_only and (root / "camera.npz").exists():
        camera0 = np.load(root / "camera.npz")                                       # the --hands-only fit
        if "static" in camera0.files and bool(camera0["static"]):
            mocap["moving"] = False
            mocap["T_head"] = np.tile(np.eye(4), (T, 1, 1))
    X0 = mocap["X0"] if mocap["X0"] is not None else (camera0["X"] if camera0 is not None else np.eye(4))
    W0 = camera0["W"] if camera0 is not None else np.eye(4)
    n_x = 3 if args.rotation_only else 6
    n_w = 6 if ego and mocap["moving"] else 0

    def unpack(p: np.ndarray):
        x = np.r_[p[:n_x], np.zeros(6 - n_x)]
        Xm = X0 @ vec_pose(x) if (not ego or mocap["moving"]) else np.eye(4)
        Wm = W0 @ vec_pose(p[n_x:n_x + 6]) if n_w else (W0 @ vec_pose(p[:6]) if ego and not mocap["moving"] else np.eye(4))
        K = dict(K0)
        if fit_focal:
            f = K0["fx"] * math.exp(p[-1])
            K.update(fx=f, fy=f)
        return Wm @ mocap["T_head"] @ Xm, K

    n_p = (6 if (ego and not mocap["moving"]) else n_x + n_w) + int(fit_focal)

    def assign(Twc: np.ndarray, K: dict) -> dict[str, np.ndarray]:
        """Per frame, the track index of each hand (-1: none), by the projected joints' median distance."""
        if labelled:                                                                # the dataset's left / right masks
            return {s: np.where(mocap["valid"][s] & tracks[k].any((1, 2)), k, -1) for k, s in enumerate(SIDES)}
        if Twc is None:                                         # no camera yet: the left hand is on the left
            xs_ = [np.array([np.flatnonzero(m.any(0)).mean() if m.any() else np.nan for m in tr]) for tr in tracks]
            idx = {s: np.full(T, -1) for s in SIDES}
            for t in range(T):
                order = [n for n in np.argsort([x[t] for x in xs_]) if np.isfinite(xs_[n][t])]
                for s, n in zip(SIDES, order[:1] + order[-1:] if len(order) > 1 else []):
                    idx[s][t] = n if mocap["valid"][s][t] else -1
            return idx
        uvs = {s: project(Twc, J(s), K)[0] for s in SIDES}
        idx = {s: np.full(T, -1) for s in SIDES}
        for t in range(T):
            cost = np.array([[np.median(sample(dts[n, t], uvs[s][t])) if mocap["valid"][s][t] else 1e9
                              for n in range(len(dts))] for s in SIDES])
            for _ in SIDES:                                                         # greedy 2x2 matching
                si, n = np.unravel_index(np.argmin(cost), cost.shape)
                if cost[si, n] > 150:
                    break
                idx[SIDES[si]][t] = n
                cost[si, :], cost[:, n] = 1e9, 1e9
        return idx

    # the support surface is horizontal: its normal, measured in the camera from the depth (scale-free),
    # must be the world's z once the camera is placed -- this pins the camera's pitch and roll
    ys, xs = np.mgrid[0:H:8, 0:W:8]
    normals = {}
    sup = None
    if use_depth:
        sup = segmentation_masks(root / "sam3" / f"{args.support}.npz")
        for t in range(0, T, 5):
            d = cv2.imread(str(root / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED)[::8, ::8] / 1000.0
            ok = sup[t][::8, ::8] & (d > 0)
            if ok.sum() < 200:
                continue
            c = np.stack([(xs[ok] - K0["cx"]) / K0["fx"] * d[ok], (ys[ok] - K0["cy"]) / K0["fy"] * d[ok], d[ok]], -1)
            n = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][2]
            normals[t] = n * -np.sign(n[1])                    # towards the camera's up (-y)
    tn = np.array(sorted(normals))
    nc = np.stack([normals[t] for t in tn]) if len(tn) else np.zeros((0, 3))

    f_prior = K0["fx"]

    def focal_prior(f: float) -> np.ndarray:
        """the focal's log off the prior, in px-like units (as much weight as ~20 joints off by its spread)"""
        if args.focal_prior <= 0 or not fit_focal:
            return np.zeros(0)
        return np.array([math.log(f / f_prior) / args.focal_prior * 10.0 * np.sqrt(20)])

    def residual(p: np.ndarray, idx: dict, frames: np.ndarray | None = None, Twc_fn=None) -> np.ndarray:
        Twc, K = (Twc_fn or unpack)(p)
        r = []
        for s in SIDES:
            uv, z = project(Twc, J(s), K)
            for t in np.flatnonzero(idx[s] >= 0):
                if frames is None or frames[t]:
                    r.append(np.where(z[t] > 0, sample(dts[idx[s][t], t], uv[t]), 200.0))
                    px = mask_px[idx[s][t]][t]
                    if cover_w > 0 and px is not None:
                        r.append(cover_w * (coverage(uv[t], z[t], px, K["fx"]) if (z[t] > 0).all() else np.full(len(px), 200.0)))
        if len(tn) and args.gravity_weight > 0 and Twc_fn is None:   # degrees off vertical, x gravity_weight px
            up = np.einsum("tij,tj->ti", Twc[tn, :3, :3], nc)
            n_hand = sum(len(v) for v in r)
            r.append(np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))) * args.gravity_weight
                     * np.sqrt(n_hand / len(tn)))           # as much weight as all the hand residuals
        if Twc_fn is None:
            r.append(focal_prior(K["fx"]))
        return np.concatenate(r) if r else np.zeros(1)

    if ego and camera0 is None:
        # from scratch: a static camera (random starts), then the chest-driven one started from it
        idx_all = assign(None, K0)
        primary = max(SIDES, key=lambda s: (idx_all[s] >= 0).sum())        # the hand seen the most places the camera
        idx = {s: idx_all[s] if s == primary else np.full(T, -1) for s in SIDES}
        sub = np.zeros(T, bool)
        sub[::3] = True
        static = lambda v: (np.repeat(vec_pose(v[:6])[None], T, 0),
                            dict(K0, fx=K0["fx"] * math.exp(v[6]), fy=K0["fy"] * math.exp(v[6])))
        prior = lambda v, idx, frames, fn: np.r_[residual(v, idx, frames, fn), focal_prior(fn(v)[1]["fx"])]
        centre = hand_centre[primary]
        best = None
        for k in range(args.starts):
            R = Rotation.random(random_state=k).as_matrix()
            v0 = np.r_[Rotation.from_matrix(R).as_rotvec(), centre - R[:, 2] * 0.6, 0.0]    # 60 cm from the hands
            sol = least_squares(lambda v: prior(v, idx, sub, static), v0, loss="soft_l1", f_scale=10.0,
                                x_scale=np.r_[[0.3] * 3, [0.05] * 3, 0.2], max_nfev=200)
            cands_s = (cands_s if k else []) + [(np.median(residual(sol.x, idx, sub, static)), sol.x)]
        best = None
        for m, v in sorted(cands_s, key=lambda c: c[0])[:5]:            # the best starts, refined on every frame
            v = least_squares(lambda u: prior(u, idx, None, static), v, loss="soft_l1", f_scale=10.0,
                              x_scale=np.r_[[0.3] * 3, [0.05] * 3, 0.2], max_nfev=300).x
            m = np.median(residual(v, idx, None, static))
            if best is None or m < best[0]:
                best = (m, v)
        print(f"  static camera on the {primary} hand: median distance of the joints to their mask {best[0]:.1f} px "
              f"(best of {args.starts} starts), focal x{math.exp(best[1][6]):.2f}", flush=True)
        Twc_static = vec_pose(best[1][:6])
        K0.update(fx=K0["fx"] * math.exp(best[1][6]), fy=K0["fy"] * math.exp(best[1][6]))
        if mocap["moving"]:
            cands = []
            for k in range(12):                                # X's rotation is unknown: W from each guess
                Xr = pose(np.zeros(3), Rotation.random(random_state=100 + k).as_matrix())
                W0 = Twc_static @ np.linalg.inv(mocap["T_head"][T // 2] @ Xr)
                X0 = Xr
                p0 = np.r_[np.zeros(n_x + n_w), np.zeros(int(fit_focal))]
                sol = least_squares(lambda v: residual(v, idx, sub), p0, loss="soft_l1", f_scale=10.0,
                                    x_scale=np.r_[[0.3] * 3, [0.05] * 3, [0.3] * 3, [0.05] * 3, [0.2] * int(fit_focal)][:n_p],
                                    max_nfev=300)
                cands.append((np.median(residual(sol.x, idx, sub)), sol.x, W0, X0))
            m, p_best, W0, X0 = min(cands, key=lambda c: c[0])
            print(f"  camera on the chest tracker: {m:.1f} px", flush=True)
            if m > 1.2 * best[0] + 0.5:                      # the head camera does not follow the chest tracker
                mocap["moving"] = False
                print("  worse than a static camera: static camera", flush=True)
            else:
                W0, X0 = W0 @ vec_pose(p_best[n_x:n_x + 6]), X0 @ vec_pose(p_best[:6])
                if fit_focal:
                    K0.update(fx=K0["fx"] * math.exp(p_best[-1]), fy=K0["fy"] * math.exp(p_best[-1]))
        if not mocap["moving"]:
            W0, X0 = Twc_static, np.eye(4)
            mocap["T_head"] = np.tile(np.eye(4), (T, 1, 1))
            n_w = 0
            n_p = 6 + int(fit_focal)
            print("  static camera", flush=True)
        other = [s for s in SIDES if s != primary][0]
        if (idx_all[other] >= 0).sum() >= 3:                   # the other glove: its own rigid correction
            only = {s: idx_all[s] if s == other else np.full(T, -1) for s in SIDES}
            p_cam = np.zeros(n_p)
            Twc_c, K_c = unpack(p_cam)
            cands = []
            for k in range(24):                                 # starts: its mask centroid lifted at several depths
                tv = np.flatnonzero(only[other] >= 0)[k % max(1, len(np.flatnonzero(only[other] >= 0)))]
                yy, xx = np.nonzero(tracks[only[other][tv], tv])
                depth = 0.3 + 0.1 * (k % 8)
                ray = np.array([(xx.mean() - K_c["cx"]) / K_c["fx"], (yy.mean() - K_c["cy"]) / K_c["fy"], 1.0]) * depth
                target = Twc_c[tv, :3, :3] @ ray + Twc_c[tv, :3, 3]
                h0 = np.r_[Rotation.random(random_state=200 + k).as_rotvec() * (k >= 8) * 0.5,
                           target - mocap["joints"][other][tv].mean(0)]

                def res_h(h):
                    corr[other] = hand_pose(h, other)
                    return residual(p_cam, only)
                sol = least_squares(res_h, h0, loss="soft_l1", f_scale=10.0, x_scale=np.r_[[0.3] * 3, [0.05] * 3], max_nfev=200)
                cands.append((np.median(res_h(sol.x)), sol.x))
            m, h_best = min(cands, key=lambda c: c[0])
            corr[other] = hand_pose(h_best, other)
            print(f"  {other} glove: rigid correction {np.degrees(np.linalg.norm(h_best[:3])):.0f} deg, "
                  f"{np.linalg.norm(h_best[3:]) * 100:.0f} cm ({m:.1f} px)", flush=True)

    x = np.zeros(n_p)
    corrected = [s for s in SIDES if not np.allclose(corr[s], np.eye(4))]
    if ego and camera0 is not None and "hand_correction" in camera0.files:          # the --hands-only fit's
        for k, s in enumerate(SIDES):
            corr[s] = camera0["hand_correction"][k]
        corrected = [s for s in SIDES if not np.allclose(corr[s], np.eye(4))]
    base = {s: corr[s].copy() for s in SIDES}

    def residual_all(v, idx):
        for k, s in enumerate(corrected):                # a small motion about the corrected hand's centre
            h, c = v[n_p + 6 * k:n_p + 6 * k + 6], base[s][:3, :3] @ hand_centre[s] + base[s][:3, 3]
            R = Rotation.from_rotvec(h[:3]).as_matrix()
            corr[s] = pose(c + h[3:6] - R @ c, R) @ base[s]
        return residual(v[:n_p], idx)
    x = np.zeros(n_p + 6 * len(corrected))
    for it in range(3):                                                             # assign, fit, reassign
        idx = assign(*unpack(x[:n_p]))
        before = np.median(residual_all(x, idx))
        x = least_squares(lambda v: residual_all(v, idx), x, loss="soft_l1", f_scale=10.0, x_scale=0.05).x
        residual_all(x, idx)
        print(f"  pass {it + 1}: {sum((idx[s] >= 0).sum() for s in SIDES)} hand-frames, median distance of the "
              f"joints to their mask {before:.1f} -> {np.median(residual_all(x, idx)):.1f} px", flush=True)
        if labelled:
            break
    x = x[:n_p]
    Twc, K = unpack(x)
    if ego:
        Xf = X0 @ vec_pose(np.r_[x[:n_x], np.zeros(6 - n_x)]) if mocap["moving"] else np.eye(4)
        Wf = Twc[0] @ np.linalg.inv(mocap["T_head"][0] @ Xf)
        print(f"  camera: focal {K['fx']:.1f} px (field of view {math.degrees(2 * math.atan(W / 2 / K['fx'])):.1f} deg)"
              + ("" if mocap["moving"] else ", static"))
    else:
        Xf, Wf = Twc[0], np.eye(4)
        Xf = np.linalg.inv(mocap["T_head"][0]) @ Twc[0]
        corr = Rotation.from_rotvec(x[:3]).magnitude()
        print(f"  correction of the calibration: {np.degrees(corr):.1f} deg, {np.linalg.norm(x[3:6]) * 100:.1f} cm")
    reproj = np.full((T, 2), np.nan)
    for k, s in enumerate(SIDES):
        uv, _ = project(Twc, J(s), K)
        for t in np.flatnonzero(idx[s] >= 0):
            reproj[t, k] = np.median(sample(dts[idx[s][t], t], uv[t]))
    if fit_focal:                                                                   # the fitted intrinsics
        meta["intrinsics"] = dict(K)
        meta["fov_x_deg"] = math.degrees(2 * math.atan(W / (2 * K["fx"])))
        meta["intrinsics_estimated"] = False
        (root / "intrinsics.yaml").write_text(f"fu: {K['fx']}\nfv: {K['fy']}\npu: {K['cx']}\npv: {K['cy']}\n")
    X0_out = mocap["X0"] if mocap["X0"] is not None else Xf

    if not use_depth:
        np.savez(root / "camera.npz", T_world_cam=Twc, X=Xf, X0=X0_out, W=Wf, reproj_px=reproj, static=not mocap["moving"],
                 hand_correction=np.stack([corr[s] for s in SIDES]))
        (root / "frames.json").write_text(json.dumps(dict(meta, world="mocap"), indent=1))
        print(f"-> {root / 'camera.npz'} (hands only: estimate the depth with this field of view, then run again)")
        return

    # depth scale: mocap joint depth vs the depth map under the joints that fall inside their mask
    ratios = []
    for k, s in enumerate(SIDES):
        uv, z = project(Twc, J(s), K)
        for t in range(0, T, 2):
            if idx[s][t] < 0:
                continue
            d = cv2.imread(str(root / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED) / 1000.0
            m = tracks[idx[s][t], t]
            for (u, v), zz in zip(np.nan_to_num(uv[t], nan=-1).astype(int), z[t]):
                if 0 <= u < W and 0 <= v < H and m[v, u] and d[v, u] > 0:
                    ratios.append((zz - args.hand_thickness) / d[v, u])
    scale = float(np.median(ratios)) if ratios else 1.0
    print(f"  depth scale (mocap / {meta.get('depth')} depth): {scale:.3f} over {len(ratios)} joints")

    # the table: its SAM3 mask, lifted with the scaled depth, in the world, RANSAC plane
    pts = []
    for t in range(0, T, 5):
        d = cv2.imread(str(root / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED)[::8, ::8] / 1000.0 * scale
        ok = sup[t][::8, ::8] & (d > 0)
        c = np.stack([(xs[ok] - K["cx"]) / K["fx"] * d[ok], (ys[ok] - K["cy"]) / K["fy"] * d[ok], d[ok]], -1)
        pts.append(c @ Twc[t, :3, :3].T + Twc[t, :3, 3])
    pts = np.concatenate(pts)
    rng, best = np.random.default_rng(0), (0, None)
    for _ in range(500):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - a) @ n) < 0.01
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    inl = pts[best[1]]
    centre = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
    normal = vt[2] * np.sign(vt[2] @ (Twc[:, :3, 3].mean(0) - centre))             # towards the camera: up
    e1 = vt[0]
    e2 = np.cross(normal, e1)
    a1, a2 = (inl - centre) @ e1, (inl - centre) @ e2
    lo, hi = (np.percentile(a1, 1), np.percentile(a2, 1)), (np.percentile(a1, 99), np.percentile(a2, 99))
    extent = np.array([centre + lo[0] * e1 + lo[1] * e2, centre + hi[0] * e1 + hi[1] * e2])
    tilt = np.degrees(np.arccos(abs(normal[2])))
    print(f"  table: {best[0]}/{len(pts)} points within 1 cm of a plane, height {centre[2]:.3f} m, "
          f"tilt {tilt:.1f} deg from the world's z, {hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} m")

    np.savez(root / "camera.npz", T_world_cam=Twc, X=Xf, X0=X0_out, W=Wf, depth_scale=scale, table_normal=normal,
             table_point=centre, table_extent=extent, reproj_px=reproj, static=not mocap["moving"],
             hand_correction=np.stack([corr[s] for s in SIDES]))
    (root / "frames.json").write_text(json.dumps(dict(meta, depth_scale=scale, world="mocap"), indent=1))

    # check video: joints with the calibration (grey) and registered (colour) over the hand masks
    writer = cv2.VideoWriter(str(root / "register_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["sampled_fps"], (W, H))
    T0 = mocap["T_head"] @ mocap["X0"] if mocap["X0"] is not None else None
    for t in range(T):
        img = cv2.imread(str(root / "frames" / f"{t:06d}.jpg"))
        for n in range(len(tracks)):
            img[tracks[n, t]] = (0.6 * img[tracks[n, t]] + [0, 60, 0]).astype(np.uint8)
        for s, c in (("left", (255, 120, 0)), ("right", (0, 120, 255))):
            if not mocap["valid"][s][t]:
                continue
            for Tw, col, Kd in ((T0, (160, 160, 160), K0), (Twc, c, K)):
                if Tw is None:
                    continue
                uv, z = project(Tw[t:t + 1], J(s)[t:t + 1], Kd)
                for (u, v), zz in zip(uv[0], z[0]):
                    if zz > 0 and np.isfinite(u) and np.isfinite(v) and abs(u) < 1e5 and abs(v) < 1e5:
                        cv2.circle(img, (int(u), int(v)), 3, col, -1)
        writer.write(img)
    writer.release()
    print(f"-> {root / 'camera.npz'}, {root / 'register_overlay.mp4'}")


if __name__ == "__main__":
    main()
