#!/usr/bin/env python3
"""EgoTouch / TouchAnything mocap hands placed in the reference camera (launch/reconstruct.sh).

The hands are measured: Rokoko gloves registered to Vive wrist trackers (`rokoko_hands.json`, 21 joints per hand in
the MediaPipe / OpenPose order, `aligned_to_vive`). What the recording lacks is the video camera: head-mounted, not
rigid to the chest tracker (a fixed chest -> camera offset leaves 34 px of median error), uncalibrated. The camera's
motion comes from the table tracked like an object (camera_traj_cotracker.npz: T_ct<-cref, in MoGe's metric), and
this fits the single similarity placing the Vive world in the reference camera,

    p_cref = s * R p_world + t,

over every frame together, from two cues:
  * the Rokoko joints, carried through T_ct<-cref and projected, against WiLoR's 2D joints (`wilor_hands.json`,
    camera frame at WiLoR's full-image focal 5000/256 * max(W, H));
  * their depth against the MoGe depth sampled at those 2D joints, which pins the scale s to the depth the object is
    lifted with, so that hand and object agree even where MoGe's metric is off.
A per-frame PnP on the same correspondences is not an alternative: WiLoR is ~20 px off on the black gloves, and 42
noisy points per frame leave the camera pose loose. Seven parameters shared by every frame are not loose.
launch/reconstruct.sh then rescales the depth by s (scale_depth.py) when it is off 1 and refits at s = 1.

WiLoR mislabels the hands: when one hand is hidden (grasping an object, behind the other), it calls the visible one
by the hidden one's side (pick_up_hammer/095941: 43 of 663 detections, on the frames the left hand holds the hammer;
twist_bottle_cap: ~6%). Those detections pull the one similarity towards the wrong hand, 2D and depth alike (there:
scale 0.58, the hand put ~2 m from the camera). With the episode's glove masks (EgoTouch `masks.npz`, one track
per glove), a detection whose joints lie on the other hand's glove is dropped (`glove_filter`).

The scale s (--scale-from): the 2D cue barely constrains it (a hand half the size, twice as far, projects nearly
onto the same joints), so the fit takes it from the depth cue, i.e. from MoGe, whose metric drifts by +-20% from
frame to frame on a moving head camera; through the one similarity and the table-tracked camera motion, the fit can
then shrink the hand and push it away (pick_up_hammer/095941: s = 0.48, the hand ~2 m from the camera at 40% of
its size in the image). `--scale-from hands` takes s from the hands' measured size instead: in every frame the
mocap hand, at its true size, is placed on WiLoR's joints alone (PnP, intrinsics.yaml), which gives its distance
to the camera, and s is the median of the depth under the hand over that distance (`hand_size_scale`); the
similarity is then fitted with s held. `fit` (Benchmark-Genesis') fits s with the rest.

--placement: the one similarity for the whole clip assumes the mocap world fixed to the camera, which it is not: the
transform world -> reference camera taken frame by frame turns by 20-50 deg and moves ~16 cm over a clip (the glove's
drift and the table-tracked camera's), so the fitted hand lags and shrinks. `per-frame` places the measured hand on
WiLoR's joints in every frame (PnP at its true size, at the depth under it) and smooths the transforms over time;
`smooth` (launch/reconstruct.sh's) then fits a transform varying smoothly over time (knots every --knot-frames) on
every frame together, WiLoR's joints with a robust loss, the glove's mask where WiLoR misses the hand (grasping,
hidden) and the depth: the mocap gives the hand's shape and fingers, the video where the hand is.

Writes --out: `{side}_joints` (T, 21, 3), `{side}_vertices` (points along the bones), `{side}_valid`, in OpenCV
coordinates of the reference camera; --transform-out: the similarity and its residuals; --overlay: the projected
mocap (lines) over the frames with WiLoR's joints (green).

Run in the `genesis` env:
    python scripts/reconstruction/place_mocap_hands.py --episode-dir ~/EgoTouch/Workbench/pick_up_screwdriver/X \\
        --camera-traj camera_traj_cotracker.npz --intrinsics intrinsics.yaml --depth-dir depth_moge \\
        --out mocap_joints_cotracker.npz --overlay hands_overlay_cotracker.mp4 --frames-dir all_frames
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")
# Parent of each of the 21 joints; the wrist (0) has none.
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-dir", required=True, help="TouchAnything episode (rokoko/wilor json)")
    parser.add_argument("--camera-traj", required=True, help="Table tracked by track_object_v2s2r.py")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--depth-dir", required=True, help="Depth the object is lifted with (uint16 mm)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--transform-out", default=None, help="Fitted similarity + diagnostics (.json)")
    parser.add_argument("--frames-dir", default=None, help="With --overlay, the frames to draw on")
    parser.add_argument("--overlay", default=None, help="Video of the projected mocap joints")
    parser.add_argument("--depth-weight", type=float, default=300.0,
                        help="Pixels per metre of depth residual; balances the two cues")
    parser.add_argument("--fixed-scale", type=float, default=None,
                        help="Hold s at this value instead of fitting it (1.0 = trust Vive metric)")
    parser.add_argument("--bone-samples", type=int, default=6)
    parser.add_argument("--fit-frames", default=None, metavar="A-B",
                        help="Fit the similarity on frames [A, B) only; every frame is still written. "
                        "For a stretch the one transform cannot explain (pick_up_wrench/094952: 70 px "
                        "and +52 cm past frame 300, the put-down)")
    parser.add_argument("--drift-knots", type=int, default=0, metavar="N",
                        help="Also fit a slowly varying translation (knots every N frames, linear in "
                        "between, kept smooth) on top of the similarity: the Vive world and the table-"
                        "tracked camera drift apart over a long clip (pick_up_wrench/094952: frames 0-150 "
                        "and 150-300 each fit at s=1, one transform for both only at s=0.48). 0 = off")
    parser.add_argument("--drift-smooth", type=float, default=1.0,
                        help="Weight of the knot-to-knot smoothness, in units of --depth-weight")
    parser.add_argument("--placement", choices=("global", "per-frame", "smooth"), default="global",
                        help="global: one similarity for the whole clip (Benchmark-Genesis'); per-frame: the measured hand "
                        "placed on WiLoR's 2D joints in every frame, smoothed over time (`per_frame_placement`); smooth: "
                        "that, then corrected by a placement varying smoothly over time fitted on every frame together, "
                        "with the gloves' masks where WiLoR misses the hand (`smooth_placement`)")
    parser.add_argument("--knot-frames", type=int, default=10, help="smooth: frames between two knots")
    parser.add_argument("--knot-smooth", type=float, default=1.0, help="smooth: weight of the knots' smoothness")
    parser.add_argument("--mask-weight", type=float, default=0.5,
                        help="smooth: weight of the glove-mask term (per px a joint lies outside its glove)")
    parser.add_argument("--pose-smooth", type=float, default=4.0,
                        help="per-frame: Gaussian smoothing of the placements over time, frames (1 sigma)")
    parser.add_argument("--max-pnp-px", type=float, default=25.0,
                        help="per-frame: a placement whose median reprojection error exceeds this is dropped")
    parser.add_argument("--hand-depth", choices=("moge", "pnp"), default="moge",
                        help="per-frame: the hand along its camera ray at the depth under it (moge: agrees with the "
                        "object, which is lifted with the same depth), or where the PnP puts it from its size")
    parser.add_argument("--scale-from", choices=("fit", "hands"), default="fit",
                        help="the scale s: fitted with the similarity (from the depth cue), or from the hands' measured "
                        "size (per-frame PnP on WiLoR against the depth under the hand); --fixed-scale overrides both")
    parser.add_argument("--glove-masks", default="auto",
                        help="the gloves' masks (EgoTouch masks.npz): WiLoR detections lying on the other hand's glove "
                        "are dropped; auto: <episode-dir>/masks.npz if there is one; none: off")
    parser.add_argument("--sides", nargs="+", default=list(SIDES), choices=SIDES,
                        help="Hands whose mocap is trusted. A hand left out does not constrain the fit and "
                        "is written with valid=False (e.g. a frozen Vive tracker: on pick_up_wrench/094952 "
                        "the right wrist sits still 1.6 m away and dragged the scale to 0.32)")
    return parser.parse_args()


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_matrix(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def similarity(x):
    return np.exp(x[6]), Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]


def glove_filter(wilor, masks, focal, width, height, min_frac=0.3, dilate=15):
    """({(t, side): False} for the WiLoR detections lying on the other hand's glove, {side: mask track}). masks (T, K, H, W) bool: the
    episode's glove tracks, whose sides are not labelled (EgoTouch: the left glove is track 1); each track's side is
    the one WiLoR gives the detections on it most often. A detection lies on a glove when > min_frac of its joints
    fall in that glove's mask (dilated by `dilate` px)."""
    kernel = np.ones((dilate, dilate), np.uint8)
    on = {}                                                   # (t, side) -> fraction of the joints on track 0, 1
    for t in range(min(len(wilor), len(masks))):
        grown = None
        for side in SIDES:
            detected = np.asarray(wilor[t].get(f"{side}_pos") or [], dtype=float)
            if detected.shape != (21, 3):
                continue
            if grown is None:
                grown = [cv2.dilate(masks[t, j].astype(np.uint8), kernel) > 0 for j in (0, 1)]
            uv = focal * detected[:, :2] / detected[:, 2:] + [width / 2, height / 2]
            px = np.clip(np.round(uv).astype(int), 0, [width - 1, height - 1])
            on[t, side] = [g[px[:, 1], px[:, 0]].mean() for g in grown]
    # the track WiLoR mostly calls left
    votes = sum(np.sign(f[0] - f[1]) * (1 if side == "left" else -1) for (t, side), f in on.items())
    track = {"left": 0 if votes > 0 else 1, "right": 1 if votes > 0 else 0}
    drop = {}
    for (t, side), f in on.items():
        own, other = f[track[side]], f[1 - track[side]]
        if other > own and other > min_frac:
            drop[t, side] = False
    counts = {side: sum(1 for t, s in drop if s == side) for side in SIDES}
    print(f"[hands] glove masks: left glove = track {track['left']}; WiLoR detections on the other hand's glove, dropped: "
          f"left {counts['left']}, right {counts['right']} of {len(on)}")
    return drop, track


def hand_size_scale(samples, K):
    """The depth's scale (depth units per true metre) from the hands' measured size: `samples` [(joints (21, 3) in the
    mocap world, WiLoR uv (21, 2), the depth under the hand)]; each hand is placed on its 2D joints at its true size
    (PnP with RANSAC), and the scale is the median over the hands of the depth under it over its PnP distance.
    Returns (scale, p10, p90, n) or None."""
    ratios = []
    for joints, uv, depth in samples:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(joints - joints[0], uv, K, None, reprojectionError=15)
        if not ok or inliers is None or len(inliers) < 12:
            continue
        z = np.median(((joints - joints[0]) @ cv2.Rodrigues(rvec)[0].T + tvec[:, 0])[:, 2])
        if 0.1 < z < 3.0:
            ratios.append(depth / z)
    if len(ratios) < 10:
        return None
    return float(np.median(ratios)), float(np.percentile(ratios, 10)), float(np.percentile(ratios, 90)), len(ratios)


def fit_global(world, keypoints, keypoint_ok, keypoint_depth, cam_R, cam_t, K, frames, offset, args):
    """Benchmark-Genesis' placement: one similarity world -> reference camera for the whole clip (x: rotation vector,
    translation, log scale), fitted on WiLoR's 2D joints and the depth under them, with an optional slow translation
    drift written into `offset` (--drift-knots)."""
    fu, fv, pu, pv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    n_frames = len(world)

    def project(x, frames):
        scale, rotation, translation = similarity(x)
        ref = scale * world[frames] @ rotation.T + translation + offset[frames][:, None]
        cam = np.einsum("tij,tkj->tki", cam_R[frames], ref) + cam_t[frames][:, None]
        uv = np.stack([fu * cam[..., 0] / cam[..., 2] + pu, fv * cam[..., 1] / cam[..., 2] + pv], -1)
        return cam, uv

    def residual(x):
        cam, uv = project(x, frames)
        pixel = (uv - keypoints[frames]) * keypoint_ok[frames][..., None]
        behind = np.clip(0.05 - cam[..., 2], 0, None) * 1e4
        hand_z = np.stack([np.median(cam[:, :21, 2], 1), np.median(cam[:, 21:, 2], 1)], 1)
        dz = np.nan_to_num((hand_z - keypoint_depth[frames]) * args.depth_weight)
        return np.concatenate([pixel.ravel(), behind.ravel(), dz.ravel()])

    # Initialise from a PnP on the first frame with detections, where T_ct<-cref is
    # (close to) the identity at the reference frame and composes with it elsewhere.
    t0 = frames[0]
    sel = keypoint_ok[t0]
    _, rvec, tvec, _ = cv2.solvePnPRansac(world[t0][sel], keypoints[t0][sel], K, None, reprojectionError=20)
    T_ct_w = to_matrix(cv2.Rodrigues(rvec)[0], tvec[:, 0])
    T_ref_w = np.linalg.inv(to_matrix(cam_R[t0], cam_t[t0])) @ T_ct_w
    x0 = np.r_[Rotation.from_matrix(T_ref_w[:3, :3]).as_rotvec(), T_ref_w[:3, 3], 0.0]

    # Coarse to fine: the single-frame PnP can start the hand tens of cm off in depth, and
    # at f_scale=15 px such a depth residual (300 px/m) is already an outlier the robust
    # loss stops pulling on -- on pick_up_wrench/094952 the fit sat at +68 cm, 2D fine.
    # A wide f_scale first brings the depth in, the tight one then rejects WiLoR's misses.
    if args.fixed_scale is not None:
        log_s = np.log(args.fixed_scale)
        y = np.r_[x0[:3], x0[3:6] * args.fixed_scale]           # the PnP placed the hand in true metres
        for f_scale in (200.0, 15.0):
            y = least_squares(lambda y: residual(np.r_[y, log_s]), y, loss="soft_l1", f_scale=f_scale).x
        x = np.r_[y, log_s]
    else:
        x = x0
        for f_scale in (200.0, 15.0):
            x = least_squares(residual, x, loss="soft_l1", f_scale=f_scale).x

    if args.drift_knots > 0:
        knots = np.unique(np.r_[np.arange(frames[0], frames[-1], args.drift_knots), frames[-1]])
        weights = np.stack([np.interp(np.arange(n_frames), knots, np.eye(len(knots))[i]) for i in range(len(knots))], 1)

        def residual_drift(d):
            offset[:] = weights @ d.reshape(-1, 3)
            smooth = np.diff(d.reshape(-1, 3), axis=0) * args.depth_weight * args.drift_smooth
            return np.concatenate([residual(x), smooth.ravel()])

        d = np.zeros(3 * len(knots))
        for f_scale in (200.0, 15.0):
            d = least_squares(residual_drift, d, loss="soft_l1", f_scale=f_scale).x
        offset[:] = weights @ d.reshape(-1, 3)
        steps = np.linalg.norm(d.reshape(-1, 3), axis=1)
        print(f"[hands] drift: {len(knots)} knots every {args.drift_knots} frames, offset "
              f"median {np.median(steps) * 100:.1f} cm, max {steps.max() * 100:.1f} cm")
    return x


def smooth_depth(depth, window=9, sigma=3.0):
    """The depth under a hand over time (T,), NaN where unknown, robustly smoothed: MoGe's metric wanders by several
    centimetres from one frame to the next, and a hand pinned to it jumps along its camera ray. A running median over
    `window` frames, then a Gaussian of `sigma` frames; unknown frames stay NaN."""
    from scipy.ndimage import gaussian_filter1d, median_filter

    ok = np.isfinite(depth)
    if ok.sum() < 3:
        return depth
    idx = np.flatnonzero(ok)
    filled = np.interp(np.arange(len(depth)), idx, depth[idx])
    out = gaussian_filter1d(median_filter(filled, size=window, mode="nearest"), sigma, mode="nearest")
    return np.where(ok, out, np.nan)


def per_frame_placement(world, uv, ok, depth, cam_R, cam_t, K, scale, args, side):
    """(T, 4, 4) world (true metres) -> reference camera, one per frame, for one hand: the measured hand (world (T, 21,
    3)) placed on WiLoR's 2D joints (uv (T, 21, 2), ok (T, 21)) by PnP at its true size, scaled into the depth's units
    (`scale`), along its camera ray at the depth under it (`depth` (T,), --hand-depth moge) or where the PnP put it,
    carried into the reference camera by the camera motion. The transforms are smoothed over time (a Gaussian of
    --pose-smooth frames on the rotations and translations, after dropping the PnPs above --max-pnp-px and the ones
    far from their neighbours), and interpolated across the frames without a placement: there the hand moves as the
    mocap says."""
    from scipy.spatial.transform import Slerp

    n = len(world)
    depth = smooth_depth(depth)
    # the transforms are smoothed and interpolated as (rotation, where the hand's mean position goes): about the
    # world's origin (~0.6 m from the hand), a turn of the transform would swing the hand along an arc
    pivot = world.reshape(-1, 3).mean(0)
    rots, trans, wts, frames = [], [], [], []
    for t in range(n):
        if ok[t].sum() < 12:
            continue
        obj, img = world[t][ok[t]], uv[t][ok[t]]
        c = obj.mean(0)
        found, rvec, tvec = cv2.solvePnP(obj - c, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not found:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj - c, img, K, None, rvec, tvec)
        proj, _ = cv2.projectPoints(obj - c, rvec, tvec, K, None)
        err = float(np.median(np.linalg.norm(proj[:, 0] - img, axis=1)))
        if err > args.max_pnp_px:
            continue
        R = cv2.Rodrigues(rvec)[0]
        p = scale * tvec[:, 0]                                    # the hand's centroid, camera of frame t, depth units
        if args.hand_depth == "moge" and np.isfinite(depth[t]):
            hand = scale * ((world[t] - c) @ R.T) + p
            p = p * (depth[t] / np.median(hand[:, 2]))              # slid along its ray to the depth under it
        T_ct = to_matrix(scale * R, p) @ to_matrix(np.eye(3), -c)    # world -> camera t
        G = np.linalg.inv(to_matrix(cam_R[t], cam_t[t])) @ T_ct
        rots.append(G[:3, :3] / scale), trans.append(G[:3, :3] @ pivot + G[:3, 3]), wts.append(1.0 / (err + 5.0))
        frames.append(t)
    frames, wts, trans = np.array(frames), np.array(wts), np.array(trans)
    if len(frames) < 3:
        raise SystemExit(f"[hands] {side}: {len(frames)} frames placed, too few for the per-frame placement")
    rots = Rotation.from_matrix(np.array(rots))

    def smooth(keep):
        """Gaussian-weighted means over time of the kept placements, at every placed frame."""
        R_s, t_s = [], []
        for t in frames:
            w = wts * keep * np.exp(-0.5 * ((frames - t) / args.pose_smooth) ** 2)
            w = w / w.sum()
            R_s.append(rots.mean(weights=w)), t_s.append(w @ trans)
        return Rotation.concatenate(R_s), np.array(t_s)

    keep = np.ones(len(frames))
    R_s, t_s = smooth(keep)
    # placements far from their neighbours' (a PnP flipped, a mislabelled detection the masks missed): dropped
    ang = np.degrees((R_s.inv() * rots).magnitude())
    off = np.linalg.norm(trans - t_s, axis=1)
    bad = (ang > max(15.0, 3 * np.median(ang))) | (off > max(0.05 * scale, 3 * np.median(off)))
    keep = (~bad).astype(float)
    R_s, t_s = smooth(keep)
    # every frame: interpolated between the placed ones, held at the ends
    times = np.arange(n)
    inside = np.clip(times, frames[0], frames[-1])
    R_all = Slerp(frames, R_s)(inside)
    t_all = np.stack([np.interp(inside, frames, t_s[:, i]) for i in range(3)], 1)
    G = np.tile(np.eye(4), (n, 1, 1))
    G[:, :3, :3] = scale * R_all.as_matrix()
    G[:, :3, 3] = t_all - G[:, :3, :3] @ pivot
    gaps = n - len(frames)
    print(f"[hands] {side}: placed by PnP on {len(frames)} frames ({int(bad.sum())} far from their neighbours dropped), "
          f"{gaps} interpolated; smoothed over {args.pose_smooth:g} frames; depth: {args.hand_depth}")
    return G


def smooth_placement(world, uv, ok, depth, mask_dt, cam_R, cam_t, K, scale, G_init, args, side):
    """(T, 4, 4) world (true metres) -> reference camera for one hand, varying smoothly over time: a correction of the
    per-frame placement `G_init` (a rotation vector and a translation, in the reference camera) at knots every
    --knot-frames frames, linear in between, fitted on every frame together:
      * WiLoR's 2D joints (uv (T, 21, 2), ok (T, 21)), robust loss: a bad frame weighs little instead of being cut;
      * the hand inside its glove's mask (mask_dt (T, h, w): px to the mask at half resolution, 0 inside): the frames
        WiLoR misses (the hand grasping, hidden) are still held where the glove is;
      * the hand's depth against the depth under it (depth (T,), under WiLoR's joints, else under the glove's mask);
      * smoothness of the knots;
    and a constant offset of the mocap's wrist in the hand's frame (the glove's wrist joint is not WiLoR's), a nuisance
    of the 2D term only: the written hand is the mocap's."""
    from scipy.sparse import lil_matrix

    n = len(world)
    fu, fv, pu, pv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    n_knots = max(2, int(np.ceil((n - 1) / args.knot_frames)) + 1)
    knots = np.linspace(0, n - 1, n_knots)
    W = np.stack([np.interp(np.arange(n), knots, np.eye(n_knots)[i]) for i in range(n_knots)], 1)      # (T, knots)
    t_init = G_init[:, :3, 3]
    # the correction turns about where the hand's mean position goes, not about the camera
    q = G_init[:, :3, :3] @ world.reshape(-1, 3).mean(0) + t_init                                       # (T, 3)
    # the hand's frame in the mocap world (wrist -> middle knuckle, palm normal), for the wrist offset
    a = world[:, 9] - world[:, 0]
    b = np.cross(a, world[:, 5] - world[:, 17])
    c = np.cross(b, a)
    H = np.stack([v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9) for v in (a, b, c)], 2)        # (T, 3, 3)
    MASK_JOINTS = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]
    has_mask = mask_dt is not None
    depth = smooth_depth(depth)
    has_depth = np.isfinite(depth)
    wm, wd = args.mask_weight, args.depth_weight
    ws_r, ws_t = args.knot_smooth * 50.0, args.knot_smooth * args.depth_weight     # px per rad, px per metre

    def hand(x, wrist_offset):
        d = W @ x[: 6 * n_knots].reshape(n_knots, 6)
        Rd = Rotation.from_rotvec(d[:, :3]).as_matrix()
        j = world.copy()
        if wrist_offset:
            j[:, 0] = j[:, 0] + np.einsum("tij,j->ti", H, x[6 * n_knots:])
        p0 = np.einsum("tij,tkj->tki", G_init[:, :3, :3], j) + t_init[:, None]
        p = np.einsum("tij,tkj->tki", Rd, p0 - q[:, None]) + (q + d[:, 3:])[:, None]
        cam = np.einsum("tij,tkj->tki", cam_R, p) + cam_t[:, None]
        z = np.clip(cam[..., 2], 0.05, None)
        return cam, np.stack([fu * cam[..., 0] / z + pu, fv * cam[..., 1] / z + pv], -1)

    def residual(x):
        cam, proj = hand(x, True)
        out = [((proj - uv) * ok[..., None]).ravel()]
        if has_mask:
            q = np.clip(np.round(proj[:, MASK_JOINTS] / 2).astype(int), 0, [mask_dt.shape[2] - 1, mask_dt.shape[1] - 1])
            out.append(wm * 2.0 * mask_dt[np.arange(n)[:, None], q[..., 1], q[..., 0]].astype(float).ravel())
        out.append(np.where(has_depth, (np.median(cam[..., 2], 1) - np.nan_to_num(depth)) * wd, 0.0))
        P = x[: 6 * n_knots].reshape(n_knots, 6)
        dP = np.diff(P, axis=0)
        out.append(np.c_[dP[:, :3] * ws_r, dP[:, 3:] * ws_t].ravel())
        return np.concatenate(out)

    # which parameters each residual depends on (each frame on its two knots, the 2D term on the wrist offset)
    n_par = 6 * n_knots + 3
    rows = []
    for t in range(n):
        ks = np.flatnonzero(W[t] > 0)
        rows.append(np.concatenate([np.arange(6 * k, 6 * k + 6) for k in ks]))
    blocks = [(21 * 2, True)] + ([(len(MASK_JOINTS), False)] if has_mask else []) + [(1, False)]
    S = lil_matrix((sum(b for b, _ in blocks) * n + (n_knots - 1) * 6, n_par), dtype=int)
    r0 = 0
    for size, offset in blocks:
        for t in range(n):
            for i in range(size):
                S[r0 + t * size + i, rows[t]] = 1
                if offset:
                    S[r0 + t * size + i, 6 * n_knots:] = 1
        r0 += size * n
    for k in range(n_knots - 1):
        for i in range(6):
            S[r0 + 6 * k + i, [6 * k + i, 6 * (k + 1) + i]] = 1
    x = least_squares(residual, np.zeros(n_par), jac_sparsity=S.tocsr(), loss="soft_l1", f_scale=10.0,
                      x_scale="jac", max_nfev=200).x
    d = W @ x[: 6 * n_knots].reshape(n_knots, 6)
    Rd = Rotation.from_rotvec(d[:, :3]).as_matrix()
    G = np.tile(np.eye(4), (n, 1, 1))
    G[:, :3, :3] = Rd @ G_init[:, :3, :3]
    G[:, :3, 3] = np.einsum("tij,tj->ti", Rd, t_init - q) + q + d[:, 3:]
    corr = np.degrees(np.linalg.norm(d[:, :3], axis=1))
    print(f"[hands] {side}: smooth placement, {n_knots} knots every {args.knot_frames} frames; correction of the "
          f"per-frame placement median {np.median(corr):.1f} deg / {np.median(np.linalg.norm(d[:, 3:], axis=1)) * 100:.1f} cm; "
          f"wrist offset (mocap vs WiLoR) {np.linalg.norm(x[6 * n_knots:]) * 100:.1f} cm (not applied to the hand)")
    return G


def main():
    args = parse_args()
    with open(args.intrinsics) as handle:
        calib = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calib[k]) for k in ("fu", "fv", "pu", "pv"))

    rokoko = read_jsonl(os.path.join(args.episode_dir, "rokoko_hands.json"))
    wilor = read_jsonl(os.path.join(args.episode_dir, "wilor_hands.json"))
    n_frames = len(rokoko)

    camera = np.load(args.camera_traj)
    if len(camera["rotation"]) != n_frames:
        raise SystemExit(f"camera track has {len(camera['rotation'])} frames, mocap {n_frames}")
    cam_valid = camera["valid"].astype(bool)
    # T_ct<-cref, the table's motion as seen by the camera.
    cam_R, cam_t = camera["rotation"], camera["translation"]

    depth_names = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    height, width = cv2.imread(os.path.join(args.depth_dir, depth_names[0]), cv2.IMREAD_UNCHANGED).shape
    wilor_focal = 5000.0 / 256.0 * max(width, height)

    keep, masks, track = {}, None, None
    mask_path = os.path.join(args.episode_dir, "masks.npz") if args.glove_masks == "auto" else args.glove_masks
    if args.glove_masks != "none" and os.path.exists(mask_path):
        masks = np.load(mask_path)["masks"].astype(bool)
        if masks.shape[-2:] == (height, width):
            keep, track = glove_filter(wilor, masks, wilor_focal, width, height)
        else:
            print(f"[hands] glove masks {masks.shape[-2:]} do not match the frames {(height, width)}: not used")
            masks = None
    elif args.glove_masks != "none":
        print("[hands] no glove masks: WiLoR's hand labels are taken as they are")

    world = np.zeros((n_frames, 42, 3))
    mocap_ok = np.zeros((n_frames, 2), dtype=bool)
    keypoints = np.zeros((n_frames, 42, 2))
    keypoint_ok = np.zeros((n_frames, 42), dtype=bool)
    keypoint_depth = np.full((n_frames, 2), np.nan)
    size_samples = []                  # (joints, WiLoR uv, depth under the hand) of the trusted hands (--scale-from hands)
    mask_depth = np.full((n_frames, 2), np.nan)                  # the depth under each glove's mask (+1 cm)
    for t in range(n_frames):
        depth = cv2.imread(os.path.join(args.depth_dir, depth_names[t]), cv2.IMREAD_UNCHANGED) / 1000.0
        for k, side in enumerate(SIDES):
            if masks is not None and t < len(masks) and masks[t, track[side]].sum() > 200:
                under = depth[masks[t, track[side]]]
                under = under[under > 0]
                if len(under) > 100:
                    mask_depth[t, k] = np.median(under) + 0.01
            joints = np.asarray(rokoko[t][f"{side}_pos"], dtype=float)
            if joints.shape == (21, 3) and np.isfinite(joints).all():
                world[t, 21 * k : 21 * k + 21] = joints
                mocap_ok[t, k] = True
            detected = np.asarray(wilor[t][f"{side}_pos"], dtype=float)
            if detected.shape != (21, 3) or not cam_valid[t] or not keep.get((t, side), True):
                continue
            uv = wilor_focal * detected[:, :2] / detected[:, 2:] + [width / 2, height / 2]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            keypoints[t, 21 * k : 21 * k + 21] = uv
            keypoint_ok[t, 21 * k : 21 * k + 21] = inside
            # The glove surface under the knuckles; the joint centres sit ~1 cm deeper,
            # which the median over the whole hand absorbs as a constant bias.
            px = np.minimum(np.round(uv[inside]).astype(int), [width - 1, height - 1])   # u < width rounds up to width
            samples = depth[px[:, 1], px[:, 0]]
            samples = samples[samples > 0]
            if len(samples) >= 8:
                keypoint_depth[t, k] = np.median(samples) + 0.01
                if mocap_ok[t, k] and side in args.sides and inside.sum() >= 12:
                    size_samples.append((joints[inside], uv[inside], keypoint_depth[t, k]))
    for k, side in enumerate(SIDES):
        if side not in args.sides:
            mocap_ok[:, k] = False
    keypoint_ok &= np.repeat(mocap_ok, 21, axis=1)

    offset = np.zeros((n_frames, 3))   # the drift, per frame (--drift-knots)
    K = np.array([[fu, 0, pu], [0, fv, pv], [0, 0, 1.0]])
    if (args.scale_from == "hands" or args.placement != "global") and args.fixed_scale is None:
        found = hand_size_scale(size_samples, K)
        if found is None:
            print("[hands] scale from the hands' size: too few hands placed, the scale is fitted")
        else:
            args.fixed_scale = found[0]
            print(f"[hands] scale from the hands' size (per-frame PnP of the measured hand on WiLoR, {found[3]} hands): "
                  f"depth / true = {found[0]:.3f} (p10 {found[1]:.3f}, p90 {found[2]:.3f})")

    frames = np.flatnonzero(keypoint_ok.any(axis=1))
    if args.placement in ("per-frame", "smooth") and args.fixed_scale is not None:
        scale = args.fixed_scale
        G = np.tile(np.eye(4), (2, n_frames, 1, 1))                   # per hand and frame: world (true m) -> cref
        for k, side in enumerate(SIDES):
            if mocap_ok[:, k].any():
                idx = slice(21 * k, 21 * k + 21)
                G[k] = per_frame_placement(world[:, idx], keypoints[:, idx], keypoint_ok[:, idx], keypoint_depth[:, k],
                                           cam_R, cam_t, K, scale, args, side)
                if args.placement == "smooth":
                    dt = None
                    if masks is not None:
                        kernel = np.ones((7, 7), np.uint8)
                        dt = np.stack([cv2.distanceTransform((cv2.dilate(m.astype(np.uint8), kernel) == 0).astype(np.uint8),
                                                             cv2.DIST_L2, 3)[::2, ::2].astype(np.float16)
                                       for m in masks[:n_frames, track[side]]])
                    G[k] = smooth_placement(world[:, idx], keypoints[:, idx], keypoint_ok[:, idx],
                                            np.where(np.isfinite(keypoint_depth[:, k]), keypoint_depth[:, k], mask_depth[:, k]),
                                            dt, cam_R, cam_t, K, scale, G[k], args, side)
        ref = np.concatenate([np.einsum("tij,tkj->tki", G[k, :, :3, :3], world[:, 21 * k:21 * k + 21]) + G[k, :, None, :3, 3]
                              for k in range(2)], 1)
        transform = {"placement": args.placement, "scale": scale,
                     "T_ref_world_per_frame": {side: G[k].tolist() for k, side in enumerate(SIDES)}}
    else:
        if args.placement != "global":
            print(f"[hands] {args.placement} placement needs the hands' scale: falling back to the global similarity")
        if args.fit_frames:
            first, last = (int(v) for v in args.fit_frames.split("-"))
            frames = frames[(frames >= first) & (frames < last)]
        x = fit_global(world, keypoints, keypoint_ok, keypoint_depth, cam_R, cam_t, K, frames, offset, args)
        scale, rotation, translation = similarity(x)
        ref = scale * world @ rotation.T + translation + offset[:, None]
        transform = {"placement": "global", "T_ref_world": to_matrix(scale * rotation, translation).tolist(),
                     "scale": scale, "drift_offset": offset.tolist()}   # (T, 3), added after T_ref_world (--drift-knots)

    # how the placed hands agree with the video
    cam = np.einsum("tij,tkj->tki", cam_R, ref) + cam_t[:, None]
    uv_all = np.stack([fu * cam[..., 0] / cam[..., 2] + pu, fv * cam[..., 1] / cam[..., 2] + pv], -1)
    cam_f, uv = cam[frames], uv_all[frames]
    err = np.linalg.norm(uv - keypoints[frames], axis=-1)[keypoint_ok[frames]]
    hand_z = np.stack([np.median(cam_f[:, :21, 2], 1), np.median(cam_f[:, 21:, 2], 1)], 1)
    dz = (hand_z - keypoint_depth[frames])[np.isfinite(keypoint_depth[frames]) & mocap_ok[frames]]
    print(f"[hands] world -> reference camera ({transform['placement']}): scale {scale:.3f}, "
          f"{len(frames)} frames, {keypoint_ok.sum()} 2D joints")
    print(f"[hands] reprojection vs WiLoR: median {np.median(err):.1f} px, p90 {np.percentile(err, 90):.1f} px")
    # the hand's size in the image against WiLoR's (palm: wrist and knuckles): ~1 when the scale is right
    palm = [0, 5, 9, 13, 17]
    spread = lambda p: np.linalg.norm(p - p.mean(-2, keepdims=True), axis=-1).mean(-1)
    sizes = []
    for k in range(2):
        idx = [21 * k + j for j in palm]
        ok = keypoint_ok[frames][:, idx].all(1)
        sizes += list(spread(uv[ok][:, idx]) / spread(keypoints[frames][ok][:, idx]))
    if sizes:
        print(f"[hands] hand size in the image vs WiLoR: median {np.median(sizes):.2f} (1: the right scale), "
              f"hand {np.median(cam_f[..., 2][keypoint_ok[frames]]) * 100:.0f} cm from the camera")
    print(f"[hands] hand depth vs MoGe: median {np.median(dz) * 100:+.1f} cm, "
          f"MAD {np.median(np.abs(dz - np.median(dz))) * 100:.1f} cm")

    out = {}
    for k, side in enumerate(SIDES):
        joints = ref[:, 21 * k : 21 * k + 21]
        # Bone-length drift is how a glove glitch shows; report it rather than hide it.
        if mocap_ok[:, k].any():
            bones = np.linalg.norm(joints[:, 1:] - joints[:, PARENTS[1:]], axis=-1)
            drift = np.abs(bones / np.median(bones[mocap_ok[:, k]], axis=0) - 1)[mocap_ok[:, k]]
            print(f"[hands] {side}: {mocap_ok[:, k].sum()}/{n_frames} frames, "
                  f"bone-length drift p99 {np.percentile(drift, 99) * 100:.1f}%")
        else:
            print(f"[hands] {side}: excluded (--sides), written with valid=False")
        weights = np.linspace(0, 1, args.bone_samples + 1)[1:]
        along = joints[:, PARENTS[1:], None] + weights[:, None] * (
            joints[:, 1:, None] - joints[:, PARENTS[1:], None]
        )
        out[f"{side}_joints"] = joints
        out[f"{side}_vertices"] = np.concatenate([joints, along.reshape(n_frames, -1, 3)], axis=1)
        out[f"{side}_valid"] = mocap_ok[:, k]
    np.savez(args.out, **out, source="touchanything_rokoko", scale=scale)
    print(f"Wrote {args.out}")

    if args.transform_out:
        with open(args.transform_out, "w") as handle:
            json.dump({**transform, "median_px": float(np.median(err)), "median_depth_cm": float(np.median(dz) * 100)},
                      handle, indent=2)

    if args.overlay and args.frames_dir:
        names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))
        writer = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        for t, name in enumerate(names):
            image = cv2.imread(os.path.join(args.frames_dir, name))
            for k, colour in enumerate([(0, 0, 255), (255, 0, 0)]):
                if not mocap_ok[t, k]:
                    continue
                pts = uv_all[t, 21 * k : 21 * k + 21]
                for j in range(1, 21):
                    a, b = pts[PARENTS[j]].astype(int), pts[j].astype(int)
                    cv2.line(image, tuple(a), tuple(b), colour, 2)
                for u, v in keypoints[t, 21 * k : 21 * k + 21][keypoint_ok[t, 21 * k : 21 * k + 21]]:
                    cv2.circle(image, (int(u), int(v)), 2, (0, 255, 0), -1)
            writer.write(image)
        writer.release()
        print(f"Wrote {args.overlay}  (lines: mocap, green: WiLoR)")


if __name__ == "__main__":
    main()
