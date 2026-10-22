#!/usr/bin/env python3
"""HaWoR's hands (estimate_hands_hawor.py) placed in the reference camera (launch/reconstruct.sh, HANDS=hawor): what
place_mocap_hands.py does for a mocap, for a video with no hand measurement.

HaWoR gives each frame's hands in that frame's camera, at the depth its MANO size and the focal imply. Two things are
left to fix:
  * scale: HaWoR's metric (a MANO hand's size) and MoGe's are not the same. s = the median, over the frames, of MoGe's
    depth at the projected joints over HaWoR's joint depth: s takes the hands into the depth's units, as
    place_mocap_hands.py's similarity scale does, and reconstruct.sh then divides the depth by s (scale_depth.py) and
    reruns this at --fixed-scale 1, which puts the scene in the hands' metres;
  * depth jitter (--per-frame-depth, on by default): a monocular hand's depth wanders by centimetres from frame to frame,
    the depth map's does not. Each frame's hand is moved along the camera rays (which keeps its image) to MoGe's depth
    at its joints, the per-frame factor median-filtered and smoothed over time; frames whose factor is off s by more
    than --depth-gate (the hand behind the basket: the depth there is the basket's) take it from their neighbours.
The camera's motion (camera_traj_cotracker.npz, T_ct<-cref from the table) then carries each frame into the reference
camera: p_cref = R_t^T (p_ct - t_t).

Writes --out: `{side}_joints` (T, 21, 3), `{side}_vertices` (T, 778, 3) MANO, `{side}_valid` (T,), in OpenCV coordinates
of the reference camera (place_mocap_hands.py's keys); --transform-out: the scale and the residuals; --overlay: the
joints projected on the frames, with the SAM3 boxes HaWoR was given.

Run in the `genesis` env:
    python scripts/reconstruction/place_video_hands.py --hands <recon>/hawor/hawor_hands.npz \\
        --camera-traj camera_traj_cotracker.npz --intrinsics intrinsics.yaml --depth-dir depth_moge \\
        --out mocap_joints_cotracker.npz --transform-out mocap_to_camera_cotracker.json \\
        --frames-dir all_frames --overlay hands_overlay_cotracker.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d, median_filter

SIDES = ("left", "right")
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", required=True, help="hawor_hands.npz (estimate_hands_hawor.py)")
    ap.add_argument("--camera-traj", required=True, help="the table's track: T_ct<-cref")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--depth-dir", required=True, help="depth the object is lifted with (uint16 mm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transform-out", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--fps", type=float, default=30.0, help="of the overlay")
    ap.add_argument("--fixed-scale", type=float, default=None, help="hold s at this value instead of fitting it")
    ap.add_argument("--no-per-frame-depth", dest="per_frame_depth", action="store_false",
                    help="keep HaWoR's per-frame depth (times s)")
    ap.add_argument("--depth-gate", type=float, default=0.25,
                    help="a frame's depth factor further than this (relative) from s is not trusted")
    ap.add_argument("--depth-window", type=int, default=9, help="median filter of the per-frame factor (frames)")
    ap.add_argument("--depth-sigma", type=float, default=2.0, help="then a Gaussian (frames)")
    ap.add_argument("--sides", nargs="+", default=list(SIDES), choices=SIDES,
                    help="hands written as valid (one HaWoR gets wrong can be dropped)")
    return ap.parse_args()


def project(p, fu, fv, pu, pv):
    return np.stack([fu * p[..., 0] / p[..., 2] + pu, fv * p[..., 1] / p[..., 2] + pv], -1)


def main():
    args = parse_args()
    calib = yaml.safe_load(open(args.intrinsics))
    fu, fv, pu, pv = (float(calib[k]) for k in ("fu", "fv", "pu", "pv"))
    hands = np.load(args.hands)
    camera = np.load(args.camera_traj)
    cam_R, cam_t, cam_valid = camera["rotation"], camera["translation"], camera["valid"].astype(bool)
    T = len(cam_R)
    depth_names = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    if len(depth_names) != T or len(hands["left_joints"]) != T:
        raise SystemExit(f"frames: camera {T}, depth {len(depth_names)}, hands {len(hands['left_joints'])}")
    H, W = cv2.imread(os.path.join(args.depth_dir, depth_names[0]), cv2.IMREAD_UNCHANGED).shape

    joints = {s: hands[f"{s}_joints"].astype(np.float64) for s in SIDES}
    verts = {s: hands[f"{s}_vertices"].astype(np.float64) for s in SIDES}
    valid = {s: hands[f"{s}_valid"].astype(bool) & cam_valid & np.isfinite(joints[s]).all((1, 2))
             & (joints[s][..., 2] > 0.05).all(1) & (s in args.sides) for s in SIDES}

    # MoGe's depth at the joints over HaWoR's, per frame and hand. The depth map sees the glove's surface; the joint
    # centres sit ~1 cm deeper.
    ratio = {s: np.full(T, np.nan) for s in SIDES}
    for t in range(T):
        if not any(valid[s][t] for s in SIDES):
            continue
        depth = cv2.imread(os.path.join(args.depth_dir, depth_names[t]), cv2.IMREAD_UNCHANGED) / 1000.0
        for s in SIDES:
            if not valid[s][t]:
                continue
            uv = np.round(project(joints[s][t], fu, fv, pu, pv)).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            d = depth[uv[inside, 1], uv[inside, 0]]
            d = d[d > 0]
            if len(d) >= 8:
                ratio[s][t] = (np.median(d) + 0.01) / np.median(joints[s][t, inside, 2])
    pooled = np.concatenate([r[np.isfinite(r)] for r in ratio.values()])
    if not len(pooled):
        raise SystemExit("no frame where a hand projects onto the depth")
    scale = float(args.fixed_scale) if args.fixed_scale is not None else float(np.median(pooled))
    print(f"[hands] MoGe / HaWoR depth: median {np.median(pooled):.3f} (p10 {np.percentile(pooled, 10):.3f}, "
          f"p90 {np.percentile(pooled, 90):.3f}) over {len(pooled)} hand-frames -> scale {scale:.3f}"
          + (" (fixed)" if args.fixed_scale is not None else ""))

    out, report = {}, {}
    for s in SIDES:
        factor = np.full(T, scale)
        r = ratio[s]
        trusted = np.isfinite(r) & (np.abs(r / scale - 1) < args.depth_gate)
        if args.per_frame_depth and trusted.sum() >= 2:
            k = np.flatnonzero(trusted)
            f = median_filter(r[k], size=min(args.depth_window, len(k)) | 1, mode="nearest")
            f = np.interp(np.arange(T), k, f)
            factor = gaussian_filter1d(f, args.depth_sigma, mode="nearest") if args.depth_sigma > 0 else f
        # along the camera rays: scaling about the camera centre keeps the image
        j_ct, v_ct = joints[s] * factor[:, None, None], verts[s] * factor[:, None, None]
        to_ref = lambda p: np.einsum("tji,tkj->tki", cam_R, p - cam_t[:, None])       # R^T (p - t)
        out[f"{s}_joints"], out[f"{s}_vertices"], out[f"{s}_valid"] = to_ref(j_ct), to_ref(v_ct), valid[s]
        out[f"{s}_joints_frame_camera"] = j_ct
        resid = (r / factor - 1)[np.isfinite(r)]
        report[s] = dict(valid_frames=int(valid[s].sum()), trusted_depth_frames=int(trusted.sum()),
                         factor_min=float(factor[valid[s]].min()) if valid[s].any() else None,
                         factor_max=float(factor[valid[s]].max()) if valid[s].any() else None,
                         depth_residual_median_pct=float(np.median(np.abs(resid)) * 100) if len(resid) else None)
        if valid[s].any():
            bones = np.linalg.norm(j_ct[:, 1:] - j_ct[:, PARENTS[1:]], axis=-1)[valid[s]].sum(1)
            print(f"[hands] {s}: valid {valid[s].sum()}/{T} frames, depth trusted on {trusted.sum()}, factor "
                  f"{report[s]['factor_min']:.3f}-{report[s]['factor_max']:.3f}, |MoGe/hand - 1| median "
                  f"{report[s]['depth_residual_median_pct']:.1f}%, total bone length {np.median(bones) * 100:.1f} cm "
                  f"(compare across frames and runs)")
        else:
            print(f"[hands] {s}: no valid frame")
    np.savez(args.out, **out, source="hawor", scale=scale)
    print(f"Wrote {args.out}")
    if args.transform_out:
        json.dump(dict(scale=scale, per_frame_depth=args.per_frame_depth, sides=report), open(args.transform_out, "w"),
                  indent=2)

    if args.overlay and args.frames_dir:
        names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))
        writer = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        for t, name in enumerate(names[:T]):
            image = cv2.imread(os.path.join(args.frames_dir, name))
            for s, colour in zip(SIDES, [(0, 0, 255), (255, 0, 0)]):
                box = hands[f"{s}_box"][t] if f"{s}_box" in hands.files else np.full(4, np.nan)
                if np.isfinite(box).all():
                    cv2.rectangle(image, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 255, 0), 1)
                if not valid[s][t]:
                    continue
                uv = project(out[f"{s}_joints_frame_camera"][t], fu, fv, pu, pv)
                for j in range(1, 21):
                    cv2.line(image, tuple(uv[PARENTS[j]].astype(int)), tuple(uv[j].astype(int)), colour, 2)
            cv2.putText(image, f"{t}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(image)
        writer.release()
        print(f"Wrote {args.overlay}  (red: left, blue: right, green: the boxes HaWoR was given)")


if __name__ == "__main__":
    main()
