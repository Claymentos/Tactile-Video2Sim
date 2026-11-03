#!/usr/bin/env python3
"""The object's camera-relative trajectory from FoundationPose instead of CoTracker (launch/reconstruct.sh
OBJECT_TRACKER=foundationpose, step 8).

FoundationPose tracks the object's mesh on RGB-D: here the RGB frames, the depth of step 3 (MoGe or DA3, monocular)
and the RecGen mesh posed in the reference camera (recgen_out_moge/<object>/posed_mesh.obj). Its pose at the reference
frame is the mesh's own (identity), refined once; then FoundationPose's tracker runs forward to the clip's end and,
from the reference again, backward to frame 0. Where it loses the object (a fast swing towards the camera: the
silhouette IoU below --reregister-iou), FoundationPose's registration runs again on that frame's SAM3 mask and its
pose is kept if it fits the mask better. It writes the same file as poses_from_points.py:

    X_t = R_t X_ref + t_t     X_ref in the reference camera, X_t in the camera of frame t

so that compose_camera_motion.py and smooth_object_kalman.py follow unchanged. Each frame is scored by the overlap of
the mesh's silhouette with the object's SAM3 mask, both outside the gloves' masks (EgoTouch masks.npz: the glove
hides the held object, and the silhouette there is not checked): `iou`, and `inliers` = round(100 iou), the number
recon_reference.measure_object_poses reads as the frame's reliability with --object-poses foundationpose.

Run in the `foundationpose` env (FP_ROOT, default ~/FoundationPose):
    python scripts/reconstruction/track_object_foundationpose.py --recon data/egotouch/recon/<key> --object hammer
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recon", required=True, help="launch/reconstruct.sh output folder")
    ap.add_argument("--object", required=True, help="object id (its SAM3 mask and RecGen mesh)")
    ap.add_argument("--ref-frame", type=int, required=True, help="the reference frame (the mesh's)")
    ap.add_argument("--episode", default=None, help="episode folder with masks.npz (default <recon>/source)")
    ap.add_argument("--out", default=None, help="default <recon>/<object>_traj_camera_foundationpose.npz")
    ap.add_argument("--init-iter", type=int, default=5, help="refinement iterations at the reference frame")
    ap.add_argument("--track-iter", type=int, default=2, help="refinement iterations per tracked frame")
    ap.add_argument("--reregister-iou", type=float, default=0.3, help="register again below this silhouette IoU")
    ap.add_argument("--reregister-every", type=int, default=3, help="at most every this many frames")
    ap.add_argument("--min-mask-px", type=int, default=150, help="a mask this large (off the gloves) to register on")
    ap.add_argument("--overlay", default=None, help="mp4 of the tracked silhouette (default next to --out)")
    return ap.parse_args()


def silhouette(points, R, t, K, shape, radius=2):
    """The mesh's image mask: its surface samples projected and closed."""
    X = points @ R.T + t
    X = X[X[:, 2] > 1e-3]
    uv = np.round(X[:, :2] / X[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]).astype(int)
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    m = np.zeros(shape, np.uint8)
    m[uv[keep, 1], uv[keep, 0]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(cv2.dilate(m, k), cv2.MORPH_CLOSE, k).astype(bool)


def main() -> None:
    args = parse_args()
    fp_root = Path(os.environ.get("FP_ROOT", "~/FoundationPose")).expanduser()
    sys.path.insert(0, str(fp_root))
    cwd = os.getcwd()
    os.chdir(fp_root)                                   # its weights are found relative to the repository
    import torch
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    os.chdir(cwd)

    recon = Path(args.recon)
    out = Path(args.out or recon / f"{args.object}_traj_camera_foundationpose.npz")
    episode = Path(args.episode or recon / "source")
    frames = sorted((recon / "all_frames").glob("*.png"))
    depths = sorted((recon / "depth_moge").glob("*.png"))
    T = min(len(frames), len(depths))
    calib = yaml.safe_load((recon / "intrinsics.yaml").read_text())
    K = np.array([[calib["fu"], 0, calib["pu"]], [0, calib["fv"], calib["pv"]], [0, 0, 1]], float)
    mask_dir = recon / "video_segmentation" / "masks"
    obj_mask = lambda t: (cv2.imread(str(mask_dir / f"frame_{t:06d}_masks" / f"{args.object}.png"), cv2.IMREAD_GRAYSCALE)
                          if (mask_dir / f"frame_{t:06d}_masks" / f"{args.object}.png").exists() else None)
    gloves = np.load(episode / "masks.npz")["masks"].astype(bool) if (episode / "masks.npz").exists() else None

    # the mesh in the reference camera, centred: its pose at the reference frame is (I, c)
    mesh = trimesh.load(recon / "recgen_out_moge" / args.object / "posed_mesh.obj", force="mesh")
    c = np.asarray(mesh.vertices).mean(0)
    model = mesh.copy()
    model.vertices = np.asarray(mesh.vertices) - c
    samples = model.sample(20000)
    os.chdir(fp_root)
    est = FoundationPose(model_pts=model.vertices, model_normals=model.vertex_normals, mesh=model,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
                         debug=0, debug_dir=str(fp_root / "debug"))
    os.chdir(cwd)
    to_centered = est.get_tf_to_centered_mesh()
    to_centered = to_centered.cpu().numpy() if torch.is_tensor(to_centered) else np.asarray(to_centered)

    def load(t):
        rgb = cv2.cvtColor(cv2.imread(str(frames[t])), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(depths[t]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        return rgb, depth

    def start(T0):
        est.pose_last = torch.as_tensor(T0 @ np.linalg.inv(to_centered), dtype=torch.float, device="cuda")

    def visible(t, shape):
        """The object's SAM3 mask and the pixels off the gloves at frame t (None: no mask)."""
        m = obj_mask(t)
        if m is None:
            return None, None
        m = cv2.resize(m, shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127
        free = np.ones_like(m)
        if gloves is not None and t < len(gloves):
            g = (gloves[t, 0] | gloves[t, 1]).astype(np.uint8)
            free = ~cv2.resize(g, shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        return m, free

    def score(t, pose, shape):
        """IoU of the mesh's silhouette and the SAM3 mask, both off the gloves (NaN: no mask)."""
        m, free = visible(t, shape)
        if m is None:
            return np.nan
        a, b = silhouette(samples, pose[:3, :3], pose[:3, 3], K, shape) & free, m & free
        union = (a | b).sum()
        return (a & b).sum() / union if union else np.nan

    poses = np.tile(np.eye(4), (T, 1, 1))
    iou = np.full(T, np.nan)
    T0 = np.eye(4)
    T0[:3, 3] = c
    rgb, depth = load(args.ref_frame)
    start(T0)
    poses[args.ref_frame] = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.init_iter)
    iou[args.ref_frame] = score(args.ref_frame, poses[args.ref_frame], rgb.shape[:2])
    moved = np.linalg.norm(poses[args.ref_frame][:3, 3] - c) * 100
    print(f"[fp] reference frame {args.ref_frame}: refined {moved:.1f} cm from the RecGen pose")
    n_reg = 0
    for order in (range(args.ref_frame + 1, T), range(args.ref_frame - 1, -1, -1)):
        start(poses[args.ref_frame])
        last_reg = -10 ** 6
        for t in order:
            rgb, depth = load(t)
            poses[t] = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_iter)
            iou[t] = score(t, poses[t], rgb.shape[:2])
            # lost (a fast swing): registered again on the frame's SAM3 mask, kept if it fits the mask better
            m, free = visible(t, rgb.shape[:2])
            if (np.nan_to_num(iou[t]) < args.reregister_iou and m is not None and (m & free).sum() >= args.min_mask_px
                    and abs(t - last_reg) >= args.reregister_every):
                last_reg = t
                pose_r = est.register(K=K, rgb=rgb, depth=depth, ob_mask=m & free, iteration=args.init_iter)
                iou_r = score(t, pose_r, rgb.shape[:2])
                if iou_r > np.nan_to_num(iou[t]):
                    poses[t], iou[t] = pose_r, iou_r
                    n_reg += 1
                start(poses[t])
    print(f"[fp] registered again on {n_reg} frames (IoU < {args.reregister_iou})")

    # X_t = P_t (X_ref - c): rotation R_t, translation t_t - R_t c
    rotation = poses[:, :3, :3]
    translation = poses[:, :3, 3] - np.einsum("tij,j->ti", rotation, c)
    writer = None
    overlay = args.overlay or str(out.with_suffix(".mp4"))
    for t in range(T):
        rgb = cv2.imread(str(frames[t]))
        sil = silhouette(samples, poses[t][:3, :3], poses[t][:3, 3], K, rgb.shape[:2])
        if writer is None:
            writer = cv2.VideoWriter(overlay, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, rgb.shape[1::-1])
        cnt, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 1)
        cv2.putText(rgb, f"{t} iou {iou[t]:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        writer.write(rgb)
    writer.release()
    fit = np.nan_to_num(iou)
    np.savez(out, rotation=rotation, translation=translation, valid=np.ones(T, bool),
             inliers=np.round(100 * fit).astype(int), iou=iou, resting=np.zeros(T, bool),
             ref_frame=np.array(args.ref_frame))
    print(f"[fp] {T} frames; silhouette IoU with the SAM3 mask off the gloves: median {np.nanmedian(iou):.2f}, "
          f">= 0.5 on {np.mean(fit >= 0.5) * 100:.0f}% of the frames -> {out}, {overlay}")


if __name__ == "__main__":
    main()
