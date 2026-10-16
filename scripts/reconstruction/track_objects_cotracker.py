#!/usr/bin/env python3
"""Perception: object trajectories with Benchmark-Genesis' CoTracker + depth method (the reference for the RL).

For every object of objects.json, Benchmark-Genesis' own scripts are run unchanged
(BG_SCRIPTS: ~/Benchmark-Genesis/reconstruction/scripts):
  1. track_points_cotracker.py (`sam3` env, CoTracker): --num-points points seeded in the object's SAM3
     mask on its RecGen reference frame, tracked forwards and backwards with CoTracker, each lifted to 3D
     with the metric depth under it (a patch median);
  2. pose_from_points.py (`genesis` env): per frame the rigid motion (Kabsch + RANSAC) carrying the
     reference frame's cloud onto that frame's, gated for impossible jumps, then a constant-velocity
     Kalman filter + RTS smoother (--smoother kalman, what Benchmark-Genesis' CoTracker pipeline runs).
The reference frame is RecGen's, where the object's pose is known: the RecGen mesh is in that camera's
frame, so T_cam_obj(t) = [R(t) | t(t)] T_cam_obj(ref), the same convention as the FoundationPose tracks
(object frame: the reference camera's axes at the mesh centre, object_mesh.obj).

Inputs written for those scripts under <perception>/cotracker/inputs/: the reference mask in their
masks/frame_XXXXXX_masks/<object>.png layout and, when the depth is monocular, the depth maps multiplied
by register_camera.py's scale (they read uint16 millimetres as metric).

Writes <perception>/cotracker/<object>.npz: T_cam_obj (T, 4, 4) (NaN where no pose), valid, inliers, and
T_world_obj when the camera is registered; <object>_points.npz and <object>_overlay.mp4 are the
Benchmark-Genesis outputs.

Needs the object models (track_objects_foundationpose.py --models-only: object_mesh.obj, model_centre).
Run in the `genesis` env (it calls the `sam3` env for CoTracker: PY_SAM3).
    python scripts/reconstruction/track_objects_cotracker.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_objects_sam3 import segmentation_masks  # noqa: E402

BG_SCRIPTS = Path(os.environ.get("BG_SCRIPTS", Path.home() / "Benchmark-Genesis/reconstruction/scripts"))
PY_SAM3 = os.environ.get("PY_SAM3", str(Path.home() / "miniconda3/envs/sam3/bin/python"))


def metric_depth_dir(root: Path, meta: dict) -> Path:
    """Depth maps the CoTracker lift can read as metric: the scaled copies when the depth is monocular."""
    scale = float(meta.get("depth_scale", 1.0))
    if abs(scale - 1.0) < 1e-6:
        return root / "depth"
    out = root / "cotracker" / "inputs" / "depth_metric"
    if not out.exists() or len(list(out.glob("*.png"))) != len(list((root / "depth").glob("*.png"))):
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted((root / "depth").glob("*.png")):
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float64) * scale
            cv2.imwrite(str(out / p.name), np.clip(d, 0, 65535).astype(np.uint16))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--objects", default=None, help="comma list of object ids (default: every reconstructed one)")
    ap.add_argument("--num-points", type=int, default=400)
    ap.add_argument("--smoother", default="kalman", choices=("kalman", "window"))
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()
    root = Path(args.perception).resolve()
    meta = json.loads((root / "frames.json").read_text())
    objects = json.loads((root / "objects.json").read_text())
    if args.objects:
        objects = {k: v for k, v in objects.items() if k in args.objects.split(",")}
    T = len(meta["frame_index"])
    out = root / "cotracker"
    (out / "inputs" / "masks").mkdir(parents=True, exist_ok=True)
    depth_dir = metric_depth_dir(root, meta)
    cam = np.load(root / "camera.npz")["T_world_cam"] if (root / "camera.npz").exists() else None

    for oid, info in objects.items():
        ref = int(info["reference_frame"])
        mask = segmentation_masks(root / "sam3" / f"{oid}.npz")[ref]
        mdir = out / "inputs" / "masks" / f"frame_{ref:06d}_masks"
        mdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mdir / f"{oid}.png"), mask.astype(np.uint8) * 255)
        points = out / f"{oid}_points.npz"
        cmd = [PY_SAM3, str(BG_SCRIPTS / "track_points_cotracker.py"), "--frames-dir", str(root / "frames"),
               "--depth-dir", str(depth_dir), "--masks-root", str(out / "inputs" / "masks"), "--object", oid,
               "--ref-frame", str(ref), "--intrinsics", str(root / "intrinsics.yaml"), "--out", str(points),
               "--num-points", str(args.num_points), "--fps", str(meta["sampled_fps"])]
        if not args.no_overlay:
            cmd += ["--overlay", str(out / f"{oid}_overlay.mp4")]
        r = subprocess.run(cmd, cwd=BG_SCRIPTS, capture_output=True, text=True,
                           env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        if r.returncode != 0 or not points.exists():
            print(f"  {oid:24s} CoTracker failed:\n{r.stderr[-1500:]}")
            continue
        rigid = [l for l in r.stdout.splitlines() if "rigid" in l.lower()]
        pose_file = out / f"{oid}_bg_pose.npz"
        r2 = subprocess.run([sys.executable, str(BG_SCRIPTS / "pose_from_points.py"), "--points", str(points),
                             "--out", str(pose_file), "--smoother", args.smoother], cwd=BG_SCRIPTS,
                            capture_output=True, text=True)
        if r2.returncode != 0 or not pose_file.exists():
            print(f"  {oid:24s} pose_from_points failed:\n{r2.stderr[-1500:]}")
            continue
        p = np.load(pose_file)
        if "model_centre" not in info:
            raise SystemExit(f"{oid}: no model yet: track_objects_foundationpose.py --models-only")
        T_ref = np.eye(4)
        T_ref[:3, 3] = info["model_centre"]                  # object_mesh.obj's frame at the reference frame
        rel = np.tile(np.eye(4), (T, 1, 1))
        rel[:, :3, :3], rel[:, :3, 3] = p["rotation"], p["translation"]
        poses = rel @ T_ref
        poses[~p["valid"]] = np.nan
        res = dict(T_cam_obj=poses, valid=p["valid"], inliers=p["inliers"], ref_frame=ref)
        if cam is not None:
            res["T_world_obj"] = cam @ poses
        np.savez(out / f"{oid}.npz", **res)
        step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
        print(f"  {oid:24s} {int(p['valid'].sum())}/{T} frames solved, median {np.median(p['inliers'][p['valid']]):.0f} "
              f"inliers, path {np.nansum(step) * 100:.0f} cm in the camera" + (f"  [{rigid[-1].strip()}]" if rigid else ""),
              flush=True)


if __name__ == "__main__":
    main()
