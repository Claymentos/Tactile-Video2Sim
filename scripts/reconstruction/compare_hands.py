#!/usr/bin/env python3
"""HaWoR's hands (from the video) against the dataset's hand mocap (HumanTouch's ground truth).

Both as the 21 MediaPipe / OpenPose joints, in the world (the mocap's z-up frame; HaWoR's are carried there
by the registered camera, register_camera.py), on HaWoR's frames (estimate_hands_hawor.py). Per hand and over
the frames both have:
  - wrist error and MPJPE (mean joint error), cm: where the hand is;
  - root-relative MPJPE (wrists aligned) and PA-MPJPE (per-frame Procrustes: rotation, translation, scale),
    cm: the hand's articulation, whatever its placement;
  - in the camera, the error split along the viewing ray (depth) and across it: a monocular estimate is
    expected to be much worse in depth;
  - the reprojection distance in the image (px).
The joint definitions differ a little (the mocap's OpenXR skeleton vs MANO's joint regressor): a few mm
of the error are that.

Writes <perception>/hawor/compare_mocap.json and compare_mocap.png (errors over time).

    python scripts/reconstruction/compare_hands.py --perception data/humantouch/perception/episode_000002 --data data/humantouch
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_hands_hawor import interp_poses  # noqa: E402
from tv2s_data.humantouch import SIDES, mediapipe_keypoints  # noqa: E402


def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a aligned onto b (both (J, 3)) with the similarity minimising the squared error."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A ** 2).sum()
    return s * A @ R + mb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (hawor/hands.npz, camera.npz)")
    ap.add_argument("--data", required=True, help="HumanTouch prepared folder (episodes/)")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    hw = np.load(root / "hawor" / "hands.npz")
    ep = np.load(Path(args.data) / "episodes" / f"{meta['key']}.npz")
    fi = hw["frame_index"]
    t = fi / meta["video_fps"]
    Twc = interp_poses(np.load(root / "camera.npz")["T_world_cam"], np.asarray(meta["times_s"]), t)
    Tcw = np.linalg.inv(Twc)
    K = meta["intrinsics"]
    report, series = {}, {}
    for s in SIDES:
        gt = mediapipe_keypoints(ep[f"{s}_skeleton"][fi]).astype(np.float64)                      # (T, 21, 3)
        gt_ok = np.asarray(ep[f"{s}_skeleton_valid"][fi]).reshape(len(fi), -1).mean(-1) > 0.5
        est = hw[f"{s}_joints_world"].astype(np.float64)
        ok = hw[f"{s}_valid"].astype(bool) & gt_ok & np.isfinite(est).all((1, 2))
        if ok.sum() < 5:
            report[s] = dict(frames=int(ok.sum()))
            continue
        g, e = gt[ok], est[ok]
        err = np.linalg.norm(e - g, axis=-1)                                                     # (n, 21)
        rel = np.linalg.norm((e - e[:, :1]) - (g - g[:, :1]), axis=-1)
        pa = np.array([np.linalg.norm(procrustes(e[k], g[k]) - g[k], axis=-1).mean() for k in range(len(g))])
        # camera frame: along / across the viewing ray of each joint
        R, tr = Tcw[ok, :3, :3], Tcw[ok, :3, 3]
        gc, ec = np.einsum("nij,nkj->nki", R, g) + tr[:, None], np.einsum("nij,nkj->nki", R, e) + tr[:, None]
        ray = gc / np.linalg.norm(gc, axis=-1, keepdims=True)
        d = ec - gc
        along = (d * ray).sum(-1)
        across = np.linalg.norm(d - along[..., None] * ray, axis=-1)
        proj = lambda p: np.stack([K["fx"] * p[..., 0] / p[..., 2] + K["cx"], K["fy"] * p[..., 1] / p[..., 2] + K["cy"]], -1)
        px = np.linalg.norm(proj(ec) - proj(gc), axis=-1)
        cm = lambda x: round(float(np.median(x)) * 100, 1)
        report[s] = dict(frames=int(ok.sum()), of=len(fi), wrist_cm=cm(err[:, 0]), mpjpe_cm=cm(err.mean(1)),
                         root_relative_mpjpe_cm=cm(rel.mean(1)), pa_mpjpe_cm=cm(pa),
                         depth_error_cm=cm(np.abs(along).mean(1)), lateral_error_cm=cm(across.mean(1)),
                         depth_bias_cm=round(float(np.median(along)) * 100, 1), reprojection_px=round(float(np.median(px)), 1))
        series[s] = (t[ok], err[:, 0] * 100, pa * 100)
        r = report[s]
        print(f"{s:5s}: {r['frames']}/{r['of']} frames | wrist {r['wrist_cm']} cm, MPJPE {r['mpjpe_cm']} cm, "
              f"root-relative {r['root_relative_mpjpe_cm']} cm, PA {r['pa_mpjpe_cm']} cm | depth {r['depth_error_cm']} cm "
              f"(bias {r['depth_bias_cm']:+}), lateral {r['lateral_error_cm']} cm, reprojection {r['reprojection_px']} px")
    (root / "hawor" / "compare_mocap.json").write_text(json.dumps(report, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for s, (tt, w, pa) in series.items():
        axes[0].plot(tt, w, ".", ms=2, label=s)
        axes[1].plot(tt, pa, ".", ms=2, label=s)
    axes[0].set_ylabel("wrist error (cm)")
    axes[1].set_ylabel("PA-MPJPE (cm)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"HaWoR vs the hand mocap, {meta['key']}")
    fig.tight_layout()
    fig.savefig(root / "hawor" / "compare_mocap.png", dpi=110)
    print(f"-> {root / 'hawor' / 'compare_mocap.json'}, compare_mocap.png")


if __name__ == "__main__":
    main()
