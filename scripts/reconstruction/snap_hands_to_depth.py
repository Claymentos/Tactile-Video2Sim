#!/usr/bin/env python3
"""Put HaWoR's hands at the depth the depth map sees them (estimate_hands_hawor.py output, corrected in place).

A monocular hand estimate is least sure along the viewing ray. Where the episode has a metric depth map
(TouchScale's sensor, or MoGe rescaled by register_camera.py), each hand is moved along the camera rays so
that its depth matches the depth under its SAM3 mask: per sampled frame the ratio of the depth map's median
under the mask to the median depth of HaWoR's vertices that project inside the mask (the visible side,
like the depth map's), median-filtered over --smooth-s and interpolated to HaWoR's frames; the hand's joints
and vertices are scaled about the camera centre by it (which keeps their projection). The joints in the world
are recomputed. The raw estimate is kept as <side>_joints_cam_raw.

Not used by default: on TouchScale 29029fe2 it made the right hand worse (its lowest keypoint went from
1.4 cm to 7.6 cm above the table, though it picks a dowel from a 2 cm tray): the SAM3 'hand' mask also
covers the wrist strap and the forearm, nearer the camera than the hand.

    python scripts/reconstruction/snap_hands_to_depth.py --perception data/touchscale/perception/<episode_id>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_hands_hawor import interp_poses  # noqa: E402
from segment_objects_sam3 import segmentation_masks  # noqa: E402

SIDES = ("left", "right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--smooth-s", type=float, default=1.0)
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    K, (W, H) = meta["intrinsics"], meta["size"]
    scale_depth = float(meta.get("depth_scale", 1.0))
    hw = dict(np.load(root / "hawor" / "hands.npz"))
    tracks = segmentation_masks(root / "sam3" / "hands.npz")
    side_of = {}
    for n, tr in enumerate(tracks):
        xs = [np.nonzero(m.any(0))[0].mean() for m in tr if m.any()]
        if xs:
            side_of.setdefault("left" if np.mean(xs) < W / 2 else "right", []).append(n)
    fi_s, times_s = np.asarray(meta["frame_index"]), np.asarray(meta["times_s"])
    fi_h = hw["frame_index"]
    t_h = fi_h / meta["video_fps"]
    for s in SIDES:
        ratio = np.full(len(fi_s), np.nan)
        for k in range(len(fi_s)):
            m = np.zeros((H, W), bool)
            for n in side_of.get(s, []):
                m |= tracks[n, k]
            d = cv2.imread(str(root / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED) / 1000.0 * scale_depth
            ok = m & (d > 0)
            if ok.sum() < 300:
                continue
            j = int(np.argmin(np.abs(fi_h - fi_s[k])))
            v = hw[f"{s}_vertices_cam"][j]
            u = (K["fx"] * v[:, 0] / v[:, 2] + K["cx"]).astype(int)
            w = (K["fy"] * v[:, 1] / v[:, 2] + K["cy"]).astype(int)
            inside = (u >= 0) & (u < W) & (w >= 0) & (w < H)
            inside[inside] &= m[w[inside], u[inside]]
            if inside.sum() < 50:
                continue
            ratio[k] = np.median(d[ok]) / np.median(v[inside, 2])
        good = np.isfinite(ratio)
        if good.sum() < 3:
            print(f"{s}: not enough depth under the hand, unchanged")
            continue
        r = np.interp(times_s, times_s[good], ratio[good])
        r = median_filter(r, size=max(1, int(args.smooth_s * meta["sampled_fps"])) | 1, mode="nearest")
        r_h = np.interp(t_h, times_s, r)
        hw.setdefault(f"{s}_joints_cam_raw", hw[f"{s}_joints_cam"].copy())
        hw[f"{s}_joints_cam"] = hw[f"{s}_joints_cam_raw"] * r_h[:, None, None]
        hw[f"{s}_vertices_cam"] = hw[f"{s}_vertices_cam"] * (r_h / hw.get(f"{s}_snap_ratio", np.ones_like(r_h)))[:, None, None]
        hw[f"{s}_snap_ratio"] = r_h
        print(f"{s}: depth ratio median {np.median(r_h):.3f} (p10 {np.percentile(r_h, 10):.3f}, p90 {np.percentile(r_h, 90):.3f}), "
              f"mean shift {np.median((r_h - 1) * hw[f'{s}_joints_cam_raw'][:, 0, 2]) * 100:+.1f} cm along the rays")
    if (root / "camera.npz").exists():
        Twc = interp_poses(np.load(root / "camera.npz")["T_world_cam"], times_s, t_h)
        for s in SIDES:
            hw[f"{s}_joints_world"] = np.einsum("tij,tkj->tki", Twc[:, :3, :3], hw[f"{s}_joints_cam"]) + Twc[:, None, :3, 3]
    np.savez_compressed(root / "hawor" / "hands.npz", **hw)
    print(f"-> {root / 'hawor' / 'hands.npz'}")


if __name__ == "__main__":
    main()
