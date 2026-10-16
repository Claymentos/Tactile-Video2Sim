#!/usr/bin/env python3
"""Perception: the camera motion and a world frame for videos without a tracked head (TouchScale).

The world is the table: origin at the centre of the support plane seen at the first frame,
z along the plane normal (the plane is fitted on the support's SAM3 mask lifted with the depth), x along
the first camera's x projected on the plane. The camera then moves with the head; objects that stay put
give its motion: for a fixed object o, T_world_o is constant, so T_world_cam(t) = T_world_o inv(T_cam_o(t)).

Fixed objects: those no hand manipulates in the scene's clips (an object put down onto, like a block a
dowel goes into, counts as fixed), that FoundationPose tracked (track_objects_foundationpose.py), and that
keep their pose relative to the other candidates (two fixed objects do not move relative to each other,
whatever the camera does): candidates whose relative pose drifts by more than --max-drift-cm or
--max-drift-deg (median over the frames) are dropped. The camera pose of a frame is the average of the
fixed objects' estimates (rotations averaged by their quaternions).

Writes <perception>/camera.npz with the keys register_camera.py writes (T_world_cam (T, 4, 4), table_normal,
table_point, table_extent, depth_scale = 1) plus fixed_objects, and sets frames.json's world to "table".

    python scripts/reconstruction/estimate_camera_motion.py --perception data/touchscale/perception/<episode_id>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_objects_sam3 import segmentation_masks  # noqa: E402


def relative_drift(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """Median drift (cm, deg) of b's pose relative to a over the frames both are tracked."""
    ok = np.isfinite(Ta).all((1, 2)) & np.isfinite(Tb).all((1, 2))
    rel = np.linalg.inv(Ta[ok]) @ Tb[ok]
    ref = rel[len(rel) // 2]
    dt = np.linalg.norm(rel[:, :3, 3] - ref[:3, 3], axis=1) * 100
    dr = np.degrees(Rotation.from_matrix(np.linalg.inv(ref[:3, :3]) @ rel[:, :3, :3]).magnitude())
    return float(np.median(dt)), float(np.median(dr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--support", default="auto", help="the support surface's object id (the world plane); "
                                                      "auto: tv2s_data.scene.support_id")
    ap.add_argument("--max-drift-cm", type=float, default=2.0)
    ap.add_argument("--max-drift-deg", type=float, default=5.0)
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    scene = json.loads(Path(meta["scene"]).read_text())
    if args.support == "auto":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from tv2s_data.scene import support_id

        args.support = support_id(scene)
    K, (W, H), T = meta["intrinsics"], meta["size"], len(meta["frame_index"])

    moved = {c[f"{s}_hand"]["object"] for c in scene["clips"] for s in ("left", "right")}
    tracks = {o["id"]: np.load(root / "foundationpose" / f"{o['id']}.npz")["T_cam_obj"] for o in scene["objects"]
              if o["id"] not in moved and o["id"] != args.support and (root / "foundationpose" / f"{o['id']}.npz").exists()}
    if not tracks:
        raise SystemExit("no tracked object that no hand moves: nothing to anchor the camera on")
    fixed = []
    for a in tracks:                                       # keep the objects consistent with the others
        drifts = [relative_drift(tracks[a], tracks[b]) for b in tracks if b != a]
        ok = not drifts or np.median([d[0] for d in drifts]) <= args.max_drift_cm and \
            np.median([d[1] for d in drifts]) <= args.max_drift_deg
        print(f"  {a:24s} drift vs the others: " + (", ".join(f"{d[0]:.1f} cm / {d[1]:.1f} deg" for d in drifts) or "-")
              + ("" if ok else "  -> not fixed"))
        if ok:
            fixed.append(a)
    if not fixed:
        raise SystemExit("the candidates disagree: no fixed object")

    # world frame: the support plane at the first frame, in the camera
    sup = segmentation_masks(root / "sam3" / f"{args.support}.npz")[0]
    d = cv2.imread(str(root / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED) / 1000.0 * float(meta.get("depth_scale", 1.0))
    ys, xs = np.nonzero(sup & (d > 0))
    pts = np.stack([(xs - K["cx"]) / K["fx"] * d[ys, xs], (ys - K["cy"]) / K["fy"] * d[ys, xs], d[ys, xs]], -1)[::20]
    rng, best = np.random.default_rng(0), (0, None)
    for _ in range(300):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - a) @ n) < 0.01
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    plane = pts[best[1]]
    centre = plane.mean(0)
    n = np.linalg.svd(plane - centre, full_matrices=False)[2][2]
    n = n * -np.sign(n[1])                                 # up: towards the camera's -y
    origin = centre                                        # the centre of the support seen at the first frame
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    R_cw = np.stack([x, np.cross(n, x), n], 1)             # world axes in the first camera
    T_cam0_world = np.eye(4)
    T_cam0_world[:3, :3], T_cam0_world[:3, 3] = R_cw, origin
    T_world_cam0 = np.linalg.inv(T_cam0_world)

    # every frame: each fixed object's estimate of the camera, averaged
    Twc = np.full((T, 4, 4), np.nan)
    fixed = [o for o in fixed if np.isfinite(tracks[o][0]).all()]     # placed in the world at the first frame
    world_obj = {o: T_world_cam0 @ tracks[o][0] for o in fixed}
    for t in range(T):
        est = [world_obj[o] @ np.linalg.inv(tracks[o][t]) for o in fixed if np.isfinite(tracks[o][t]).all()]
        if not est:
            continue
        Twc[t] = np.eye(4)
        Twc[t, :3, :3] = Rotation.from_matrix(np.stack([e[:3, :3] for e in est])).mean().as_matrix()
        Twc[t, :3, 3] = np.mean([e[:3, 3] for e in est], 0)
    Twc[0] = T_world_cam0
    path = np.nansum(np.linalg.norm(np.diff(Twc[:, :3, 3], axis=0), axis=1))
    # table extent in the world, from the first frame's plane points
    wp = plane @ T_world_cam0[:3, :3].T + T_world_cam0[:3, 3]
    extent = np.array([np.percentile(wp, 1, 0), np.percentile(wp, 99, 0)])
    np.savez(root / "camera.npz", T_world_cam=Twc, depth_scale=float(meta.get("depth_scale", 1.0)),
             table_normal=np.array([0.0, 0, 1]), table_point=np.zeros(3), table_extent=extent,
             fixed_objects=np.array(fixed))
    (root / "frames.json").write_text(json.dumps(dict(meta, world="table"), indent=1))
    print(f"camera from {fixed}: {np.isfinite(Twc).all((1, 2)).sum()}/{T} frames, head path {path * 100:.0f} cm, "
          f"camera {Twc[0, 2, 3]:.2f} m above the table at the first frame -> {root / 'camera.npz'}")


if __name__ == "__main__":
    main()
