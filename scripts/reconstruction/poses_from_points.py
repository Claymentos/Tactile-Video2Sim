#!/usr/bin/env python3
"""A rigid trajectory from CoTracker's lifted points (track_points_cotracker.py): the camera's motion when the points
are on the table, the object's (in the camera of each frame) when they are on the object.

The pose of frame t is the rigid motion carrying the reference frame's cloud onto frame t's, solved by Kabsch with
RANSAC (a point whose depth patch caught the background, or that slid off the object during an occlusion, is an
outlier in 3D even when its 2D track looks fine). Solves that cannot follow from the previous one are dropped
(tv2s_data.rigid_poses.gate_poses), then:
  --smoother window   the gaps interpolated and the poses smoothed on a sliding window (the camera)
  --smoother none     the gated solves only, gaps left: the object, Kalman-smoothed later (smooth_object_kalman.py)
                      once the camera's motion is composed out (compose_camera_motion.py)

Writes --out (.npz): rotation (T, 3, 3), translation (T, 3) (X_t = R X_ref + t), valid, inliers, ref_frame.

Run in the `genesis` env:
    python scripts/reconstruction/poses_from_points.py --points table_points_cotracker.npz --out camera_traj_cotracker.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tv2s_data.rigid_poses import clean_poses, gate_poses, kabsch_ransac  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True, help="track_points_cotracker.py output")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-points", type=int, default=12,
                        help="points a frame needs, lifted and shared with the reference frame, to be solved")
    parser.add_argument("--ransac-iters", type=int, default=200)
    parser.add_argument("--ransac-thresh", type=float, default=0.012, help="inlier distance, metres")
    parser.add_argument("--max-step", type=float, default=0.06, help="largest plausible travel between frames, metres")
    parser.add_argument("--max-turn", type=float, default=20.0, help="largest plausible rotation between frames, degrees")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smoother", choices=("window", "none"), default="window")
    args = parser.parse_args()

    data = np.load(args.points)
    points_3d = data["points_3d"]
    reference = int(data["ref_frame"])
    n_frames = len(points_3d)
    source_all = points_3d[reference]
    seeded = np.isfinite(source_all).all(axis=1)
    if seeded.sum() < args.min_points:
        raise SystemExit(f"only {seeded.sum()} points lifted at the reference frame")

    rotations = np.tile(np.eye(3), (n_frames, 1, 1))
    translations = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)
    inliers = np.zeros(n_frames, dtype=int)
    rng = np.random.default_rng(0)
    for index in range(n_frames):
        usable = seeded & np.isfinite(points_3d[index]).all(axis=1)
        if usable.sum() < args.min_points:
            continue
        rotation, translation, mask = kabsch_ransac(source_all[usable], points_3d[index][usable], args.ransac_iters,
                                                    args.ransac_thresh, rng)
        if rotation is None:
            continue
        rotations[index], translations[index] = rotation, translation
        inliers[index] = int(mask.sum())
        valid[index] = True
    print(f"[pose] solved {valid.sum()}/{n_frames} frames from {seeded.sum()} reference points, "
          f"median {np.median(inliers[valid]) if valid.any() else 0:.0f} inliers")

    # continuity on the tracked points' centroid, not on the translation (see gate_poses)
    centre = source_all[seeded].mean(axis=0)
    solved = int(valid.sum())
    if args.smoother == "none":
        valid = gate_poses(rotations, translations, valid, args.max_step, args.max_turn, centre=centre)
        print(f"[pose] continuity gate kept {valid.sum()}/{solved} solves")
    else:
        kept = gate_poses(rotations, translations, valid, args.max_step, args.max_turn, centre=centre)
        print(f"[pose] continuity gate kept {kept.sum()}/{solved} solves; the rest are interpolated")
        rotations, translations, valid = clean_poses(rotations, translations, valid, args.max_step, args.max_turn,
                                                     args.smooth_window, confidence=inliers.astype(float), centre=centre)
    print(f"[pose] {valid.sum()}/{n_frames} frames after dropping implausible jumps and smoothing")
    np.savez(args.out, rotation=rotations, translation=translations, valid=valid, inliers=inliers,
             resting=np.zeros(n_frames, dtype=bool), ref_frame=reference)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
