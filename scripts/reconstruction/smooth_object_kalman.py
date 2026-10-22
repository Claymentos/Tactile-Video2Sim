#!/usr/bin/env python3
"""Kalman-smooth an object trajectory in the reference camera, once the camera's motion is composed out of it
(compose_camera_motion.py): a constant-velocity model describes how the object moves -- resting, lifted, set down --
not how the head does.

What is filtered is the object's centre p = R c + t, not the pose translation, which any rotation of a mesh posed
~1 m from the camera swings by tens of centimetres; the rotation is filtered on the manifold, and the pose rebuilt
as t = p - R c (tv2s_data.rigid_poses.kalman_smooth_poses: per-frame noise from the RANSAC inlier count, an
innovation gate, an RTS backward pass).

Run in the `genesis` env:
    python scripts/reconstruction/smooth_object_kalman.py --traj hammer_traj_composed_cotracker.npz \\
        --mesh recgen_out_moge/hammer/posed_mesh.obj --out hammer_traj_kalman_cotracker.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tv2s_data.rigid_poses import kalman_smooth_poses  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Kalman-smooth an object trajectory (reference camera)")
    parser.add_argument("--traj", required=True, help="Composed trajectory (compose_camera_motion.py)")
    parser.add_argument("--mesh", required=True, help="Mesh the trajectory moves, reference camera")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pos-noise", type=float, default=0.010,
                        help="Centre measurement noise, metres (1 sigma)")
    parser.add_argument("--accel", type=float, default=0.005,
                        help="Process noise: unmodelled acceleration, metres/frame^2")
    # Tuned on TouchAnything pick_up_screwdriver: 4 deg / 2 left the rotation shaking
    # (1.39 deg/frame^2 of rate change); 8 / 1 brings it to 0.41 at a mask IoU of 0.333
    # against 0.345; 12 / 0.5 and beyond start dragging the pose off the image.
    parser.add_argument("--rot-noise", type=float, default=8.0,
                        help="Orientation measurement noise, degrees (1 sigma)")
    parser.add_argument("--ang-accel", type=float, default=1.0,
                        help="Process noise: unmodelled angular acceleration, degrees/frame^2")
    parser.add_argument("--gate", type=float, default=6.0,
                        help="Innovation gate, Mahalanobis distance")
    return parser.parse_args()


def roughness(centres, rotations, valid):
    """Median frame-to-frame acceleration of the centre (mm/frame^2) and turn rate change (deg/frame^2)."""
    from scipy.spatial.transform import Rotation

    run = valid[:-2] & valid[1:-1] & valid[2:]
    accel = np.linalg.norm(centres[2:] - 2 * centres[1:-1] + centres[:-2], axis=1)[run]
    r = Rotation.from_matrix(rotations)
    step = (r[:-1].inv() * r[1:]).as_rotvec()
    spin = np.degrees(np.linalg.norm(step[1:] - step[:-1], axis=1))[run]
    return 1000 * np.median(accel), np.median(spin)


def main():
    args = parse_args()
    track = dict(np.load(args.traj))
    rotation, translation = track["rotation"], track["translation"]
    valid = track["valid"].astype(bool)
    confidence = track["inliers"].astype(float) if "inliers" in track else None

    centre = np.asarray(trimesh.load(args.mesh, process=False).vertices).mean(axis=0)
    centres = np.einsum("tij,j->ti", rotation, centre) + translation

    before = roughness(centres, rotation, valid)
    smooth_rotation, smooth_centres, smooth_valid = kalman_smooth_poses(
        rotation, centres, valid,
        position_sigma=args.pos_noise,
        accel_sigma=args.accel,
        rotation_sigma=args.rot_noise,
        angular_accel_sigma=args.ang_accel,
        gate=args.gate,
        confidence=confidence,
    )
    after = roughness(smooth_centres, smooth_rotation, smooth_valid)
    print(f"[kalman] {valid.sum()} measured frames -> {smooth_valid.sum()} smoothed")
    print(f"[kalman] centre acceleration median {before[0]:.1f} -> {after[0]:.1f} mm/frame^2, "
          f"rotation-rate change median {before[1]:.2f} -> {after[1]:.2f} deg/frame^2")

    track["rotation"] = smooth_rotation
    track["translation"] = smooth_centres - np.einsum("tij,j->ti", smooth_rotation, centre)
    track["valid"] = smooth_valid
    track["measured"] = valid
    np.savez(args.out, **track)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
