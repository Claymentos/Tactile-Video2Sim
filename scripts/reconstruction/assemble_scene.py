#!/usr/bin/env python3
"""The simulation scene of an episode: the table as the ground, every reconstructed object with its trajectory.

Gathers the perception of an episode (perception/<key>/: camera.npz from register_camera.py or
estimate_camera_motion.py, objects.json and recgen/<id>/object_mesh.obj from the RecGen stage, and the
object tracks: cotracker/<id>.npz (Benchmark-Genesis' method, default) or foundationpose/<id>.npz) into one frame, the simulation's: the support plane is
z = 0 with z up (the world frame rotated so the fitted plane is exactly horizontal, then shifted),
x / y those of the world projected on the plane.

Per object: the mesh (metres, centred), whether it is fixed (no hand manipulates it in the scene's clips
and it stays within --fixed-cm of its median position: then its median pose is used for the whole
episode), a mass (the mesh's volume -- its convex hull's when the mesh is not watertight, at least
--box-fill of its bounding box since RecGen meshes are open shells -- times a density from the scene's
material label), and its trajectory, a pose per sampled frame: T_sim_obj =
T_sim_world T_world_cam(t) T_cam_obj(t).

Writes <perception>/sim_scene.json (fps, frame_index, T_sim_world, table extent, objects) and
sim_trajectories.npz (<id>: (T, 4, 4)); the hands (HumanTouch's mocap, retargeted) go through the same
T_sim_world.

    python scripts/reconstruction/assemble_scene.py --perception data/touchscale/perception/<episode_id>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

DENSITY = {"wood": 600, "plastic": 950, "metal": 7800, "aluminum": 2700, "steel": 7800, "glass": 2500,
           "ceramic": 2300, "porcelain": 2400, "paper": 700, "cardboard": 300, "rubber": 1100, "fabric": 300,
           "foam": 50}                                       # kg/m3; unknown: water's


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--trajectories", choices=("cotracker", "foundationpose"), default="cotracker",
                    help="object tracks: Benchmark-Genesis' CoTracker + depth, or FoundationPose")
    ap.add_argument("--box-fill", type=float, default=0.5, help="volume at least this share of the bounding box")
    ap.add_argument("--fixed-cm", type=float, default=3.0, help="an unmanipulated object moving less is fixed")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    scene = json.loads(Path(meta["scene"]).read_text())
    cam = np.load(root / "camera.npz")
    objects = json.loads((root / "objects.json").read_text())
    labels = {o["id"]: o for o in scene["objects"]}
    moved = {c[f"{s}_hand"]["object"] for c in scene["clips"] for s in ("left", "right")}

    # simulation frame: the support plane at z = 0
    n, p0 = cam["table_normal"], cam["table_point"]
    R = Rotation.align_vectors([[0, 0, 1.0]], [n])[0].as_matrix()  # world -> sim rotation, plane normal -> z
    T_sim_world = np.eye(4)
    T_sim_world[:3, :3] = R
    T_sim_world[:3, 3] = -R @ p0
    Twc = cam["T_world_cam"]
    T = len(meta["frame_index"])

    out_objects, trajectories = [], {}
    for oid, info in objects.items():
        fp = root / args.trajectories / f"{oid}.npz"
        mesh_path = Path(info["mesh"]).parent / "object_mesh.obj"
        if not fp.exists() or not mesh_path.exists():
            print(f"  {oid:24s} no trajectory / model, skipped")
            continue
        poses = T_sim_world @ Twc @ np.load(fp)["T_cam_obj"]                  # (T, 4, 4)
        ok = np.isfinite(poses).all((1, 2))
        pos = poses[ok, :3, 3]
        spread = float(np.percentile(np.linalg.norm(pos - np.median(pos, 0), axis=1), 90) * 100)
        fixed = oid not in moved and spread < args.fixed_cm
        if fixed:                                                            # one pose: the median one
            k = np.argmin(np.linalg.norm(pos - np.median(pos, 0), axis=1))
            poses[:] = poses[np.flatnonzero(ok)[k]]
        mesh = trimesh.load(mesh_path, force="mesh")
        material = labels.get(oid, {}).get("material", "unknown")
        if labels.get(oid, {}).get("mass_g"):                               # Gemini's estimate, when asked for
            mass = float(labels[oid]["mass_g"]) / 1000
        else:
            # RecGen gives open shells (the seen side): their hull can be nearly flat, so the volume is at least
            # a share of the robust box; containers and metal objects (cans) are hollow: a small share
            box = float(np.prod(info["extent_cm"])) * 1e-6                 # robust extent (percentiles 1-99), m3
            hollow = labels.get(oid, {}).get("is_container") or material in ("metal", "aluminum", "steel")
            fill = (0.03 if labels.get(oid, {}).get("is_container") else 0.1) if hollow else args.box_fill
            volume = max(mesh.volume if mesh.is_watertight else mesh.convex_hull.volume, fill * box)
            mass = float(np.clip(volume * DENSITY.get(material, 1000), 0.005, 2.0))
        low = min(float((mesh.vertices @ p[:3, :3].T + p[:3, 3])[:, 2].min()) for p in poses[ok][:: max(1, ok.sum() // 20)])
        trajectories[oid] = poses
        out_objects.append(dict(id=oid, mesh=str(mesh_path.resolve()), fixed=bool(fixed), mass=round(mass, 4),
                                material=material, description=labels.get(oid, {}).get("description", ""),
                                spread_cm=round(spread, 1), manipulated=oid in moved, lowest_point_cm=round(low * 100, 1)))
        print(f"  {oid:24s} {'fixed' if fixed else 'moving'} (90% within {spread:.1f} cm), {mass * 1000:.0f} g "
              f"({material}), lowest point {low * 100:+.1f} cm", flush=True)

    ext = (np.c_[cam["table_extent"], np.ones(2)] @ T_sim_world.T)[:, :3] if "table_extent" in cam.files else None
    scene_out = dict(key=meta["key"], dataset=meta["dataset"], fps=meta["sampled_fps"], frame_index=meta["frame_index"],
                     trajectories=args.trajectories,
                     video=meta["video"], T_sim_world=T_sim_world.tolist(),
                     table=dict(height=0.0, extent=None if ext is None else ext.round(3).tolist()), objects=out_objects)
    (root / "sim_scene.json").write_text(json.dumps(scene_out, indent=1))
    np.savez(root / "sim_trajectories.npz", **trajectories)
    print(f"{len(out_objects)} objects -> {root / 'sim_scene.json'}, {root / 'sim_trajectories.npz'}")


if __name__ == "__main__":
    main()
