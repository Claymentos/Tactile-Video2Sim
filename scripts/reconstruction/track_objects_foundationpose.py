#!/usr/bin/env python3
"""Perception: 6-DoF trajectory of every reconstructed object with FoundationPose (the reference for the RL).

Per object of objects.json (reconstruct_objects_recgen.py): its RecGen mesh, simplified (FoundationPose
renders it at every step) and centred, is its model; the object frame is the reference frame's camera
axes at the mesh centre, so the pose at the reference frame is known exactly from RecGen. Tracking starts
there and runs forwards to the end of the video, then backwards to its start, on the RGB-D frames (depth
in metres: the sensor's, or MoGe's rescaled by register_camera.py). After each step the tracked mesh is
checked against the object's SAM3 mask where that mask is reliable (at least --min-mask-ratio of the
reference frame's area): when the projected model falls mostly outside it (a fast motion, a jump),
FoundationPose re-registers the object from that mask.

Writes <perception>/foundationpose/<object>.npz: T_cam_obj (T, 4, 4) per sampled frame, mask_agreement
(T,) fraction of the projected model points inside the SAM3 mask (NaN without a mask), reregistered
(frame indices), and T_world_obj (T, 4, 4) when the camera is registered (camera.npz); the model,
recgen/<object>/object_mesh.obj (also the simulation's mesh); overlay.mp4 (the tracked models over the frames).

Run in the `foundationpose` env (FOUNDATIONPOSE_DIR: the checkout, default ~/FoundationPose).
    python scripts/reconstruction/track_objects_foundationpose.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_objects_sam3 import COLORS, segmentation_masks  # noqa: E402

FP_DIR = Path(os.environ.get("FOUNDATIONPOSE_DIR", Path.home() / "FoundationPose"))


def simplified_model(path: Path, faces: int):
    """RecGen's posed mesh (camera frame of its reference frame), decimated, vertex colours kept, and its
    centre: the model is centred there."""
    import open3d as o3d
    import trimesh

    m = trimesh.load(path, force="mesh")
    vis = m.visual.to_color() if hasattr(m.visual, "to_color") else m.visual    # texture -> vertex colours
    colors = np.asarray(vis.vertex_colors)[:, :3] / 255.0
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    o.vertex_colors = o3d.utility.Vector3dVector(colors)
    if len(m.faces) > faces:
        o = o.simplify_quadric_decimation(target_number_of_triangles=faces)
    o.remove_unreferenced_vertices()
    v = np.asarray(o.vertices)
    centre = np.median(v, 0)
    out = trimesh.Trimesh(v - centre, np.asarray(o.triangles), vertex_colors=(np.asarray(o.vertex_colors) * 255).astype(np.uint8),
                          process=False)
    return out, centre


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--objects", default=None, help="comma list of object ids (default: every reconstructed one)")
    ap.add_argument("--faces", type=int, default=20000, help="model triangles after decimation")
    ap.add_argument("--register-iter", type=int, default=5)
    ap.add_argument("--track-iter", type=int, default=2)
    ap.add_argument("--models-only", action="store_true",
                    help="only write the models (object_mesh.obj, model_centre in objects.json), for the CoTracker tracks")
    ap.add_argument("--min-mask-ratio", type=float, default=0.5,
                    help="a mask smaller than this share of the reference frame's is not trusted")
    ap.add_argument("--min-agreement", type=float, default=0.3,
                    help="re-register when fewer projected model points fall inside a visible SAM3 mask")
    args = ap.parse_args()
    root = Path(args.perception).resolve()
    meta = json.loads((root / "frames.json").read_text())
    objects = json.loads((root / "objects.json").read_text())
    if args.objects:
        objects = {k: v for k, v in objects.items() if k in args.objects.split(",")}
    K = np.array([[meta["intrinsics"]["fx"], 0, meta["intrinsics"]["cx"]], [0, meta["intrinsics"]["fy"], meta["intrinsics"]["cy"]],
                  [0, 0, 1.0]])
    T, scale = len(meta["frame_index"]), float(meta.get("depth_scale", 1.0))
    cam = np.load(root / "camera.npz")["T_world_cam"] if (root / "camera.npz").exists() else None
    if args.models_only:
        all_objects = json.loads((root / "objects.json").read_text())
        for oid, info in objects.items():
            model, centre = simplified_model(Path(info["mesh"]), args.faces)
            model.export(Path(info["mesh"]).parent / "object_mesh.obj")
            all_objects[oid]["model_centre"] = centre.tolist()
            print(f"  {oid:24s} model: {len(model.faces)} faces, centre {np.round(centre, 3).tolist()}")
        (root / "objects.json").write_text(json.dumps(all_objects, indent=1))
        return
    rgbs = [cv2.cvtColor(cv2.imread(str(root / "frames" / f"{t:06d}.jpg")), cv2.COLOR_BGR2RGB) for t in range(T)]
    depths = [cv2.imread(str(root / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 * scale
              for t in range(T)]
    out = root / "foundationpose"
    out.mkdir(exist_ok=True)

    sys.path.insert(0, str(FP_DIR))
    cwd = os.getcwd()
    os.chdir(FP_DIR)
    import nvdiffrast.torch as dr
    import torch
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    os.chdir(cwd)
    logging.getLogger().setLevel(logging.WARNING)
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()

    tracks = {}
    for oid, info in objects.items():
        model, centre = simplified_model(Path(info["mesh"]), args.faces)
        model.export(Path(info["mesh"]).parent / "object_mesh.obj")
        all_objects = json.loads((root / "objects.json").read_text())
        all_objects[oid]["model_centre"] = centre.tolist()
        (root / "objects.json").write_text(json.dumps(all_objects, indent=1))
        masks = segmentation_masks(root / "sam3" / f"{oid}.npz")
        est = FoundationPose(model_pts=model.vertices, model_normals=model.vertex_normals, mesh=model, scorer=scorer,
                             refiner=refiner, glctx=glctx, debug=0, debug_dir=str(out / "debug"))
        to_centered = est.get_tf_to_centered_mesh().cpu().numpy()
        pts = model.vertices[np.random.default_rng(0).choice(len(model.vertices), min(500, len(model.vertices)), replace=False)]
        poses, agree, rereg = np.full((T, 4, 4), np.nan), np.full(T, np.nan), []
        t_ref = int(info["reference_frame"])
        P_ref = np.eye(4)
        P_ref[:3, 3] = centre                                   # exact at the reference frame (RecGen's frame)

        ref_area = max(int(masks[t_ref].sum()), 1)

        def agreement(P, t):
            """Fraction of the projected model inside the SAM3 mask, only where the mask is reliable: the
            object at least half as visible as on the reference frame (a held object's mask shrinks to a
            few fingertip pixels, a thin one's slides onto the glove)."""
            m = masks[t]
            if m.sum() < max(100, args.min_mask_ratio * ref_area):
                return np.nan
            q = pts @ P[:3, :3].T + P[:3, 3]
            ok = q[:, 2] > 0.05
            uv = (q[ok, :2] / q[ok, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
            inside = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
            return float(m[v[inside], u[inside]].sum() / max(len(pts), 1))

        for direction in (1, -1):                               # forwards from the reference, then backwards
            est.pose_last = torch.as_tensor((P_ref @ np.linalg.inv(to_centered))[None], device="cuda", dtype=torch.float)
            poses[t_ref], agree[t_ref] = P_ref, agreement(P_ref, t_ref)
            for t in range(t_ref + direction, T if direction > 0 else -1, direction):
                P = est.track_one(rgb=rgbs[t], depth=depths[t], K=K, iteration=args.track_iter)
                a = agreement(P, t)
                if np.isfinite(a) and a < args.min_agreement:
                    P = est.register(K=K, rgb=rgbs[t], depth=depths[t], ob_mask=masks[t], iteration=args.register_iter)
                    a = agreement(P, t)
                    rereg.append(t)
                poses[t], agree[t] = P, a
        tracks[oid] = (poses, pts)
        res = dict(T_cam_obj=poses, mask_agreement=agree, reregistered=np.array(sorted(set(rereg))), centre_ref=centre)
        if cam is not None:
            res["T_world_obj"] = cam @ poses
        np.savez(out / f"{oid}.npz", **res)
        path = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
        print(f"  {oid:24s} mask agreement median {np.nanmedian(agree):.2f}, {len(set(rereg))} re-registrations, "
              f"path {np.nansum(path) * 100:.0f} cm in the camera", flush=True)

    writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["sampled_fps"], tuple(meta["size"]))
    for t in range(T):
        img = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2BGR)
        for k, (oid, (poses, pts)) in enumerate(tracks.items()):
            if not np.isfinite(poses[t]).all():
                continue
            q = pts @ poses[t][:3, :3].T + poses[t][:3, 3]
            q = q[q[:, 2] > 0.05]
            uv = (q[:, :2] / q[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            color = COLORS[k % len(COLORS)][::-1]
            for u, v in uv.astype(int):
                cv2.circle(img, (u, v), 1, color, -1)
            cv2.putText(img, oid, tuple(uv.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"{len(tracks)} objects tracked -> {out}")


if __name__ == "__main__":
    main()
