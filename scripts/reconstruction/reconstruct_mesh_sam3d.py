#!/usr/bin/env python3
"""One object mesh from the reference frame with SAM 3D Objects (github.com/facebookresearch/sam-3d-objects), in place of
RecGen (launch/reconstruct.sh MESH_MODEL=sam3d): the same outputs, in the reference camera (OpenCV) in metres. Its
layout post-optimisation (the pose and scale refined against the mask and the point map) is on: without it the posed
mesh covered the SAM3 mask with an IoU of 0.04-0.65 on 8 EgoTouch objects (RecGen's: 0.69-0.86), so the mesh's pose
and scale are then registered to the frame (register_to_observation): SAM 3D's shape, the frame's pose.

SAM 3D Objects takes an RGB image, the object's mask and a point map, and returns the object's shape in a canonical
frame (a mesh, y-up once exported) and its layout in the camera: rotation (quaternion w x y z), translation and
scale, local -> camera in PyTorch3D's camera convention (x left, y up, z forward), applied as row vectors
(p_cam = s p_local R + t). Its own point map comes from a monocular depth model; here it is the reconstruction's
depth (DA3 / MoGe, rescaled to the hands' metres) lifted with intrinsics.yaml, so that the mesh lands at the
reconstruction's scale and the object trajectory, the table and the hands agree with it.

  <out>/mesh.obj          the canonical mesh (SAM 3D Objects' local frame, z-up), vertex colours
  <out>/posed_mesh.obj    the mesh in the reference camera, metres (what the rest of the pipeline reads)
  <out>/metadata.json     SAM 3D's layout, the registered pose, the extents, the mask IoU of the projection
  <out>/overlay.png       the projected mesh (green) over the frame, the mask's outline (red)

Run in the `sam3d` env (launch/config.sh PY_SAM3D; SAM3D_DIR: the sam-3d-objects checkout with checkpoints/hf):
    python scripts/reconstruction/reconstruct_mesh_sam3d.py --rgb all_frames/000000.png --depth depth_moge/000000.png \\
        --mask video_segmentation/masks/frame_000000_masks/hammer.png --intrinsics intrinsics.yaml \\
        --out sam3d_out/hammer --name hammer
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import yaml

SAM3D_DIR = Path(os.environ.get("SAM3D_DIR", os.path.expanduser("~/sam-3d-objects")))
GLB_TO_LOCAL = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T     # to_glb rotated z-up -> y-up by v @ this.T
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0])                                    # PyTorch3D camera <-> OpenCV camera


def offline_torch_hub() -> None:
    """Compute nodes have no internet: SAM 3D Objects' DINOv2 (torch.hub.load('facebookresearch/dinov2', ...,
    source='github')) from torch.hub's cached checkout instead (fetched once on a login node)."""
    load = torch.hub.load

    def local_load(repo_or_dir, model, *args, source="github", **kwargs):
        cached = Path(torch.hub.get_dir()) / (repo_or_dir.replace("/", "_") + "_main")
        if source == "github" and cached.is_dir():
            return load(str(cached), model, *args, source="local", **kwargs)
        return load(repo_or_dir, model, *args, source=source, **kwargs)

    torch.hub.load = local_load


def point_map(depth_m: np.ndarray, K: dict) -> np.ndarray:
    """(H, W, 3) points in the PyTorch3D camera convention from metric depth; holes filled from their neighbours."""
    d = depth_m.copy()
    hole = d <= 0
    if hole.any():
        d = cv2.inpaint(d.astype(np.float32), hole.astype(np.uint8), 5, cv2.INPAINT_NS)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - K["pu"]) / K["fu"] * d
    Y = (v - K["pv"]) / K["fv"] * d
    return (np.stack([X, Y, d], -1) @ P3D_TO_CV).astype(np.float32)


def project_iou(V: np.ndarray, F: np.ndarray, K: dict, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """IoU of the mesh's silhouette (camera-frame vertices, OpenCV) with the mask, and the silhouette."""
    H, W = mask.shape
    z = np.maximum(V[:, 2], 1e-6)
    uv = np.stack([K["fu"] * V[:, 0] / z + K["pu"], K["fv"] * V[:, 1] / z + K["pv"]], -1)
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, [np.round(uv[f]).astype(np.int32) for f in F], 1)
    sil = sil.astype(bool)
    return float((sil & mask).sum() / max((sil | mask).sum(), 1)), sil


def observed_points(depth_m: np.ndarray, mask: np.ndarray, K: dict, max_points: int = 4000) -> np.ndarray:
    """The object's visible surface (camera, OpenCV, metres): the mask's pixels lifted with the depth, its eroded
    inside only (the depth bleeds into the table at the mask's edge)."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    v, u = np.nonzero(inner & (depth_m > 0))
    if len(u) > max_points:
        keep = np.random.default_rng(0).choice(len(u), max_points, replace=False)
        u, v = u[keep], v[keep]
    z = depth_m[v, u]
    return np.stack([(u - K["pu"]) / K["fu"] * z, (v - K["pv"]) / K["fv"] * z, z], -1)


def register_to_observation(local: np.ndarray, F: np.ndarray, T0: np.ndarray, obs: np.ndarray, K: dict,
                            mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """The mesh's pose (4x4 similarity, local -> camera) that best explains the frame: candidates from SAM 3D
    Objects' own layout and from aligning the mesh's principal axes with the visible points' (the 4 proper sign
    choices), each at SAM 3D's scale and at the scale matching the visible length, refined by ICP of the visible
    points onto the mesh's surface; the one whose silhouette covers the mask best (IoU), ICP residual breaking ties.
    SAM 3D's layout alone covered the SAM3 masks of 8 EgoTouch tools with an IoU of 0.04-0.65 (a screwdriver
    turned 90 degrees): its shape is kept, its pose is not trusted."""
    from trimesh.registration import icp

    s0 = np.cbrt(abs(np.linalg.det(T0[:3, :3])))
    samples = trimesh.Trimesh(local, F, process=False).sample(6000, seed=0)
    Ec = lambda P: np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, ::-1]
    Em, Eo = Ec(local), Ec(obs)
    ext = lambda P, E: np.ptp((P - P.mean(0)) @ E, 0)
    s_len = ext(obs, Eo)[0] / max(ext(local, Em)[0], 1e-9)
    cands = [("sam3d", T0)]
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        D = np.diag([sx, sy, sx * sy * np.linalg.det(Eo) * np.linalg.det(Em)])
        R = Eo @ D @ Em.T
        for name, sc in (("sam3d scale", s0), ("visible length", s_len)):
            T = np.eye(4)
            T[:3, :3] = sc * R
            T[:3, 3] = obs.mean(0) - sc * R @ local.mean(0)
            cands.append((f"axes {sx:+d}{sy:+d}, {name}", T))
    best = None
    for name, T in cands:
        # icp(a, b): the transform a -> b; a = the visible points, b = the mesh's surface at the candidate's scale
        sc = np.cbrt(abs(np.linalg.det(T[:3, :3])))
        Rn = T[:3, :3] / sc
        init = np.eye(4)
        init[:3, :3], init[:3, 3] = Rn.T, -Rn.T @ T[:3, 3]
        M, _, cost = icp(obs, samples * sc, initial=init, threshold=1e-6, max_iterations=60, reflection=False, scale=False)
        Rr = M[:3, :3]
        Tr = np.eye(4)
        Tr[:3, :3], Tr[:3, 3] = sc * Rr.T, -Rr.T @ M[:3, 3]
        V = local @ Tr[:3, :3].T + Tr[:3, 3]
        iou = project_iou(V, F, K, mask)[0]
        if best is None or (iou, -cost) > (best[1]["mask_iou"], -best[1]["icp_cost"]):
            best = (Tr, dict(candidate=name, mask_iou=iou, icp_cost=float(cost), scale=float(sc)))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True, help="uint16 millimetres (depth_moge/<frame>.png)")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--intrinsics", required=True, help="intrinsics.yaml (fu, fv, pu, pv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="object")
    ap.add_argument("--checkpoint", default="hf", help="SAM3D_DIR/checkpoints/<tag>/pipeline.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-register", action="store_true", help="keep SAM 3D Objects' layout as the pose (no "
                    "registration to the visible points and the mask)")
    ap.add_argument("--no-layout-refine", action="store_true", help="skip SAM 3D Objects' layout post-optimisation "
                    "(the pose and scale refined against the mask and the point map; off in its demo, on here)")
    ap.add_argument("--own-depth", action="store_true", help="SAM 3D Objects' own depth model instead of --depth "
                    "(its scale is then not the reconstruction's)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(args.mask, 0) > 128
    K = {k: float(v) for k, v in yaml.safe_load(Path(args.intrinsics).read_text()).items()}
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    pm = None if args.own_depth else torch.from_numpy(point_map(depth, K))

    offline_torch_hub()
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)     # notebook/inference.py sets CUDA_HOME from it (unactivated env)
    sys.path.insert(0, str(SAM3D_DIR / "notebook"))
    sys.path.insert(0, str(SAM3D_DIR))
    from inference import Inference
    from pytorch3d.transforms import quaternion_to_matrix

    inference = Inference(str(SAM3D_DIR / "checkpoints" / args.checkpoint / "pipeline.yaml"), compile=False)
    # notebook/inference.py's Inference.__call__, with the layout post-optimisation it leaves off
    res = inference._pipeline.run(inference.merge_mask_to_rgba(rgb, mask), None, args.seed, stage1_only=False,
                                  with_mesh_postprocess=False, with_texture_baking=False,
                                  with_layout_postprocess=not args.no_layout_refine, use_vertex_color=True,
                                  stage1_inference_steps=None, pointmap=pm)
    glb = res["glb"]
    if isinstance(glb, trimesh.Scene):
        glb = glb.dump(concatenate=True)
    s = res["scale"][0].detach().cpu().numpy().astype(float)                   # (3,), equal
    R = quaternion_to_matrix(res["rotation"]).detach().cpu().numpy()[0].astype(float)
    t = res["translation"][0].detach().cpu().numpy().astype(float)

    local = np.asarray(glb.vertices, float) @ GLB_TO_LOCAL                     # back to the z-up local frame
    F = np.asarray(glb.faces)
    T0 = np.eye(4)                                                               # local -> camera (OpenCV), metres:
    T0[:3, :3], T0[:3, 3] = P3D_TO_CV @ R.T * s[0], P3D_TO_CV @ t                # p = (s l R + t) P3D_TO_CV
    iou0 = project_iou(local @ T0[:3, :3].T + T0[:3, 3], F, K, mask)[0]
    reg = dict(candidate="sam3d", mask_iou=iou0, scale=float(s[0]))
    if not args.no_register:
        T0, reg = register_to_observation(local, F, T0, observed_points(depth, mask, K), K, mask)
    V = local @ T0[:3, :3].T + T0[:3, 3]                                         # camera, OpenCV, metres
    colors = getattr(glb.visual, "vertex_colors", None)
    trimesh.Trimesh(local, F, vertex_colors=colors, process=False).export(out / "mesh.obj")
    posed = trimesh.Trimesh(V, F, vertex_colors=colors, process=False)
    posed.export(out / "posed_mesh.obj")

    iou, sil = project_iou(V, F, K, mask)
    over = cv2.imread(args.rgb)
    over[sil] = (0.5 * over[sil] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(over, cnt, -1, (0, 0, 255), 1)
    cv2.imwrite(str(out / "overlay.png"), over)
    obj_depth = float(np.median(depth[mask & (depth > 0)])) if (mask & (depth > 0)).any() else float("nan")
    meta = dict(model="sam-3d-objects", seed=args.seed, layout_refine=not args.no_layout_refine, depth="sam3d" if args.own_depth else args.depth,
                scale=s.tolist(), rotation_wxyz=res["rotation"][0].tolist(), translation_p3d=t.tolist(),
                sam3d_layout_mask_iou=iou0, registration=reg, local_to_camera=T0.tolist(),
                extents_m=np.asarray(posed.extents).tolist(), centroid_m=np.asarray(posed.centroid).tolist(),
                faces=int(len(F)), watertight=bool(posed.is_watertight), mask_iou=iou, mask_depth_median_m=obj_depth)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"[sam3d] {args.name}: {len(F)} faces, extents {np.round(np.sort(posed.extents)[::-1] * 100, 1)} cm, "
          f"centroid depth {posed.centroid[2]:.2f} m (mask depth {obj_depth:.2f} m), projection vs mask IoU {iou:.2f} "
          f"(SAM 3D's layout {iou0:.2f}; pose from {reg['candidate']}) "
          f"-> {out / 'posed_mesh.obj'}")


if __name__ == "__main__":
    main()
