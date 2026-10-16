#!/usr/bin/env python3
"""Perception stage 4: a mesh of every (movable) object with RecGen, from one RGB-D frame and its mask.

Per object of the scene file, the reference frame is picked among the sampled frames within --window-s
of `visible_at_s` (where Gemini saw it unoccluded): the largest SAM3 masks that do not touch the image
border (a cut mask gives a cut mesh) and have depth on most of their pixels; RecGen runs on the best
--candidates of them (1 s apart) and the mesh whose projection best covers its SAM3 mask (silhouette IoU:
scale and pose from a single view vary with the view) is kept. A container (Gemini's
`is_container`) that holds other objects at some point is reconstructed while it is empty instead
(empty_windows: the objects' `initial_location` and the clips' from / to give what is inside it when),
from the frame where the hands and the other objects cover least of it -- otherwise its mesh carries
the contents. Without such a window, --window-s around `visible_at_s` as for any object. RecGen
(TRI-ML/RecGen, recgen_base.multiview_stereo) then reconstructs the object from that frame's RGB, depth
(sensor or MoGe, prepare_frames.py / estimate_depth_moge.py), mask (segment_objects_sam3.py) and the
intrinsics. Its mask erosion (5x5) deletes thin parts: --no-mask-erosion keeps them.

The support surface (--support, 'table': a plane in the simulation) is skipped; static objects (a drilled
block, a mat) are reconstructed too, the simulation keeps them fixed. Writes <perception>/recgen/<object_id>/
(RecGen's mesh.obj, posed_mesh.obj -- the mesh in the camera frame of the reference frame, metres --
metadata.json, overlay.png, plus input.png: the frame with the mask) and <perception>/objects.json: per
object the reference frame, the mesh's centre in the camera frame and its extent (robust_extent_cm) next
to Gemini's size estimate (a scale sanity check of the depth), and T_world_mesh, the camera pose of the
reference frame (posed_mesh.obj -> world) when the camera is registered (register_camera.py, which also
gives the depth scale applied here).

Run in the `recgen` conda env (RECGEN_DIR: the RecGen checkout, default ~/recgen).
    python scripts/reconstruction/reconstruct_objects_recgen.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_objects_sam3 import segmentation_masks  # noqa: E402

RECGEN_DIR = Path(os.environ.get("RECGEN_DIR", Path.home() / "recgen"))


def robust_extent_cm(vertices: np.ndarray) -> list[float]:
    """Extent along the mesh's principal axes, between the 1st and 99th percentiles, largest first: RecGen
    meshes carry small detached fragments that blow up a bounding box."""
    v = vertices - vertices.mean(0)
    _, _, axes = np.linalg.svd(v[:: max(1, len(v) // 20000)], full_matrices=False)
    p = v @ axes.T
    return sorted(((np.percentile(p, 99, 0) - np.percentile(p, 1, 0)) * 100).round(1).tolist(), reverse=True)


def empty_windows(scene: dict, cid: str, duration_s: float, margin_s: float = 0.5) -> list[tuple[float, float]]:
    """Time windows where Gemini's scene puts nothing inside container `cid`: an object is inside from the
    start if its initial_location names the container, then a clip moving it `from` the container takes
    it out at the clip's end and one moving it `to` the container puts it back at the clip's start
    (`margin_s` kept clear of both). [] when nothing is ever inside."""
    name = cid.replace("_", " ")
    inside = {o["id"] for o in scene["objects"]
              if o["id"] != cid and (cid in o.get("initial_location", "") or name in o.get("initial_location", ""))}
    events = []
    for c in scene.get("clips", []):
        for side in ("left_hand", "right_hand"):
            a = c.get(side, {})
            if a.get("object") in (None, "none"):
                continue
            if a.get("from") == cid:
                events.append((c["end_s"] + margin_s, a["object"], False))
            if a.get("to") == cid:
                events.append((c["start_s"] - margin_s, a["object"], True))
    if not inside and not any(e[2] for e in events):
        return []
    windows, t0 = [], 0.0 if not inside else None
    for t, obj, put in sorted(events):
        was_empty = not inside
        inside.add(obj) if put else inside.discard(obj)
        if was_empty and inside and t0 is not None and t > t0:
            windows.append((t0, t))
        if not was_empty and not inside:
            t0 = t
    if not inside and t0 is not None and t0 < duration_s:
        windows.append((t0, duration_s))
    return windows


def reference_frames(masks: np.ndarray, depth_dir: Path, times: np.ndarray, t_vis: float, window_s: float,
                     windows: list[tuple[float, float]] | None = None, occluders: np.ndarray | None = None,
                     k: int = 1, min_gap_s: float = 1.0) -> list[int]:
    """Sampled frames to reconstruct from, best first (at most k, min_gap_s apart): the largest masks near
    t_vis (or inside `windows`), not cut by the border, with depth; with `occluders` (T, H, W: hands and
    other objects), weighted by the share of the mask's bounding box they leave free."""
    scores = {}
    near = np.abs(times - t_vis) <= window_s
    if windows:
        near = np.zeros(len(times), bool)
        for a, b in windows:
            near |= (times >= a) & (times <= b)
    for t in np.flatnonzero(near):
        m = masks[t]
        area = int(m.sum())
        dpath = depth_dir / f"{t:06d}.png"
        if area < 200 or not dpath.exists():
            continue
        border = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        coverage = float((cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)[m] > 0).mean())
        score = area * coverage * (0.3 if border else 1.0) * (coverage > 0.8)
        if occluders is not None and area:
            ys, xs = np.nonzero(m)
            box = occluders[t, ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            score *= (1.0 - box.mean()) ** 4
        if score > 0:
            scores[int(t)] = score
    out = []
    for t in sorted(scores, key=scores.get, reverse=True):
        if all(abs(times[t] - times[u]) >= min_gap_s for u in out):
            out.append(t)
        if len(out) == k:
            break
    return out


def silhouette_iou(vertices: np.ndarray, K: np.ndarray, mask: np.ndarray) -> float:
    """IoU between the posed mesh's projection (camera frame) and the SAM3 mask: how well RecGen's mesh,
    its scale and its pose explain the frame it was made from."""
    v = vertices[vertices[:, 2] > 1e-3]
    uv = (v @ K.T)[:, :2] / v[:, 2:3]
    H, W = mask.shape
    sil = np.zeros((H, W), np.uint8)
    u, w = np.round(uv).astype(int).T
    ok = (u >= 0) & (u < W) & (w >= 0) & (w < H)
    sil[w[ok], u[ok]] = 1
    sil = cv2.morphologyEx(cv2.dilate(sil, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    sil = sil.astype(bool)
    return float((sil & mask).sum() / max((sil | mask).sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--objects", default=None, help="comma list of object ids (default: every movable object)")
    ap.add_argument("--support", default="auto", help="support surface, not reconstructed (a plane in sim); "
                                                      "auto: tv2s_data.scene.support_id")
    ap.add_argument("--window-s", type=float, default=2.0, help="reference frame search around visible_at_s")
    ap.add_argument("--no-mask-erosion", action="store_true", help="keep thin parts (RecGen erodes 5x5)")
    ap.add_argument("--checkpoint", default="recgen_base.multiview_stereo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidates", type=int, default=3,
                    help="frames reconstructed per object; the mesh whose projection best fits its mask is kept")
    args = ap.parse_args()
    root = Path(args.perception).resolve()
    meta = json.loads((root / "frames.json").read_text())
    if not meta.get("depth"):
        raise SystemExit(f"{root}: no depth yet (estimate_depth_moge.py)")
    scene = json.loads(Path(meta["scene"]).read_text())
    if args.support == "auto":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from tv2s_data.scene import support_id

        args.support = support_id(scene)
    seg = json.loads((root / "sam3" / "summary.json").read_text())
    times = np.asarray(meta["times_s"])
    intr = yaml.safe_load((root / "intrinsics.yaml").read_text())
    K = np.array([[intr["fu"], 0, intr["pu"]], [0, intr["fv"], intr["pv"]], [0, 0, 1]], float)
    objects = [o for o in scene["objects"]
               if (o["id"] in args.objects.split(",") if args.objects else o["id"] != args.support)]

    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(RECGEN_DIR))
    os.chdir(RECGEN_DIR)                                  # RecGen resolves its configs from its checkout
    import trimesh
    from recgen_inference import build_recgen, generate

    pipeline = build_recgen.build(args.checkpoint)
    cam = root / "camera.npz"                                  # register_camera.py: posed_mesh.obj -> world
    world = np.load(cam)["T_world_cam"] if cam.exists() else None
    summary_path = root / "objects.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    import shutil

    for o in objects:
        if not seg.get(o["id"], {}).get("prompt"):
            print(f"  {o['id']:20s} no SAM3 mask, skipped")
            summary.pop(o["id"], None)                         # a mesh of an earlier run is stale
            shutil.rmtree(root / "recgen" / o["id"], ignore_errors=True)
            continue
        masks = segmentation_masks(root / "sam3" / f"{o['id']}.npz")
        windows, occluders = [], None
        if o.get("is_container"):
            windows = empty_windows(scene, o["id"], float(times[-1]))
            if windows:
                occluders = np.zeros(masks.shape, bool)
                for other in [p for p in scene["objects"] if p["id"] not in (o["id"], args.support)] + [dict(id="hands")]:
                    f = root / "sam3" / f"{other['id']}.npz"
                    if f.exists():
                        occluders |= segmentation_masks(f).any(0) if other["id"] == "hands" else segmentation_masks(f)
                print(f"  {o['id']:20s} container, empty during " + ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in windows))
        frames = reference_frames(masks, root / "depth", times, o["visible_at_s"], args.window_s, windows, occluders,
                                  k=args.candidates)
        if not frames:
            print(f"  {o['id']:20s} no usable frame within {args.window_s} s of {o['visible_at_s']} s, skipped")
            continue
        out = root / "recgen" / o["id"]
        scale = float(meta.get("depth_scale", 1.0))           # register_camera.py: monocular depth -> metric
        best = None
        for t in frames:                                       # RecGen from each candidate, the best fit kept
            rgb = cv2.cvtColor(cv2.imread(str(root / "frames" / f"{t:06d}.jpg")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(root / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED)
            depth = np.clip(depth.astype(np.float64) * scale, 0, 65535).astype(np.uint16)
            mask = masks[t].astype(np.uint8) * 255
            cand = out.parent / f".{o['id']}_frame{t}"
            shutil.rmtree(cand, ignore_errors=True)
            cand.mkdir(parents=True)
            view = rgb.copy()
            view[masks[t]] = (0.5 * view[masks[t]] + [0, 127, 0]).astype(np.uint8)
            cv2.imwrite(str(cand / "input.png"), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            result = generate(pipeline, image=rgb, depth=depth, mask=mask, intrinsics=K, seed=args.seed,
                              mask_erosion_enabled=not args.no_mask_erosion)
            result.save(cand)
            iou = silhouette_iou(trimesh.load(cand / "posed_mesh.obj", force="mesh").vertices, K, masks[t])
            print(f"  {o['id']:20s} candidate frame {t} ({times[t]:.1f} s): silhouette IoU {iou:.2f}", flush=True)
            if best is None or iou > best[1]:
                best = (t, iou, cand)
        t, iou, cand = best
        shutil.rmtree(out, ignore_errors=True)
        shutil.move(str(cand), str(out))
        for f in out.parent.glob(f".{o['id']}_frame*"):
            shutil.rmtree(f, ignore_errors=True)
        mesh = trimesh.load(out / "posed_mesh.obj", force="mesh")
        extent_cm = robust_extent_cm(mesh.vertices)
        gemini_cm = sorted(o["size_cm"], reverse=True)
        summary[o["id"]] = dict(
            reference_frame=t, reference_time_s=float(times[t]), video_frame=int(meta["frame_index"][t]),
            depth_scale=scale, T_world_mesh=world[t].tolist() if world is not None else None,
            mask_area_px=int(masks[t].sum()), depth_source=meta["depth"], mesh=str(out / "posed_mesh.obj"),
            centre_camera_m=np.median(mesh.vertices, 0).round(4).tolist(), extent_cm=extent_cm,
            gemini_size_cm=gemini_cm, scale_vs_gemini=round(extent_cm[0] / max(gemini_cm[0], 1e-3), 2),
            n_vertices=int(len(mesh.vertices)), n_components=len(mesh.split(only_watertight=False)),
            silhouette_iou=round(iou, 3), candidates=frames)
        print(f"  {o['id']:20s} frame {t} ({times[t]:.1f} s): {len(mesh.vertices)} vertices, extent "
              f"{extent_cm} cm (Gemini {gemini_cm}), {np.median(mesh.vertices, 0)[2]:.2f} m away", flush=True)
        summary_path.write_text(json.dumps(summary, indent=1))
    print(f"{len(summary)} meshes -> {root / 'recgen'}")


if __name__ == "__main__":
    main()
