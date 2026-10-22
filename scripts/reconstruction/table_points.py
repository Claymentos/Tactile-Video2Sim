#!/usr/bin/env python3
"""Point prompts on the table at the reference frame, for when SAM3's text prompt misses it there.

SAM3 prompted with 'table' can find the table only later in the video (pick_up_screwdriver/103227: from frame
71 on), and Benchmark-Genesis' reconstruction (run_touchanything_pipeline.sh) needs it at the reference frame,
where it seeds the camera tracking; Benchmark-Genesis then takes clicked points (TABLE_POINTS). This picks them:
the table mask of the frame nearest the reference one where the text prompt found it, carried to the reference
frame (the head camera moves: the table is a plane, so a homography per frame, from Lucas-Kanade tracks of corners
inside the mask + RANSAC, chained), eroded, minus the object's mask and the gloves' (dilated) at the reference frame;
--n points spread over what remains (farthest-point sampling). Prints "x,y;x,y;..." (TABLE_POINTS), nothing
when no frame has a table or the mask cannot be carried.

    python scripts/reconstruction/table_points.py --recon data/egotouch/bg_recon/<key> --frame 0 --object yellow_screwdriver
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="the reconstruction folder (video_segmentation/masks, source/)")
    ap.add_argument("--frame", type=int, default=0, help="the reference frame")
    ap.add_argument("--object", required=True, help="the object's mask name")
    ap.add_argument("--static", default="table", help="the static mask name")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--erode", type=int, default=50, help="px taken off the carried table's border")
    ap.add_argument("--clear", type=int, default=25, help="px kept away from the object and the gloves")
    args = ap.parse_args()
    masks = Path(args.recon) / "video_segmentation" / "masks"
    read = lambda f, name: cv2.imread(str(masks / f"frame_{f:06d}_masks" / f"{name}.png"), 0)
    frames = sorted(int(p.name[6:12]) for p in masks.glob("frame_*_masks"))
    found = [f for f in frames if (m := read(f, args.static)) is not None and (m > 128).sum() > 100]
    if not found:
        return
    k = min(found, key=lambda f: abs(f - args.frame))
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    table = (read(k, args.static) > 128).astype(np.uint8)
    frame = lambda f: cv2.imread(str(Path(args.recon) / "all_frames" / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = table.shape
    step = 1 if args.frame > k else -1
    for a in range(k, args.frame, step):
        pa = cv2.goodFeaturesToTrack(frame(a), 800, 0.005, 7, mask=cv2.erode(table, disk(5)))
        if pa is None or len(pa) < 12:
            return
        pb, ok, _ = cv2.calcOpticalFlowPyrLK(frame(a), frame(a + step), pa, None, winSize=(31, 31), maxLevel=4)
        ok = ok.ravel() == 1
        if ok.sum() < 12:
            return
        H, inl = cv2.findHomography(pa[ok], pb[ok], cv2.RANSAC, 2.0)
        if H is None or inl.sum() < 12:
            return
        table = cv2.warpPerspective(table, H, (w, h), flags=cv2.INTER_NEAREST)
    table = cv2.erode(table, disk(args.erode)) > 0
    away = read(args.frame, args.object) > 128
    gloves = Path(args.recon) / "source" / "masks.npz"
    if gloves.exists():
        g = np.load(gloves)["masks"]
        if args.frame < len(g):
            away |= g[args.frame].astype(bool).any(0)
    table &= ~(cv2.dilate(away.astype(np.uint8), disk(args.clear)) > 0)
    ys, xs = np.nonzero(table)
    if len(xs) == 0:
        return
    pts = np.stack([xs, ys], 1).astype(float)
    chosen = [pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]]     # the most central one first
    d = np.linalg.norm(pts - chosen[0], axis=1)
    while len(chosen) < args.n:
        chosen.append(pts[np.argmax(d)])
        d = np.minimum(d, np.linalg.norm(pts - chosen[-1], axis=1))
    print(f"table found by text on frame {k}, points on frame {args.frame}", file=sys.stderr)
    print(";".join(f"{x:.0f},{y:.0f}" for x, y in chosen))


if __name__ == "__main__":
    main()
