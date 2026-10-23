#!/usr/bin/env python3
"""Depth-Anything-3 metric depth of every frame, as uint16-millimetre PNGs: the format and folder of
estimate_depth_moge.py, so that the rest of launch/reconstruct.sh reads it unchanged (DEPTH=da3).

DA3METRIC-LARGE, one frame at a time: metric depth = focal * output / 300, the focal in pixels of the image the
network sees (intrinsics.yaml's, scaled to it). On pick_up_screwdriver/103227 MoGe puts the screwdriver, at rest on
the table, at 121 cm in frame 20 and 102 cm in frame 46 while DA3 keeps it at 88-92 cm, and at the glove's contact
the object is 1-2 cm behind the glove with DA3 against 5-7 cm with MoGe: the hands, placed frame by frame on the
depth, then meet the object whose rest pose comes from the first frames.

Run in the `da3` env (launch/config.sh PY_DA3: a venv over the moge env with ~/Depth-Anything-3 installed):
    python scripts/reconstruction/estimate_depth_da3.py --frames-dir <recon>/all_frames --out-dir <recon>/depth_moge \\
        --intrinsics <recon>/intrinsics.yaml
"""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, help="RGB frames")
    parser.add_argument("--out-dir", required=True, help="depth PNGs")
    parser.add_argument("--intrinsics", required=True, help="intrinsics.yaml (fu, fv, pu, pv)")
    parser.add_argument("--checkpoint", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--process-res", type=int, default=504, help="DA3's input size (longest side)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from depth_anything_3.api import DepthAnything3

    with open(args.intrinsics) as handle:
        calib = yaml.safe_load(handle)
    focal = 0.5 * (float(calib["fu"]) + float(calib["fv"]))
    model = DepthAnything3.from_pretrained(args.checkpoint).to(torch.device(args.device))
    frames = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg")))
    os.makedirs(args.out_dir, exist_ok=True)
    for i, name in enumerate(frames):
        path = os.path.join(args.frames_dir, name)
        height, width = cv2.imread(path).shape[:2]
        pred = model.inference([path], process_res=args.process_res)
        out = pred.depth[0]
        depth = focal * out.shape[1] / width * out / 300.0                    # metres, the processed image's focal
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_mm = np.clip(np.where(np.isfinite(depth), depth * 1000.0, 0.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png"), depth_mm)
        if i % 25 == 0 or i == len(frames) - 1:
            print(f"[da3] {i + 1}/{len(frames)} {name}", flush=True)
    print(f"[da3] wrote {len(frames)} depth maps to {args.out_dir}")


if __name__ == "__main__":
    main()
