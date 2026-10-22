#!/usr/bin/env python3
"""MoGe metric depth of every frame, as uint16-millimetre PNGs (RecGen's format), one per frame.

The model is loaded once for the whole video. --fov-x (estimate_intrinsics_moge.py) locks MoGe's focal length to
the camera's.

Run in the `moge` env:
    python scripts/reconstruction/estimate_depth_moge.py --frames-dir <recon>/all_frames --out-dir <recon>/depth_moge --fov-x 63.7
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, help="RGB frames")
    parser.add_argument("--out-dir", required=True, help="depth PNGs")
    parser.add_argument("--fov-x", type=float, default=None, help="horizontal field of view, degrees")
    parser.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moge-repo", default=None, help="local MoGe checkout for sys.path")
    args = parser.parse_args()
    if args.moge_repo:
        sys.path.insert(0, args.moge_repo)
    from moge.model import import_model_class_by_version

    device = torch.device(args.device)
    model = import_model_class_by_version(args.version).from_pretrained(args.checkpoint).to(device).eval()
    frames = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg")))
    os.makedirs(args.out_dir, exist_ok=True)
    infer_kwargs = {"fov_x": args.fov_x}
    if args.version == "v3":
        infer_kwargs["refine_steps"] = args.refine_steps
    for i, name in enumerate(frames):
        image = cv2.cvtColor(cv2.imread(os.path.join(args.frames_dir, name)), cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            output = model.infer(tensor, **infer_kwargs)
        depth = output["depth"].cpu().numpy()
        finite = np.isfinite(depth) & output["mask"].cpu().numpy()
        depth_mm = np.clip(np.where(finite, depth * 1000.0, 0.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png"), depth_mm)
        if i % 25 == 0 or i == len(frames) - 1:
            print(f"[moge] {i + 1}/{len(frames)} {name}", flush=True)
    print(f"[moge] wrote {len(frames)} depth maps to {args.out_dir}")


if __name__ == "__main__":
    main()
