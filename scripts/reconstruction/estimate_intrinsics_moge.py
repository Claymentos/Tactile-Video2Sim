#!/usr/bin/env python3
"""Camera intrinsics from MoGe on one frame, for footage without calibration (EgoTouch): intrinsics.yaml
(fu, fv, pu, pv), which every later stage reads.

Prints the horizontal field of view (FOV_X=...), which estimate_depth_moge.py passes to MoGe to lock its focal
length on every frame.

Run in the `moge` env:
    python scripts/reconstruction/estimate_intrinsics_moge.py --image <recon>/all_frames/000000.png --out <recon>/intrinsics.yaml
"""

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="the reference frame")
    parser.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--moge-repo", default=None, help="local MoGe checkout for sys.path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import torch

    if args.moge_repo:
        sys.path.insert(0, args.moge_repo)
    from moge.model import import_model_class_by_version

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = import_model_class_by_version(args.version).from_pretrained(args.checkpoint).to(device).eval()
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        intrinsics = model.infer(tensor)["intrinsics"].cpu().numpy()          # normalised by the image size
    height, width = image.shape[:2]
    fx, fy = float(intrinsics[0, 0] * width), float(intrinsics[1, 1] * height)
    cx, cy = float(intrinsics[0, 2] * width), float(intrinsics[1, 2] * height)
    horizontal_fov = math.degrees(2 * math.atan(int(round(cx * 2)) / (2 * fx)))
    with open(args.out, "w") as handle:
        handle.write(f"fu: {fx}\nfv: {fy}\npu: {cx}\npv: {cy}\n")
    print(f"Wrote {args.out}")
    print(f"FOV_X={horizontal_fov}")
    print(f"FOCAL={fx}")


if __name__ == "__main__":
    main()
