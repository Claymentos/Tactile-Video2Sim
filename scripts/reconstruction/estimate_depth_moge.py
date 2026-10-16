#!/usr/bin/env python3
"""Perception stage 3: MoGe metric depth of the sampled frames (videos without a depth sensor).

MoGe (Ruicheng/moge-3-vitl) is given the camera's horizontal field of view from the intrinsics, so its
focal length is the real one and only the depth is estimated (Benchmark-Genesis' run_moge_depth_batch.py).
Writes <perception>/depth/<frame>.png, uint16 millimetres (RecGen's format; 0 = no depth), and sets
frames.json's `depth` to "moge". A folder whose depth comes from a sensor (TouchScale) is left alone
unless --force.

Run in the `moge` conda env.
    python scripts/reconstruction/estimate_depth_moge.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (prepare_frames.py)")
    ap.add_argument("--frames", default=None, help="comma list of frame indices (default: every frame)")
    ap.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    ap.add_argument("--version", default="v3", choices=("v1", "v2", "v3"))
    ap.add_argument("--refine-steps", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="replace a sensor depth")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    if meta.get("depth") == "sensor" and not args.force:
        raise SystemExit(f"{root} has sensor depth (--force to replace it with MoGe)")

    from moge.model import import_model_class_by_version

    device = torch.device("cuda")
    model = import_model_class_by_version(args.version).from_pretrained(args.checkpoint).to(device).eval()
    kwargs = dict(fov_x=meta["fov_x_deg"], **({"refine_steps": args.refine_steps} if args.version == "v3" else {}))
    frames = sorted((root / "frames").glob("*.jpg"))
    if args.frames:
        wanted = {int(i) for i in args.frames.split(",")}
        frames = [f for f in frames if int(f.stem) in wanted]
    (root / "depth").mkdir(exist_ok=True)
    for k, path in enumerate(frames):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            o = model.infer(torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1), **kwargs)
        depth = o["depth"].cpu().numpy()
        ok = np.isfinite(depth) & o["mask"].cpu().numpy()
        cv2.imwrite(str(root / "depth" / f"{path.stem}.png"),
                    np.clip(np.where(ok, depth * 1000.0, 0.0), 0, 65535).astype(np.uint16))
        if k % 50 == 0 or k == len(frames) - 1:
            print(f"  {k + 1}/{len(frames)} frame {path.stem}: median {np.median(depth[ok]):.2f} m", flush=True)
    meta["depth"] = "moge"
    (root / "frames.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(frames)} MoGe depth maps -> {root / 'depth'}")


if __name__ == "__main__":
    main()
