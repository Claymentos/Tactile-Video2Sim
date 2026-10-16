#!/usr/bin/env python3
"""Perception: WiLoR hand detections on the sampled frames (2D joints, per-frame MANO).

Per sampled frame (prepare_frames.py), WiLoR's YOLO detector finds the hands and WiLoR regresses MANO for
each; the best detection of each side is kept. The 21 joints (MediaPipe / OpenPose order) are projected
into the frame with WiLoR's own camera (focal 5000/256 * max(W, H), principal point at the image centre:
the convention its keypoints reproject onto the image with), which gives their pixel coordinates.

These 2D joints place a hand mocap in the camera (register_camera.py), and WiLoR's per-frame MANO is the
fallback hand source of videos without mocap.

Writes <perception>/hands_wilor.npz, per side (left, right):
  <side>_kp2d (T, 21, 2) pixels, NaN where the hand is not detected; <side>_conf (T,) detector confidence;
  <side>_joints_cam (T, 21, 3) and <side>_cam_t (T, 3) in WiLoR's camera; <side>_global_orient (T, 3, 3),
  <side>_hand_pose (T, 15, 3, 3), <side>_betas (T, 10) (MANO of the right-hand model, mirrored for the left)
and focal_wilor.

Run in the `wilor` conda env (WILOR_DIR: the WiLoR checkout, default ~/WiLoR).
    python scripts/reconstruction/detect_hands_wilor.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

WILOR_DIR = Path(os.environ.get("WILOR_DIR", Path.home() / "WiLoR"))
SIDES = ("left", "right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (prepare_frames.py)")
    ap.add_argument("--conf", type=float, default=0.3, help="detector confidence threshold")
    ap.add_argument("--rescale-factor", type=float, default=2.0, help="bounding box padding (WiLoR's default)")
    args = ap.parse_args()
    root = Path(args.perception).resolve()
    frames = sorted((root / "frames").glob("*.jpg"))

    sys.path.insert(0, str(WILOR_DIR))
    os.chdir(WILOR_DIR)                                   # WiLoR loads its configs relative to its checkout
    # torch >= 2.6 loads weights_only by default, which rejects the YOLO and Lightning checkpoints (they
    # pickle their classes); these are WiLoR's own local files (as run_hawor.py in Benchmark-Genesis)
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    device = torch.device("cuda")
    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                            cfg_path="./pretrained_models/model_config.yaml")
    model = model.to(device).eval()
    detector = YOLO("./pretrained_models/detector.pt").to(device)

    T = len(frames)
    out = {f"{s}_{k}": np.full((T, *shape), np.nan, np.float32) for s in SIDES
           for k, shape in (("kp2d", (21, 2)), ("joints_cam", (21, 3)), ("cam_t", (3,)), ("global_orient", (3, 3)),
                            ("hand_pose", (15, 3, 3)), ("betas", (10,)))}
    out.update({f"{s}_conf": np.zeros(T, np.float32) for s in SIDES})
    focal = None
    for t, path in enumerate(frames):
        img = cv2.imread(str(path))
        dets = detector(img, conf=args.conf, verbose=False)[0]
        best = {}                                         # side -> (confidence, box)
        for d in dets:
            data = d.boxes.data.cpu().numpy().squeeze()
            side = SIDES[int(d.boxes.cls.cpu().item())]   # class 1 = right hand
            if side not in best or data[4] > best[side][0]:
                best[side] = (float(data[4]), data[:4])
        if not best:
            continue
        sides = list(best)
        boxes = np.stack([best[s][1] for s in sides])
        right = np.array([float(s == "right") for s in sides])
        batch = next(iter(torch.utils.data.DataLoader(
            ViTDetDataset(cfg, img, boxes, right, rescale_factor=args.rescale_factor), batch_size=len(sides))))
        batch = recursive_to(batch, device)
        with torch.no_grad():
            pred = model(batch)
        sign = 2 * batch["right"] - 1
        cam = pred["pred_cam"]
        cam[:, 1] = sign * cam[:, 1]
        size = batch["img_size"].float()
        focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * float(size.max())
        cam_t = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(), size, focal).cpu().numpy()
        for n, s in enumerate(sides):
            j = pred["pred_keypoints_3d"][n].cpu().numpy()
            j[:, 0] *= float(sign[n])
            p = j + cam_t[n]
            w, h = size[n].cpu().numpy()
            out[f"{s}_kp2d"][t] = focal * p[:, :2] / p[:, 2:] + [w / 2, h / 2]
            out[f"{s}_joints_cam"][t], out[f"{s}_cam_t"][t] = j, cam_t[n]
            mp = pred["pred_mano_params"]
            out[f"{s}_global_orient"][t] = mp["global_orient"][n, 0].cpu().numpy()
            out[f"{s}_hand_pose"][t] = mp["hand_pose"][n].cpu().numpy()
            out[f"{s}_betas"][t] = mp["betas"][n].cpu().numpy()
            out[f"{s}_conf"][t] = best[s][0]
        if t % 100 == 0 or t == T - 1:
            print(f"  {t + 1}/{T}: " + ", ".join(f"{s} {np.isfinite(out[f'{s}_kp2d'][:t + 1, 0, 0]).mean():.0%}"
                                                 for s in SIDES), flush=True)
    np.savez_compressed(root / "hands_wilor.npz", focal_wilor=np.float32(focal or 0), **out)
    print(f"WiLoR on {T} frames: " + ", ".join(f"{s} hand in {np.isfinite(out[f'{s}_kp2d'][:, 0, 0]).mean():.0%}"
                                               for s in SIDES) + f" -> {root / 'hands_wilor.npz'}")


if __name__ == "__main__":
    main()
