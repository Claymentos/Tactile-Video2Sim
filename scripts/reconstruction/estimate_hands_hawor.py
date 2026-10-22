#!/usr/bin/env python3
"""Both hands as MANO from the video with HaWoR, for videos without a hand mocap (launch/reconstruct.sh, HANDS=hawor).

HaWoR (do-as-i-do/reconstruction/modules/HaWoR, run as Benchmark-Genesis' run_hawor.py does) regresses MANO per hand
over time, with --static_camera: each frame's hands come out in that frame's camera (OpenCV axes), and the camera's
own motion (reconstruct.sh's table track) carries them into the reference camera afterwards (place_video_hands.py).
Its focal is MoGe's (intrinsics.yaml).

Its YOLO hand detector misses black gloves (HumanTouch's, ACE-Data-0's), so with --instances the detections are given
to it: the hand boxes come from SAM3's hand instances (segment_video_sam3.py --instances, e.g. "black glove"), each
instance's side from its mean image position (in an egocentric view the left hand is on the left), written where
HaWoR looks for its own (tracks_<start>_<end>/model_{boxes,tracks}.npy), so it skips its detection.

Writes <work>/hawor_hands.npz: per side <side>_joints (T, 21, 3) (OpenPose / MediaPipe order) and <side>_vertices
(T, 778, 3) in each frame's camera, <side>_valid (T,), the MANO parameters (<side>_rot, _hand_pose, _betas, _trans),
<side>_box (T, 4) the boxes given (NaN: none), img_focal.

Run in the `hawor` env (HAWOR_DIR: the checkout):
    python scripts/reconstruction/estimate_hands_hawor.py --frames-dir <recon>/all_frames --intrinsics <recon>/intrinsics.yaml \\
        --work <recon>/hawor --instances <recon>/video_segmentation/masks/instances_glove.npz
"""

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HAWOR_DIR = Path(os.environ.get("HAWOR_DIR", Path.home() / "do-as-i-do/reconstruction/modules/HaWoR"))
SIDES = ("left", "right")


def hand_boxes(masks: np.ndarray, margin: float, min_pixels: int = 50) -> tuple[dict, dict]:
    """Per side, (T, 4) xyxy boxes (NaN where no hand) from SAM3 instances (T, N, H, W): each instance's side by its
    mean x over the frames it is in; per frame the union of a side's masks, padded by `margin` of its size."""
    T, N, H, W = masks.shape
    sides = {}
    for n in range(N):
        xs = [np.nonzero(m.any(0))[0].mean() for m in masks[:, n] if m.sum() > min_pixels]
        if xs:
            sides.setdefault("left" if np.mean(xs) < W / 2 else "right", []).append(n)
    out = {}
    for side in SIDES:
        b = np.full((T, 4), np.nan)
        for t in range(T):
            m = masks[t, sides.get(side, [])].any(0)
            if m.sum() > min_pixels:
                ys, xs = np.nonzero(m)
                w, h = xs.max() - xs.min(), ys.max() - ys.min()
                b[t] = [max(0, xs.min() - margin * w), max(0, ys.min() - margin * h),
                        min(W - 1, xs.max() + margin * w), min(H - 1, ys.max() + margin * h)]
        out[side] = b
    return out, sides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="every frame of the video (000000.png, ...)")
    ap.add_argument("--intrinsics", required=True, help="intrinsics.yaml (fu: HaWoR's focal)")
    ap.add_argument("--work", required=True, help="HaWoR's working folder; hawor_hands.npz is written there")
    ap.add_argument("--instances", default=None, help="SAM3 hand instances (segment_video_sam3.py --instances); "
                    "without: HaWoR's own detector")
    ap.add_argument("--margin", type=float, default=0.15, help="box padding (share of its size)")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    frames = sorted(Path(args.frames_dir).resolve().glob("*.png"))
    T = len(frames)
    focal = float(yaml.safe_load(Path(args.intrinsics).read_text())["fu"])
    seq = work / "video"                                  # HaWoR's sequence folder for <work>/video.mp4
    shutil.rmtree(seq, ignore_errors=True)
    (seq / "extracted_images").mkdir(parents=True)
    for i, f in enumerate(frames):
        (seq / "extracted_images" / f"{i:06d}.png").symlink_to(f)

    boxes = {s: np.full((T, 4), np.nan) for s in SIDES}
    if args.instances:
        inst = np.load(args.instances)
        if len(inst["frames"]) != T or not (inst["frames"] == np.arange(T)).all():
            raise SystemExit(f"{args.instances}: {len(inst['frames'])} frames, the video {T}")
        boxes, sides = hand_boxes(inst["masks"], args.margin)
        tracks = {}
        for side, tid in (("left", 5000), ("right", 10000)):
            tracks[tid] = [dict(frame=t, det=True, det_box=np.r_[boxes[side][t], 1.0][None],
                                det_handedness=np.array([float(side == "right")]))
                           for t in range(T) if np.isfinite(boxes[side][t]).all()]
        tdir = seq / f"tracks_0_{T}"
        tdir.mkdir()
        np.save(tdir / "model_boxes.npy", np.array([], dtype=object))
        np.save(tdir / "model_tracks.npy", np.array(tracks, dtype=object))
        ids = inst["obj_ids"]
        print(f"{T} frames; SAM3 instances per side { {s: [int(ids[n]) for n in v] for s, v in sides.items()} }; "
              "boxes on " + ", ".join(f"{s} {np.isfinite(boxes[s][:, 0]).mean():.0%}" for s in SIDES), flush=True)

    # HaWoR, as Benchmark-Genesis' run_hawor.py runs it (static camera: no SLAM)
    import torch

    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
    (work / "video.mp4").touch()                          # placeholder: HaWoR finds the frames already extracted
    os.chdir(HAWOR_DIR)
    sys.path.insert(0, str(HAWOR_DIR))
    from lib.vis.renderer import Renderer

    Renderer.render_multiple = lambda self, *a, **k: (np.zeros((self.height, self.width, 3), np.uint8),
                                                     np.zeros((self.height, self.width), bool))
    sys.argv = ["demo.py", "--video_path", str(work / "video.mp4"), "--vis_mode", "cam", "--img_focal", str(focal),
                "--static_camera"]
    try:
        runpy.run_path(str(HAWOR_DIR / "demo.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:                                # the visualisation at the end may fail; the npz is written before
        if not (seq / "all_hand_meshes.npz").exists():
            raise
        print(f"(HaWoR's visualisation failed: {type(e).__name__}; the hands are written)")

    d = np.load(seq / "all_hand_meshes.npz")
    res = dict(img_focal=np.array(focal))
    for s in SIDES:
        valid = np.asarray(d[f"{s}_valid"]).astype(bool).reshape(-1)[:T]
        if args.instances:
            valid &= np.isfinite(boxes[s][:, 0])
        res.update({f"{s}_joints": d[f"{s}_joints"], f"{s}_vertices": d[f"{s}_vertices"], f"{s}_valid": valid,
                    f"{s}_box": boxes[s]})
        for k in ("rot", "hand_pose", "betas", "trans"):
            res[f"{s}_{k}"] = d[f"{s}_{k}"]
    np.savez_compressed(work / "hawor_hands.npz", **res)
    print("HaWoR hands: " + ", ".join(f"{s} valid on {res[f'{s}_valid'].mean():.0%}" for s in SIDES)
          + f" -> {work / 'hawor_hands.npz'}")


if __name__ == "__main__":
    main()
