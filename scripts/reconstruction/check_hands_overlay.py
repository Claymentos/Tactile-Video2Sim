#!/usr/bin/env python3
"""Check video of the mocap hands' placement (place_mocap_hands.py) over the head-camera video (launch/reconstruct.sh).

Per frame, over chest.mp4 (scaled up --scale times): both mocap hands as placed (mocap_joints_cotracker.npz, the
reference camera) carried into the frame's camera by the camera motion and projected, WiLoR's 2D joints, and the outlines of the gloves' masks
(EgoTouch masks.npz). A WiLoR detection lying on the other hand's glove (place_mocap_hands.glove_filter: WiLoR
calls the visible hand by the hidden one's side) is drawn grey and flagged in the band, which also gives the
left / right mocap's median distance to WiLoR's joints of the same side.

    red     the left mocap hand           magenta  the right mocap hand
    green   WiLoR 'left'                  yellow   WiLoR 'right'           grey  a WiLoR hand on the other glove
    blue    the left glove's mask         orange   the right glove's mask

Run in the `genesis` env:
    python scripts/reconstruction/check_hands_overlay.py --recon data/egotouch/recon/<key>   # -> <recon>/hands_wilor_check.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from place_mocap_hands import PARENTS, SIDES, glove_filter, read_jsonl  # noqa: E402

COLOURS = {("mocap", "left"): (0, 0, 255), ("mocap", "right"): (255, 0, 255), ("wilor", "left"): (0, 255, 0),
           ("wilor", "right"): (0, 255, 255), ("glove", "left"): (255, 120, 0), ("glove", "right"): (0, 140, 255)}
LEGEND = [("mocap left (placed)", ("mocap", "left")), ("mocap right (placed)", ("mocap", "right")), ("WiLoR 'left'", ("wilor", "left")),
          ("WiLoR 'right'", ("wilor", "right")), ("left glove", ("glove", "left")), ("right glove", ("glove", "right"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="launch/reconstruct.sh output folder")
    ap.add_argument("--out", default=None, help="default <recon>/hands_wilor_check.mp4")
    ap.add_argument("--scale", type=int, default=2, help="upscaling of the frames")
    args = ap.parse_args()
    R = Path(args.recon)
    ep = (R / "source").resolve()
    placed = np.load(R / "mocap_joints_cotracker.npz")
    K = yaml.safe_load((R / "intrinsics.yaml").read_text())
    fu, fv, pu, pv = (float(K[k]) for k in ("fu", "fv", "pu", "pv"))
    cam = np.load(R / "camera_traj_cotracker.npz")
    wilor = read_jsonl(ep / "wilor_hands.json")
    names = sorted(os.listdir(R / "all_frames"))
    H, W = cv2.imread(str(R / "all_frames" / names[0])).shape[:2]
    focal = 5000.0 / 256.0 * max(W, H)
    masks, dropped, track = None, {}, {"left": 1, "right": 0}
    if (ep / "masks.npz").exists():
        masks = np.load(ep / "masks.npz")["masks"].astype(bool)
        if masks.shape[-2:] == (H, W):
            dropped, track = glove_filter(wilor, masks, focal, W, H)
        else:
            masks = None
    S = args.scale

    import imageio_ffmpeg
    out = Path(args.out or R / "hands_wilor_check.mp4")
    writer = imageio_ffmpeg.write_frames(str(out), (W * S, H * S + 60), fps=30, codec="libx264",
                                         output_params=["-crf", "22"], macro_block_size=1)
    writer.send(None)
    for t, name in enumerate(names):
        img = cv2.resize(cv2.imread(str(R / "all_frames" / name)), (W * S, H * S))

        def skeleton(uv, colour, width):
            for j in range(1, 21):
                cv2.line(img, tuple((uv[PARENTS[j]] * S).astype(int)), tuple((uv[j] * S).astype(int)), colour, width, cv2.LINE_AA)

        if masks is not None and t < len(masks):
            for side in SIDES:
                contours, _ = cv2.findContours(masks[t, track[side]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, [c * S for c in contours], -1, COLOURS["glove", side], 2)
        mocap = {}
        for side in SIDES:
            if not placed[f"{side}_valid"][t]:
                continue
            c = placed[f"{side}_joints"][t] @ cam["rotation"][t].T + cam["translation"][t]
            mocap[side] = np.c_[fu * c[:, 0] / c[:, 2] + pu, fv * c[:, 1] / c[:, 2] + pv]
            skeleton(mocap[side], COLOURS["mocap", side], 3 if side == "left" else 1)
        status, flags = [], []
        for side in SIDES:
            detected = np.asarray(wilor[t].get(f"{side}_pos") or [], dtype=float)
            if detected.shape != (21, 3):
                continue
            uv = focal * detected[:, :2] / detected[:, 2:] + [W / 2, H / 2]
            bad = (t, side) in dropped
            colour = (150, 150, 150) if bad else COLOURS["wilor", side]
            skeleton(uv, colour, 2)
            for p in uv:
                cv2.circle(img, tuple((p * S).astype(int)), 4, colour, -1)
            if side in mocap:
                status.append(f"{side} mocap vs WiLoR {np.median(np.linalg.norm(uv - mocap[side], axis=1)):4.0f} px")
            if bad:
                flags.append(f"WiLoR '{side}' ON THE {'RIGHT' if side == 'left' else 'LEFT'} GLOVE (dropped from the fit)")
        for i, (text, key) in enumerate(LEGEND):
            cv2.putText(img, text, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS[key], 2, cv2.LINE_AA)
        band = np.full((60, W * S, 3), 25, np.uint8)
        cv2.putText(band, f"frame {t:3d}  {t / 30:5.2f} s   " + "   ".join(status), (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(band, "   ".join(flags) or "ok", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 60, 255) if flags else (120, 220, 120), 2 if flags else 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(cv2.cvtColor(np.vstack([img, band]), cv2.COLOR_BGR2RGB)))
    writer.close()
    print(f"{len(names)} frames, {len({t for t, _ in dropped})} with a WiLoR hand on the other glove -> {out}")


if __name__ == "__main__":
    main()
