#!/usr/bin/env python3
"""Retarget HaWoR's hands (estimate_hands_hawor.py) onto the Wuji Hands, as retarget_humantouch.py does the mocap.

HaWoR's 21 joints are already in the MediaPipe / OpenPose order the retargeter reads (Benchmark-Genesis'
hawor_to_wuji_qpos.py), in the world (<side>_joints_world: the registered or estimated camera motion).
Frames where HaWoR has no hand are filled by interpolation and the keypoints smoothed
(retarget_humantouch.fill_and_smooth), then each frame is retargeted (retarget_side).

Writes <perception>/hawor/retarget.npz in retarget_humantouch.py's format: hz, frame_index (video frames),
per side <side>_qpos, <side>_palm_pos, <side>_palm_quat (wxyz), <side>_keypoints, <side>_valid.

Run in the `wuji_retargeting` env.
    python scripts/data/retarget_hawor.py --perception data/touchscale/perception/<episode_id>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
from retarget_humantouch import fill_and_smooth, retarget_side, tip_error_mm  # noqa: E402
from tv2s_env.robots.wuji_model import DEFAULT_WUJI_DIR, WujiHand  # noqa: E402

SIDES = ("left", "right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (hawor/hands.npz)")
    ap.add_argument("--config-dir", default=str(REPO / "configs" / "retargeting"))
    ap.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR)
    ap.add_argument("--smooth-window", type=int, default=5, help="Savitzky-Golay window, in frames")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.wuji_dir))
    root = Path(args.perception)
    d = np.load(root / "hawor" / "hands.npz")
    meta = json.loads((root / "frames.json").read_text())
    hz = meta["video_fps"] / max(1, int(np.median(np.diff(d["frame_index"]))))
    res = dict(hz=np.array(hz), frame_index=d["frame_index"])
    for side in SIDES:
        valid = d[f"{side}_valid"].astype(bool) & np.isfinite(d[f"{side}_joints_world"]).all((1, 2))
        kp = fill_and_smooth(np.nan_to_num(d[f"{side}_joints_world"]), valid, args.smooth_window)
        hand = WujiHand(side)
        out = retarget_side(kp, side, str(Path(args.config_dir) / f"wuji_mano_{side}.yaml"), hand)
        for k, v in out.items():
            res[f"{side}_{k}"] = v.astype(np.float32)
        res[f"{side}_keypoints"], res[f"{side}_valid"] = kp.astype(np.float32), valid
        err = tip_error_mm(hand, out, kp)
        print(f"{side}: valid {valid.mean():.0%}, tip err (mm) " + " ".join(f"{e:.0f}" for e in err))
    np.savez_compressed(root / "hawor" / "retarget.npz", **res)
    print(f"{len(d['frame_index'])} frames @ {hz:g} Hz -> {root / 'hawor' / 'retarget.npz'}")


if __name__ == "__main__":
    main()
