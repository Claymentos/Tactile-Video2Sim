#!/usr/bin/env python3
"""Retarget both HumanTouch hand skeletons onto the Wuji Hands (wuji_retargeting), resampled to --hz.

Follows Benchmark-Genesis' hawor_to_wuji_qpos.py: the 25-joint skeleton is reduced to the 21 MediaPipe
keypoints, invalid frames are filled and the keypoints smoothed, each frame is retargeted
(configs/retargeting/wuji_mano_<side>.yaml), and the palm_link frame is put on the human wrist with the
retargeter's own wrist frame (estimate_frame_from_hand_points @ OPERATOR2MANO).

Reads <data>/episodes/*.npz (extract_humantouch.py), writes <data>/retarget/episode_XXXXXX.npz with
  hz, frame_index (60 Hz source frame of every output frame), and per side:
  <side>_qpos (T, 20) in WujiHand.dof_names order, <side>_palm_pos (T, 3), <side>_palm_quat (T, 4) wxyz,
  <side>_keypoints (T, 21, 3) MediaPipe keypoints (world), <side>_valid (T,)

    python scripts/data/retarget_humantouch.py --data data/humantouch --hz 30
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.humantouch import SIDES, mediapipe_keypoints  # noqa: E402
from tv2s_env.robots.wuji_model import DEFAULT_WUJI_DIR, WujiHand  # noqa: E402

MEDIAPIPE_TIPS = [4, 8, 12, 16, 20]
WUJI_TIPS = ["thumb", "index", "middle", "ring", "little"]


def fill_and_smooth(kp: np.ndarray, valid: np.ndarray, smooth_window: int) -> np.ndarray:
    """Interpolate invalid frames (held at the ends), then Savitzky-Golay smoothing."""
    from scipy.signal import savgol_filter

    kp = kp.astype(np.float64).copy()
    known = np.flatnonzero(valid)
    if len(known) == 0:
        return kp
    t = np.arange(len(kp))
    for j in range(kp.shape[1]):
        for a in range(3):
            kp[:, j, a] = np.interp(t, known, kp[known, j, a])
    window = min(smooth_window, len(kp) - (1 - len(kp) % 2))
    if window >= 5:
        kp = savgol_filter(kp, window, 2, axis=0)
    return kp


def retarget_side(kp: np.ndarray, side: str, config: str, hand: WujiHand) -> dict[str, np.ndarray]:
    from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points
    from wuji_retargeting.retarget import Retargeter

    retargeter = Retargeter.from_yaml(config, hand_side=side)
    operator2mano = OPERATOR2MANO_LEFT if side == "left" else OPERATOR2MANO_RIGHT
    names = list(retargeter.optimizer.robot.dof_joint_names)
    order = [names.index(n) for n in hand.dof_names]
    T = len(kp)
    qpos, pos, quat = np.zeros((T, len(order))), np.zeros((T, 3)), np.zeros((T, 4))
    for t in range(T):
        q = np.asarray(retargeter.retarget(kp[t]))
        qpos[t] = q[order]
        R = estimate_frame_from_hand_points(kp[t] - kp[t, 0]) @ operator2mano
        xyzw = Rotation.from_matrix(R).as_quat()
        pos[t], quat[t] = kp[t, 0], np.r_[xyzw[3], xyzw[:3]]
    # quaternion sign continuity
    for t in range(1, T):
        if quat[t] @ quat[t - 1] < 0:
            quat[t] *= -1
    return dict(qpos=qpos, palm_pos=pos, palm_quat=quat)


def tip_error_mm(hand: WujiHand, out: dict, kp: np.ndarray) -> np.ndarray:
    base = np.tile(np.eye(4), (len(kp), 1, 1))
    base[:, :3, :3] = Rotation.from_quat(out["palm_quat"][:, [1, 2, 3, 0]]).as_matrix()
    base[:, :3, 3] = out["palm_pos"]
    fk = hand.fk(out["qpos"], base)
    tips = np.stack([fk[hand.link_name(f, "tip_link")][:, :3, 3] for f in WUJI_TIPS], 1)
    return 1000 * np.median(np.linalg.norm(tips - kp[:, MEDIAPIPE_TIPS], axis=-1), 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--config-dir", default=str(REPO / "configs" / "retargeting"))
    ap.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR)
    ap.add_argument("--smooth-window", type=int, default=9, help="Savitzky-Golay window, in 60 Hz frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.wuji_dir))
    data = Path(args.data)
    out_dir = data / "retarget"
    out_dir.mkdir(parents=True, exist_ok=True)
    hands = {s: WujiHand(s) for s in SIDES}

    episodes = sorted((data / "episodes").glob("episode_*.npz"))
    if not episodes:
        raise SystemExit(f"no episodes in {data / 'episodes'}: run extract_humantouch.py first")
    for path in episodes:
        dst = out_dir / path.name
        if dst.exists() and not args.overwrite:
            print(f"{path.stem}: exists")
            continue
        ep = np.load(path)
        fps = float(ep["fps"])
        n = len(ep["head_pose"])
        frame_index = np.unique(np.round(np.arange(0, n, fps / args.hz)).astype(int).clip(max=n - 1))
        res = dict(hz=np.array(args.hz), frame_index=frame_index, episode_index=ep["episode_index"])
        msg = []
        for side in SIDES:
            valid = ep[f"{side}_skeleton_valid"].astype(bool)
            if valid.ndim > 1:
                valid = valid.reshape(len(valid), -1).all(1)
            kp60 = mediapipe_keypoints(ep[f"{side}_skeleton"])
            valid &= np.isfinite(kp60).all(axis=(1, 2))
            kp = fill_and_smooth(np.nan_to_num(kp60), valid, args.smooth_window)[frame_index]
            out = retarget_side(kp, side, str(Path(args.config_dir) / f"wuji_mano_{side}.yaml"), hands[side])
            for k, v in out.items():
                res[f"{side}_{k}"] = v.astype(np.float32)
            res[f"{side}_keypoints"] = kp.astype(np.float32)
            res[f"{side}_valid"] = valid[frame_index]
            err = tip_error_mm(hands[side], out, kp)
            msg.append(f"{side}: valid {valid.mean():.0%}, tip err (mm) " + " ".join(f"{e:.0f}" for e in err))
        np.savez_compressed(dst, **res)
        print(f"{path.stem}: {len(frame_index)} frames @ {args.hz:g} Hz | " + " | ".join(msg), flush=True)


if __name__ == "__main__":
    main()
