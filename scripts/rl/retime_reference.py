#!/usr/bin/env python3
"""A reference slowed down where a Franka Panda cannot follow it: data/references/<name>_retimed/.

The arm's joint reference (franka_wuji.place_arm, the IK the env runs with --env.arm.enabled=True) asks, on
pick_up_hammer/095958 (frames 189-207, the wrist at 1.8 m/s), ~3x the Panda's joint speeds; with its torques capped
(87 / 12 N m) the arm lags 3 cm there and the hand drops the hammer the floating hand keeps. So the clip's time is
stretched where it is too fast and only there:

  slowdown  s(t) = max(1, max_j |dq_j(t)| / (--max-speed-frac * vmax_j)) up to 1 s after the last glove contact
            (after it the IK may flip: not the manipulation), widened (max filter, --widen frames) and
            smoothed (Gaussian, --smooth frames) so that the clip slows down before the fast motion and speeds up after it
  time map  new frame k plays the original time u_k, u_{k+1} = u_k + 1 / s(u_k): the clip gets longer, its fps stays
  arrays    every per-frame array resampled at u: floats linearly (rotation matrices re-orthonormalised), booleans,
            integers and strings from the nearest frame; grasp_frames (frame indices) mapped onto the new frames;
            source_frame (T') keeps u for the videos' real image, retime_slowdown (T') keeps s

The floating hand and the arm both train on the retimed reference, so their comparison is on the same motion.

Run in the `genesis` env:
    python scripts/rl/retime_reference.py --reference data/references/<name>/reference.npz [--max-speed-frac 0.5]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def arm_joint_speeds(z: dict, fps: float) -> tuple[np.ndarray, int]:
    """(|dq| / vmax (T, 7) of the arm's IK reference, placed as the env places it; the manipulation's end: 1 s after
    the last glove contact, the frame the placement is scored to)."""
    from tv2s_env.sim.envs.config.schema import ArmArgs
    from tv2s_env.sim.robots.franka_wuji import ARM_VELOCITY, place_arm

    n_contact = np.asarray(z["real_contact"]).sum(-1)
    touch = np.flatnonzero(n_contact >= 5)
    T = len(z["wrist_pos"])
    end = min(T, int(touch[-1]) + int(fps)) if len(touch) else T
    pl = place_arm(ArmArgs(enabled=True), np.asarray(z["wrist_pos"]), np.asarray(z["wrist_rot"]), np.asarray(z["cam_pos"]),
                   np.asarray(z["obj_pos"])[:1], end)
    dq = np.gradient(pl["q"], axis=0) * fps
    return np.abs(dq) / ARM_VELOCITY[None], end


def time_map(ratio: np.ndarray, frac: float, widen: int, smooth: float) -> tuple[np.ndarray, np.ndarray]:
    """(u (T',) the original time of each new frame, s (T,) the slowdown per original frame)."""
    s = np.maximum(1.0, ratio / frac)
    s = maximum_filter1d(s, size=2 * widen + 1)
    s = np.maximum(1.0, gaussian_filter1d(s, smooth)) if smooth > 0 else s
    T, u = len(s), [0.0]
    while u[-1] < T - 1:
        u.append(u[-1] + 1.0 / float(np.interp(u[-1], np.arange(T), s)))
    u[-1] = T - 1.0
    return np.asarray(u), s


def resample(v: np.ndarray, u: np.ndarray, nearest: bool = False) -> np.ndarray:
    T = v.shape[0]
    if nearest or v.dtype.kind in "biuUSO":                     # booleans, integers, strings: nearest frame
        return v[np.clip(np.round(u).astype(int), 0, T - 1)]
    i0 = np.clip(np.floor(u).astype(int), 0, T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (u - i0).reshape((-1,) + (1,) * (v.ndim - 1))
    out = (1 - w) * v[i0] + w * v[i1]
    if v.shape[-2:] == (3, 3):                                  # rotations: back onto SO(3)
        U, _, Vt = np.linalg.svd(out)
        d = np.sign(np.linalg.det(U @ Vt))
        U[..., :, -1] *= d[..., None]
        out = U @ Vt
    return out.astype(v.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="reference.npz (prepare_reference.py)")
    ap.add_argument("--out", default=None, help="the retimed reference's folder (default: <folder>_retimed)")
    ap.add_argument("--max-speed-frac", type=float, default=0.7,
                    help="the joint speed asked at most, as a fraction of the Panda's datasheet speed: the env places "
                    "the arm again on the retimed clip and its IK differs (pick_up_hammer/095958: 1.0 gives 1.77x "
                    "after it, 0.7 gives 0.81x)")
    ap.add_argument("--widen", type=int, default=5, help="frames the slowdown is spread over before and after")
    ap.add_argument("--smooth", type=float, default=3.0, help="Gaussian smoothing of the slowdown (frames)")
    args = ap.parse_args()
    src = Path(args.reference)
    out = Path(args.out) if args.out else src.parent.with_name(src.parent.name + "_retimed")
    z = dict(np.load(src, allow_pickle=True))
    fps, T = float(z["fps"]), len(z["wrist_pos"])

    ratio, end = arm_joint_speeds(z, fps)
    ratio = ratio.max(1)
    ratio[end:] = 0.0              # after the manipulation (pick_up_hammer/095958: the IK flips at frame 388, 7x)
    u, s = time_map(ratio, args.max_speed_frac, args.widen, args.smooth)
    slow = np.flatnonzero(s > 1.01)
    print(f"{src}: {T} frames, arm joint speed at most {ratio.max():.2f} x the Panda's (frame {ratio.argmax()}); "
          f"slowed where above {args.max_speed_frac}: {len(slow)} frames, slowdown up to {s.max():.2f} "
          f"-> {len(u)} frames ({len(u) / fps:.1f} s instead of {T / fps:.1f} s)")

    for k, v in list(z.items()):
        if k == "grasp_frames":
            z[k] = np.unique(np.searchsorted(u, np.asarray(v, float)))
        elif k == "obj_kf":            # a keyframe stays one frame: the new frame nearest to it, not every copy of it
            kf = np.zeros(len(u), bool)
            kf[np.abs(u[:, None] - np.flatnonzero(v)[None]).argmin(0)] = True
            z[k] = kf
        elif isinstance(v, np.ndarray) and v.ndim and v.shape[0] == T and not k.startswith("grasp_candidates"):
            # the measured object poses stay measurements: each new frame takes its nearest frame's (its keyframe flag too)
            z[k] = resample(v, u, nearest=k in ("obj_pos_measured", "obj_rot_measured"))
    z["source_frame"] = u
    z["retime_slowdown"] = np.interp(u, np.arange(T), s)

    out.mkdir(parents=True, exist_ok=True)
    for key in ("object_visual", "object_collision"):
        p = Path(str(z[key]))
        shutil.copy(p, out / p.name)
        z[key] = np.array(str((out / p.name).resolve()))
    np.savez(out / "reference.npz", **z)
    (out / "retime.json").write_text(json.dumps(dict(source=str(src.resolve()), frames=T, frames_retimed=len(u),
                                                     max_speed_frac=args.max_speed_frac, peak_ratio=float(ratio.max()),
                                                     max_slowdown=float(s.max()), slowed_frames=int(len(slow))), indent=1))
    check, end2 = arm_joint_speeds(z, fps)
    print(f"retimed: arm joint speed at most {check[:end2].max():.2f} x the Panda's up to the manipulation's end "
          f"(the IK placed again) -> {out / 'reference.npz'}")


if __name__ == "__main__":
    main()
