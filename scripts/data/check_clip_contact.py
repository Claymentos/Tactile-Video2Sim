#!/usr/bin/env python3
"""Gate before training on an episode clip: can the simulated glove read the real glove's contacts at all?

Kinematic replay in Genesis (no physics, one environment) of the clip's holds: on every --every-th held frame,
each active hand at its reference and its scheduled object placed
  video  at the clip's object pose (build_episode_clip.py --objects guide: the CoTracker + depth trajectory);
  grasp  at reference palm x hold_offset (the grasp pose in the palm that explains the glove, grasp_pose.py:
         the target rl_recon's assist carries the object to),
and the contact F1 of the simulated taxels (distance < the probe radius) against the clip's glove.
The policy can only learn the tactile reward if the grasp replay reads contacts: below --min-f1 the script
exits with an error (launch/egotouch_video.sh then does not train). It also fails when the video trajectory is
more than --max-guide-cm from the grasp pose while held (the reward's guide would be unreachable: a bad
camera / mocap registration). The video F1 is printed for information
(the monocular track is often a few cm off the mocap hand).

    python scripts/data/check_clip_contact.py --clip-dir data/episodes/et_charger_151150
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "policy"))
from common import apply_overrides_generic, create_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--env-name", default="bimanual_recon")
    ap.add_argument("--every", type=int, default=2, help="replay every n-th held frame")
    ap.add_argument("--min-f1", type=float, default=0.1, help="least grasp-replay contact F1 to pass")
    ap.add_argument("--max-guide-cm", type=float, default=15.0,
                    help="most median distance between the video trajectory and the grasp pose while held (rl_recon "
                         "terminates at 20 cm): beyond it the video and the mocap hands are not registered together")
    ap.add_argument("--out", default=None, help="JSON summary (default: <clip-dir>/contact_check.json)")
    args = ap.parse_args()

    from tv2s_env.sim.envs.config.registry import EnvArgsRegistry

    env_args = apply_overrides_generic(EnvArgsRegistry[args.env_name], {
        "env.data.clip_dir": args.clip_dir, "env.active_hands": "auto", "env.train_start": "first"}, prefixes=("env.",))
    env = create_env(env_args, num_envs=1, eval_mode=True)
    dev = env.device
    envs = torch.arange(1, device=dev)
    radius = env._args.tactile.probe_radius
    sched, off = env.ref["schedule"][0], env.ref["hold_offset"][0]                 # (T, H), (T, H, 4, 4)
    counts = {m: np.zeros(3, int) for m in ("video", "grasp")}                     # sim, real, true positives
    frames, gaps = 0, []
    for h, side in enumerate(env.sides):
        held = torch.isfinite(off[:, h]).all(-1).all(-1) & (sched[:, h] >= 0)
        for t in torch.nonzero(held).flatten().tolist()[:: args.every]:
            tt = torch.tensor([t], device=dev)
            env._reset_hands_and_objects(envs, tt, env.ref["holding"][env.env_clip, tt])
            env.t[:] = t
            for mode in ("video", "grasp"):
                if mode == "grasp":
                    P = torch.eye(4, device=dev)
                    P[:3, :3], P[:3, 3] = env.ref["wrist_rot"][0, t, h], env.ref["wrist_pos"][0, t, h]
                    T = P @ off[t, h]
                    gaps.append(float((T[:3, 3] - env.ref["all_obj_pos"][0, t, int(sched[t, h])]).norm()))
                    q = Rotation.from_matrix(T[:3, :3].cpu().numpy()).as_quat()[[3, 0, 1, 2]]
                    e = env.scene_entities[int(sched[t, h])]
                    e.set_pos(T[:3, 3][None], envs_idx=envs, zero_velocity=True)
                    e.set_quat(torch.tensor(q, dtype=torch.float32, device=dev)[None], envs_idx=envs, zero_velocity=True)
                env.update_buffers()
                sim = (env.tactile_distance[0, h] < radius).cpu().numpy()
                real = env.target_contact[0, h].cpu().numpy().astype(bool)
                counts[mode] += [sim.sum(), real.sum(), (sim & real).sum()]
            frames += 1
    if frames == 0:
        raise SystemExit("no held frame with a grasp pose (hold_offsets) in the clip: build it with --objects guide")
    # physics: from frame 0, hands on the reference, every object must stay on the table until its first grasp
    # (an inertia inconsistent with the mass once made a thin object fall through the ground)
    first = int(torch.nonzero((sched >= 0).any(-1) & torch.isfinite(off).all(-1).all(-1).any(-1)).min())
    env._reset_hands_and_objects(envs, torch.zeros(1, dtype=torch.long, device=dev), env.ref["holding"][env.env_clip, :1])
    env.t[:] = 0
    env.update_buffers()
    low = []
    for _ in range(max(first - 1, 1)):
        env.step(torch.zeros((1, env.action_dim), device=dev))
        low.append(float(torch.stack([e.get_pos()[0, 2] for e in env.scene_entities]).min()))
    rest_low = min(low)
    f1 = {m: 2 * c[2] / max(c[0] + c[1], 1) for m, c in counts.items()}
    gap = float(np.median(gaps)) * 100
    summary = dict(frames=frames, min_f1=args.min_f1, guide_gap_cm=gap, rest_min_z_cm=rest_low * 100, **{f"{m}_f1": float(f1[m]) for m in f1},
                   **{f"{m}_counts": dict(sim=int(c[0]), real=int(c[1]), tp=int(c[2])) for m, c in counts.items()})
    Path(args.out or Path(args.clip_dir) / "contact_check.json").write_text(json.dumps(summary, indent=1))
    print(f"contact replay over {frames} held frames: grasp pose F1 {f1['grasp']:.3f} (sim {counts['grasp'][0]}, "
          f"real {counts['grasp'][1]}), video trajectory F1 {f1['video']:.3f}; video trajectory vs grasp pose: median "
          f"{gap:.1f} cm")
    print(f"objects at rest before the first grasp (frame {first}): lowest centre {rest_low * 100:+.1f} cm")
    if rest_low < -0.02:
        raise SystemExit(f"an object falls through the table before its grasp (centre at {rest_low * 100:+.1f} cm): "
                         "not training on it")
    if f1["grasp"] < args.min_f1:
        raise SystemExit(f"grasp replay F1 {f1['grasp']:.3f} < {args.min_f1}: the simulated glove cannot match this "
                         "demonstration's contacts; not training on it")
    if gap > args.max_guide_cm:
        raise SystemExit(f"the video trajectory is {gap:.1f} cm (median) from the grasp pose while held, > "
                         f"{args.max_guide_cm} cm: camera and mocap hands are not registered; not training on it")


if __name__ == "__main__":
    main()
