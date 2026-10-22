#!/usr/bin/env python3
"""Watch a whole demonstration in simulation: an rl_recon policy replaying it, or the reference itself.

  --exp NAME        a trained rl_recon run (scripts/policy/run_ppo_recon.py), deterministic, assist and termination
                    off, from frame 0 (--start first) or in hand (--start in_hand, the first in-hand start frame):
                    pure physics (a dropped object stays dropped)
  --reference NPZ   the target itself (kinematic, no policy): the hand and the object put on the reference frame by
                    frame (scripts/reconstruction/replay_raw_genesis.py shows the same in the interactive viewer)
Each video frame: the real head-camera frame | the simulation seen from the recorded camera | a view of the table,
with the reference as translucent ghosts (orange hand, green object); a band gives the time, the errors and the
glove contact F1.

Writes <out>/<tag>.mp4 (H.264), <tag>.npz (the simulated trajectory: hand joints and palm pose, object pose, the
simulated glove, the per-frame errors -- the reconstructed episode) and <tag>.json (a summary).

    python scripts/policy/visualize_episode.py --exp et_screwdriver [--checkpoint best|best_inhand|latest|N] [--start in_hand]
    python scripts/policy/visualize_episode.py --reference data/references/et_screwdriver/reference.npz
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, create_env, latest_run_dir, load_config  # noqa: E402

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def load_policy(env, algo_args, run: Path, checkpoint: str):
    """The inference policy of a checkpoint: best (model_best.pt), best_inhand, latest, or a number."""
    from tv2s_agent.algos.ppo import PPO
    ck = run / "checkpoints"
    named = {"best": "model_best.pt", "best_inhand": "model_best_inhand.pt"}
    if checkpoint in named and (ck / named[checkpoint]).exists():
        path = ck / named[checkpoint]
    elif checkpoint in ("best", "best_inhand", "latest"):
        path = sorted(ck.glob("checkpoint_*.pt"))[-1]
    else:
        path = ck / f"checkpoint_{int(checkpoint):05d}.pt"
    ppo = PPO(env, algo_args, env.device)
    ppo.load(path, load_optimizer=False)
    return ppo.get_inference_policy(), path


def main() -> None:
    from tv2s_env.sim.envs.config.schema import ReferenceDataArgs, RlReconEnvArgs, load_env_args

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp", help="rl_recon run to replay")
    src.add_argument("--reference", help="play this reference itself (no policy)")
    ap.add_argument("--checkpoint", default="best", help="best, best_inhand, latest, or a number")
    ap.add_argument("--start", choices=("first", "in_hand"), default="first")
    ap.add_argument("--height", type=int, default=300, help="panel height (px)")
    ap.add_argument("--out", default=str(REPO / "outputs" / "episodes"))
    args = ap.parse_args()

    if args.reference:
        env_args = RlReconEnvArgs(data=ReferenceDataArgs(reference=args.reference))
        tag = f"{Path(args.reference).parent.name}_reference"
    else:
        from tv2s_agent.algos.config.schema import PPOArgs
        run = latest_run_dir(REPO / "logs" / args.exp)
        env_args = load_env_args(run / "configs" / "env_args.json").model_copy(update={"eval_start": args.start})
        algo_args = load_config(PPOArgs, run / "configs" / "algo_cfg.json")
        tag = f"{args.exp}_{args.start}"
    ref = np.load(env_args.data.reference, allow_pickle=True)
    W0, H0 = (int(v) for v in ref["image_size"])
    h = args.height
    w = int(round(h * W0 / H0)) // 2 * 2
    env = create_env(env_args, num_envs=1, eval_mode=True, render={"res": (w, h), "fov": float(ref["fov_y"])})
    env.disable_termination = True
    policy = None
    if not args.reference:
        policy, ck = load_policy(env, algo_args, run, args.checkpoint)
        tag += f"_{Path(ck).stem}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(str(out / f"{tag}.mp4"), (3 * w, h + 34), fps=env.hz, codec="libx264",
                                         output_params=["-crf", "26"], macro_block_size=1)
    writer.send(None)
    envs = torch.zeros(1, dtype=torch.long, device=env.device)
    obs, _ = env.get_observations()
    start = int(env.start[0])
    rec = {k: [] for k in ("t", "hand_dof", "palm_pos", "palm_quat", "object_pos", "object_quat", "tactile_pressure",
                           "e_wrist", "e_obj", "tac_f1")}
    frames_dir = Path(str(ref["frames_dir"]))
    for t in range(start, env.T - 1):
        if args.reference:                                    # kinematic: everything on the reference
            env.t[:] = t
            env._reset_hands_and_objects(envs, torch.tensor([t], device=env.device))
            env.update_buffers()
        else:
            with torch.no_grad():
                action, _ = policy(obs, deterministic=True)
            obs = env.step(action)[0]
            env.t[:] = t + 1
        m = env.common_metrics()
        e_w = float(m["e_wrist"][0])
        e_o = float(((env.object_pos - env.target_object_pos).norm(dim=-1) * env.object_valid).amax())
        tp, ns, nr = float(m["tac_tp"][0]), float(m["tac_sim"][0]), float(m["tac_real"][0])
        f1 = 2 * tp / max(ns + nr, 1) if nr > 0 else float("nan")
        for k, v in (("t", t), ("hand_dof", env.hand_dof_pos[0].cpu().numpy()), ("palm_pos", env.base_pos[0].cpu().numpy()),
                     ("palm_quat", env.base_quat[0].cpu().numpy()), ("object_pos", env.all_object_pos[0].cpu().numpy()),
                     ("object_quat", env.all_object_quat[0].cpu().numpy()),
                     ("tactile_pressure", env.tactile_pressure[0].cpu().numpy()), ("e_wrist", e_w), ("e_obj", e_o), ("tac_f1", f1)):
            rec[k].append(v)

        # render: real | from the recorded camera | table view, ghosts on the reference
        tt = int(env.t[0])
        env.set_reference_ghosts()
        T_cam = np.eye(4)
        T_cam[:3, :3], T_cam[:3, 3] = ref["cam_rot"][tt], ref["cam_pos"][tt]
        env.camera.set_pose(transform=T_cam @ CV_TO_GL)
        ego = np.asarray(env.camera.render(rgb=True, force_render=True)[0])
        c = env.base_pos[0].mean(0).cpu().numpy()
        env.camera.set_pose(pos=c + np.array([0.0, -0.55, 0.45]), lookat=c, up=(0, 0, 1))
        side = np.asarray(env.camera.render(rgb=True, force_render=True)[0])
        img = cv2.imread(str(frames_dir / f"{tt:06d}.png"))
        real = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img is not None else np.zeros((h, w, 3), np.uint8)
        row = np.hstack([real, cv2.cvtColor(ego, cv2.COLOR_RGB2BGR), cv2.cvtColor(side, cv2.COLOR_RGB2BGR)])
        band = np.full((34, 3 * w, 3), 25, np.uint8)
        txt = (f"{tt / env.hz:5.1f} s   {'reference (no policy)' if args.reference else tag}   wrist {e_w * 100:4.1f} cm   "
               f"object {e_o * 100:4.1f} cm   contact F1 {f1:.2f}")
        cv2.putText(band, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(row, "real", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(row, "simulation, recorded camera", (w + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(row, "table view (ghosts: reference)", (2 * w + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(cv2.cvtColor(np.vstack([row, band]), cv2.COLOR_BGR2RGB)))
    writer.close()
    arr = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(out / f"{tag}.npz", **arr)
    f1 = arr["tac_f1"][np.isfinite(arr["tac_f1"])]
    summary = dict(tag=tag, start=start, frames=len(arr["t"]), wrist_cm=round(float(np.median(arr["e_wrist"])) * 100, 2),
                   object_cm=round(float(np.median(arr["e_obj"])) * 100, 2), contact_f1=round(float(f1.mean()), 3) if len(f1) else None,
                   object_max_height_cm=round(float(arr["object_pos"][:, 0, 2].max()) * 100, 1))
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), f"-> {out / tag}.mp4")


if __name__ == "__main__":
    main()
