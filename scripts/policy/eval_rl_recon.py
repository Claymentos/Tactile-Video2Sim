#!/usr/bin/env python3
"""Evaluate an rl_recon run (env rl_recon, rl_recon_env.py) as Benchmark-Genesis' rl_recon/train.py does.

Each checkpoint is rolled out deterministically, early termination and assist off, on --n-envs identical envs
(GPU contact solving is not bit-exact, so they diverge: the spread is reported), over the whole clip from frame 0,
and in hand: from the first in-hand start frame with the object at its pose in the reference palm. Per rollout:
contact precision / recall / F1 against the glove, palm / keypoint / human-keypoint errors, the object's depth-aware
and 3D errors, its median offset across the camera ray and long-axis error, its highest lift, and the first frame
that would have terminated it (`held_fraction`: envs that never would). The best checkpoints by mean reward are
copied to model_best.pt (from frame 0) and model_best_inhand.pt (in hand), as rl_recon's.

    python scripts/policy/eval_rl_recon.py --exp_name screwdriver_v1 [--checkpoint 500 | --all] [--n-envs 32]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, create_env, latest_run_dir, load_config  # noqa: E402

KEYS = ("tac_tp", "tac_sim", "tac_real", "e_wrist", "e_kp", "e_human", "e_obj_depth", "e_obj", "e_obj_across", "e_axis",
        "obj_height")


@torch.no_grad()
def rollout(env, policy, mode: str) -> dict:
    """rl_recon/train.py evaluate(): per-env summaries of one deterministic rollout, then their mean and spread."""
    env._args = env._args.model_copy(update={"eval_start": mode})
    env.disable_termination, env.assist_scale = True, 0.0
    env.reset()
    obs, _ = env.get_observations()
    start = int(env.start[0])
    rows, rewards = {k: [] for k in KEYS}, []
    for _ in range(env.T - 1 - start):
        obs, _, reward, _, _, info = env.step(policy(obs, deterministic=True)[0])
        for k in KEYS:
            rows[k].append(info["metrics"][k].cpu().numpy())
        rows.setdefault("e_wrist_max", []).append(env._metrics["e_wrist_h"].amax(-1).cpu().numpy())
        rows.setdefault("e_kp_max", []).append(env._metrics["e_kp_h"].amax(-1).cpu().numpy())
        rows.setdefault("e_obj_task_max", []).append(env._metrics["e_obj_task"].amax(-1).cpu().numpy())
        rewards.append(reward[:, 0].cpu().numpy())
    R = {k: np.stack(v) for k, v in rows.items()}                                         # (S, B)
    a = env._args
    per_env = []
    for j in range(env.num_envs):
        tp, ns, nr = R["tac_tp"][:, j].sum(), R["tac_sim"][:, j].sum(), R["tac_real"][:, j].sum()
        p, r = tp / max(ns, 1), tp / max(nr, 1)
        fail = np.flatnonzero((R["e_wrist_max"][:, j] > a.term_wrist) | (R["e_kp_max"][:, j] > a.term_kp)
                              | (R["e_obj_task_max"][:, j] > a.term_obj))
        per_env.append(dict(reward=float(np.mean([x[j] for x in rewards])), tac_precision=p, tac_recall=r,
                            tac_f1=2 * p * r / max(p + r, 1e-9), e_wrist_cm=R["e_wrist"][:, j].mean() * 100,
                            e_kp_mm=R["e_kp"][:, j].mean() * 1e3, e_human_mm=R["e_human"][:, j].mean() * 1e3,
                            e_obj_cm=R["e_obj_depth"][:, j].mean() * 100, e_obj_3d_cm=R["e_obj"][:, j].mean() * 100,
                            obj_max_height_cm=R["obj_height"][:, j].max() * 100,
                            e_axis_deg=float(np.degrees(np.median(R["e_axis"][:, j]))),
                            e_obj_across_cm=float(np.median(R["e_obj_across"][:, j]) * 100),
                            first_failure=int(fail[0]) + 1 if len(fail) else len(rewards) + 1))
    out = {k: float(np.mean([s[k] for s in per_env])) for k in per_env[0]}
    for k in ("tac_f1", "tac_precision", "tac_recall", "e_axis_deg", "e_obj_across_cm", "e_kp_mm"):
        out[k + "_std"] = float(np.std([s[k] for s in per_env]))
    out.update(held_fraction=float(np.mean([s["first_failure"] == len(rewards) + 1 for s in per_env])),
               n_envs=env.num_envs, start_frame=start)
    return out


def line(tag: str, s: dict) -> str:
    return (f"{tag:>28s} reward {s['reward']:.3f} | contact P {s['tac_precision']:.2f} R {s['tac_recall']:.2f} "
            f"F1 {s['tac_f1']:.2f} +-{s['tac_f1_std']:.2f} | wrist {s['e_wrist_cm']:.1f} cm kp {s['e_kp_mm']:.0f} mm "
            f"human {s['e_human_mm']:.0f} mm | object {s['e_obj_cm']:.1f} cm (across {s['e_obj_across_cm']:.1f}), axis "
            f"{s['e_axis_deg']:.0f} deg, lift {s['obj_max_height_cm']:.1f} cm | held {s['held_fraction'] * 100:.0f}%")


def main() -> None:
    from tv2s_agent.algos.config.schema import PPOArgs
    from tv2s_agent.algos.ppo import PPO
    from tv2s_env.sim.envs.config.schema import load_env_args

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--checkpoint", default=None, help="a checkpoint number or path (default: the latest)")
    ap.add_argument("--all", action="store_true", help="every checkpoint; copy the best ones")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run = latest_run_dir(REPO / "logs" / args.exp_name)
    env_args = load_env_args(run / "configs" / "env_args.json")
    if env_args.env_name != "RlReconEnv":
        raise SystemExit(f"{run}: env {env_args.env_name}, not an rl_recon run")
    ckpts = sorted((run / "checkpoints").glob("checkpoint_*.pt"))
    if args.checkpoint:
        p = Path(args.checkpoint)
        ckpts = [p if p.exists() else run / "checkpoints" / f"checkpoint_{int(args.checkpoint):05d}.pt"]
    elif not args.all:
        ckpts = ckpts[-1:]
    env = create_env(env_args, args.n_envs, args.device, eval_mode=True)
    ppo = PPO(env, load_config(PPOArgs, run / "configs" / "algo_cfg.json"), torch.device(args.device))
    best = {"first": (-1e9, None), "in_hand": (-1e9, None)}
    out_dir = run / "eval_rl_recon"
    out_dir.mkdir(exist_ok=True)
    for ck in ckpts:
        ppo.load(ck, load_optimizer=False)
        policy = ppo.get_inference_policy()
        res = {"first": rollout(env, policy, "first")}
        if len(env.rsi_frames):
            res["in_hand"] = rollout(env, policy, "in_hand")
        (out_dir / f"{ck.stem}.json").write_text(json.dumps(res, indent=1))
        print(f"{ck.name}")
        for mode, s in res.items():
            print(line("from frame 0" if mode == "first" else f"in hand from frame {s['start_frame']}", s), flush=True)
            if s["reward"] > best[mode][0]:
                best[mode] = (s["reward"], ck)
    if args.all or len(ckpts) > 1:
        for mode, name in (("first", "model_best.pt"), ("in_hand", "model_best_inhand.pt")):
            if best[mode][1] is not None:
                shutil.copy(best[mode][1], run / "checkpoints" / name)
                print(f"{name}: {best[mode][1].name} (reward {best[mode][0]:.3f})")


if __name__ == "__main__":
    main()
