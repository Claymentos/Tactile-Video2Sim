#!/usr/bin/env python3
"""Rollout videos of a run's checkpoints: a few fixed clips, deterministic, training aids off.

Each frame tiles the clips (one tile per clip). A tile is the 3D view (both Wuji hands, the proxy objects,
the reference hands as translucent orange ghosts) above the glove images of both hands, simulated vs real
(the human glove), with the clip's running contact F1. Clips that ended (success or termination) freeze
on their last frame, with a green / red border.

Writes <run>/videos/rollout_<ckpt>.mp4 and <run>/videos/rollout_<ckpt>.json (per-clip metrics). The
training process uploads new videos to wandb (tv2s_agent.runners.onpolicy_runner.MetricLogger).

    python scripts/policy/render_rollouts.py --exp_name=dextag_v1 --method=dextag --checkpoint=latest
    python scripts/policy/render_rollouts.py --exp_name=dextag_v1 --method=dextag --watch --every=1000
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import REPO, create_env, latest_run_dir, load_config  # noqa: E402

GLOVE_PX = 6          # glove cell -> pixels


def pick_clips(env_args, k: int) -> list[str]:
    """k clips spread over the selected ones (held-out clips when the run trained on the train split)."""
    from tv2s_env.sim.envs.manipulation.bimanual_tactile_env import select_clip_names
    data = env_args.data
    if data.split_mode == "train":
        data = data.model_copy(update={"split_mode": "eval"})
    names = select_clip_names(data)
    idx = np.unique(np.linspace(0, len(names) - 1, min(k, len(names))).round().astype(int))
    return [names[i] for i in idx]


def glove_panel(sim: np.ndarray, real: np.ndarray) -> np.ndarray:
    """(H, 23, 20) simulated and real glove images -> (h, w, 3) uint8: per hand, sim | real, each map
    normalised by its own peak (DexTaG's rendered tactile maps)."""
    import matplotlib.cm as cm

    tiles = []
    for h in range(sim.shape[0]):
        row = []
        for img in (sim[h], real[h]):
            norm = img / max(float(img.max()), 1e-6)
            rgb = (cm.inferno(norm)[..., :3] * 255).astype(np.uint8)
            rgb = rgb.repeat(GLOVE_PX, 0).repeat(GLOVE_PX, 1)
            row.append(np.pad(rgb, ((2, 2), (2, 2), (0, 0)), constant_values=255))
        tiles.append(np.concatenate(row, 1))
    return np.concatenate(tiles, 1)


def put_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((6, 4 + 12 * i), line, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def add_border(tile: np.ndarray, color, width: int = 4) -> np.ndarray:
    tile = tile.copy()
    tile[:width], tile[-width:], tile[:, :width], tile[:, -width:] = color, color, color, color
    return tile


def compose_tile(rgb: np.ndarray, glove: np.ndarray, lines: list[str]) -> np.ndarray:
    w = rgb.shape[1]
    g = glove
    if g.shape[1] != w:                                # centre the glove strip under the render
        pad = max(0, w - g.shape[1])
        g = np.pad(g, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), constant_values=255)[:, :w]
    label = np.full((14, w, 3), 255, np.uint8)
    return np.concatenate([put_text(rgb, lines), g, label], 0)


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    out = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


@torch.no_grad()
def render_checkpoint(env, ppo, ckpt: Path, out_dir: Path, fps: float, cols: int, max_frames: int) -> dict:
    import imageio.v2 as imageio

    ppo.load(ckpt, load_optimizer=False)
    policy = ppo.get_inference_policy()
    B = env.num_envs
    env.reset()
    obs, _ = env.get_observations()
    running = np.ones(B, bool)
    tp, ns, nr = np.zeros(B), np.zeros(B), np.zeros(B)
    steps, success = np.zeros(B, int), np.zeros(B, bool)
    last = [None] * B
    frames = []

    def tiles(idx, lines_of):
        rgb = env.render_frames()
        sim, real = env.tactile_map.cpu().numpy(), env.target_tactile_map.cpu().numpy()
        for i in idx:
            last[i] = compose_tile(rgb[i], glove_panel(sim[i], real[i]), lines_of(i))

    # first frame: every clip has a tile even if its episode ends on the first step
    tiles(range(B), lambda i: [f"{env.clip_names[int(env.env_clip[i])]}  t={int(env.t[i])}", "", ""])
    frames.append(grid(last, cols))
    for _ in range(max_frames):
        if not running.any():
            break
        action, _ = policy(obs, deterministic=True)
        obs, _, _, terminated, truncated, info = env.step(action)
        m = {k: v.cpu().numpy() for k, v in info["metrics"].items() if k in ("tac_tp", "tac_sim", "tac_real")}
        done = (terminated | truncated)[:, 0].cpu().numpy()
        succ = truncated[:, 0].cpu().numpy() | info["termination"].get("succeeded", torch.zeros_like(truncated[:, 0])).cpu().numpy()
        r = running.astype(float)
        tp += m["tac_tp"] * r
        ns += m["tac_sim"] * r
        nr += m["tac_real"] * r
        steps += running
        # envs that finished this step were reset inside step(): keep their previous frame
        alive = running & ~done
        if alive.any():
            def lines(i):
                p, rc = tp[i] / max(ns[i], 1), tp[i] / max(nr[i], 1)
                f1 = 2 * p * rc / max(p + rc, 1e-9)
                return [f"{env.clip_names[int(env.env_clip[i])]}  t={int(env.t[i])}",
                        f"contact F1 {f1:.2f}  P {p:.2f}  R {rc:.2f}", "glove: L sim | L real | R sim | R real"]
            tiles(np.flatnonzero(alive), lines)
        for i in np.flatnonzero(running & done):
            success[i] = bool(succ[i])
            if last[i] is not None:
                last[i] = add_border(last[i], (40, 180, 60) if success[i] else (220, 50, 50))
        running &= ~done
        frames.append(grid(last, cols))
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"rollout_{ckpt.stem}.mp4"
    tmp = video.with_suffix(".tmp.mp4")
    imageio.mimwrite(tmp, frames, fps=fps, macro_block_size=8, quality=7)
    clips = []
    for i in range(B):
        p, rc = tp[i] / max(ns[i], 1), tp[i] / max(nr[i], 1)
        clips.append(dict(clip=env.clip_names[int(env.env_clip[i])], steps=int(steps[i]), success=bool(success[i]),
                          tac_f1=2 * p * rc / max(p + rc, 1e-9), tac_precision=p, tac_recall=rc))
    meta = dict(checkpoint=ckpt.name, iteration=int(ckpt.stem.split("_")[-1]) if ckpt.stem[-1].isdigit() else -1,
                video=video.name, clips=clips)
    video.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    tmp.rename(video)                                  # the trainer only picks up complete .mp4 files
    print(f"video: {video} ({len(frames)} frames)", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--method", required=True, choices=["dextag", "recon"])
    ap.add_argument("--checkpoint", default="latest", help="'latest', 'best' (model_best.pt) or a path")
    ap.add_argument("--watch", action="store_true", help="render every new checkpoint until training ends")
    ap.add_argument("--every", type=int, default=0, help="--watch: only checkpoints whose iteration is a multiple")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--res", default="480x360")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tv2s_agent.algos.config.schema import PPOArgs
    from tv2s_agent.algos.ppo import PPO
    from tv2s_env.sim.envs.config.schema import DexTaGEnvArgs, ReconEnvArgs

    exp_dir = REPO / "logs" / args.exp_name
    while True:                                        # --watch may start before the first checkpoint
        try:
            run_dir = latest_run_dir(exp_dir)
            break
        except FileNotFoundError:
            if not args.watch:
                raise
            time.sleep(args.poll)
    env_cls = DexTaGEnvArgs if args.method == "dextag" else ReconEnvArgs
    env_args = load_config(env_cls, run_dir / "configs" / "env_args.json")
    clips = pick_clips(env_args, args.clips)
    env_args = env_args.model_copy(update={"data": env_args.data.model_copy(update={"clips": clips, "max_clips": 0,
                                                                                     "split_mode": "all"})})
    algo = load_config(PPOArgs, run_dir / "configs" / "algo_cfg.json")
    w, h = map(int, args.res.split("x"))
    env = create_env(env_args, num_envs=len(clips), device=args.device, eval_mode=True, render={"res": (w, h)})
    ppo = PPO(env, algo, torch.device(args.device))
    print(f"rendering {args.exp_name}: clips {clips}", flush=True)

    if not args.watch:
        ck = {"latest": None, "best": run_dir / "checkpoints" / "model_best.pt"}.get(args.checkpoint, Path(args.checkpoint))
        ck = ck or sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))[-1]
        render_checkpoint(env, ppo, ck, run_dir / "videos", env.hz, args.cols, args.max_frames)
        return
    done = set()
    while True:
        run_dir = latest_run_dir(exp_dir)              # a resumed training writes a new run folder
        ckpts = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
        todo = [c for c in ckpts if c not in done and (not args.every or int(c.stem.split("_")[-1]) % args.every == 0)]
        finished = (run_dir / "TRAINING_DONE").exists()
        if finished and ckpts and ckpts[-1] not in done:
            todo = sorted(set(todo) | {ckpts[-1]})
        for ck in todo[-1:]:                           # newest only: never fall behind training
            time.sleep(5)                              # let torch.save finish
            render_checkpoint(env, ppo, ck, run_dir / "videos", env.hz, args.cols, args.max_frames)
        done.update(todo)
        if finished and not todo:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
