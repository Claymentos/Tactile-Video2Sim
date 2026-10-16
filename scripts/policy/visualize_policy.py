#!/usr/bin/env python3
"""Watch trained policies: replay them on their clips and render one video per clip, the runs side by side.

On top, the human demonstration: HumanTouch's head-camera video over the clip, frame-synchronised with
the simulation (the egocentric camera below sits at the same recorded head pose). Each run is replayed
as in its evaluation (one environment per clip, deterministic actions, training
aids off) with two cameras per environment: an egocentric one at the recorded head pose, looking at the
hands, and a front one. Green spheres are the reference (the retargeted human hand keypoints and the
proxy object's axis); the robot hands and objects are the simulated state. Under the views, per hand:
the glove pressure of the human demonstration, and the contact agreement of the simulated glove with it
(green: both, blue: glove only, red: simulation only). The header gives the frame, the frame's contact
F1, the object error, and how the episode ended. Brown plates: where the objects rest before the
grasp; blue-grey plates: where they were put down.

Videos:   outputs/videos/<name>/<clip>.mp4 (the human video, then the runs stacked top to bottom; a run
          whose episode ended holds its last frame), plus each run's own videos and trajectories (npz)
          under <name>/<exp_name>/. --compose_only rebuilds the stacked videos from those.
    python scripts/policy/visualize_policy.py --exp_names=ep000002_dextag,ep000002_recon
    python scripts/policy/visualize_policy.py --exp_names=ep000002_recon --clips=ep000002_clip03 --checkpoint=500
Interactive Genesis viewer (one run, loops until Ctrl-C):
    python scripts/policy/visualize_policy.py --exp_names=ep000002_recon --clips=ep000002_clip03 --viewer
"""

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from common import REPO, apply_overrides_generic, count_clips, create_env, latest_run_dir, load_config, parse_cli, \
    split_overrides
from tv2s_data.clip_video import head_video_path, read_head_frames  # noqa: E402

MAIN_ARGS = {"exp_names", "checkpoint", "clips", "name", "res", "fps", "viewer", "compose_only", "device"}
GRID_SCALE = 9                                   # glove image pixels per cell
PRESSURE_MAX = 40.0                              # glove colour scale (cleaned units)
REF_COLOR = (0.1, 0.9, 0.3, 0.6)
UP = (0.0, 0.0, 1.0)
PLACE_TOL = 0.06                                 # as evaluate_policy's success
STATUS_KEYS = ("hand_failure", "object_failure", "support_failure", "drift", "error")


# ====================================================================== runs
def load_run(exp_name: str, checkpoint: str | None, clips: str | None):
    """(method, env args, algo args, checkpoint path) of a run; `checkpoint` is a path, a number or None
    (model_best.pt if the run was evaluated with --all_checkpoints, else its last checkpoint)."""
    from tv2s_agent.algos.config.schema import PPOArgs
    from tv2s_env.sim.envs.config.schema import load_env_args

    run_dir = latest_run_dir(REPO / "logs" / exp_name)
    cfg = run_dir / "configs" / "env_args.json"
    method = json.loads(cfg.read_text())["method"]
    env_args = load_env_args(cfg)
    if clips:
        env_args = apply_overrides_generic(env_args, {"env.data.clips": clips}, prefixes=("env.",))
    ckpts = run_dir / "checkpoints"
    if checkpoint is None:
        ck = ckpts / "model_best.pt"
        ck = ck if ck.exists() else sorted(ckpts.glob("checkpoint_*.pt"))[-1]
    elif str(checkpoint).isdigit():
        ck = ckpts / f"checkpoint_{int(checkpoint):05d}.pt"
    else:
        ck = Path(checkpoint)
    return method, env_args, load_config(PPOArgs, run_dir / "configs" / "algo_cfg.json"), ck


def load_policy(env, algo_args, ckpt: Path, device: str):
    from tv2s_agent.algos.ppo import PPO
    ppo = PPO(env, algo_args, torch.device(device))
    ppo.load(ckpt, load_optimizer=False)
    return ppo.get_inference_policy()


# ====================================================================== drawing
def draw_reference(env) -> None:
    """Reference keypoints and object axis of every environment, as debug spheres (offset per env)."""
    scene = env.scene
    scene.clear_debug_objects()
    offset = torch.as_tensor(scene.envs_offset, dtype=torch.float32, device=env.device)          # (B, 3)
    kp = env.target_keypoints + offset[:, None, None]                                          # (B, H, K, 3)
    s = torch.linspace(-1, 1, 5, device=env.device)
    axis = (env.target_object_pos[..., None, :] + s[:, None] * (env.obj_half_length[..., None, None]
                                                                * env.target_object_axis[..., None, :]))
    axis = (axis + offset[:, None, None])[env.object_valid]                                    # (n, 5, 3)
    scene.draw_debug_spheres(kp.reshape(-1, 3), radius=0.006, color=REF_COLOR)
    if len(axis):
        scene.draw_debug_spheres(axis.reshape(-1, 3), radius=0.008, color=REF_COLOR)


def glove_panel(env, i: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(human glove pressure, contact agreement) images of hand h in env i, fingers up."""
    sensor = env.sensors[h]
    assigned = sensor.to_image(torch.ones(1, env.n_channels, device=env.device))[0] > 0
    sim = env.tactile_distance[i, h] < env._args.tactile.probe_radius                          # noqa: SLF001
    real = env.target_contact[i, h]
    img = lambda x: sensor.to_image(x[None].float())[0].cpu().numpy()
    assigned, sim, real = assigned.cpu().numpy(), img(sim) > 0.5, img(real) > 0.5
    pressure = img(env.target_pressure[i, h] * env.target_contact[i, h])

    p = (np.sqrt(np.clip(pressure / PRESSURE_MAX, 0, 1)) * 255).astype(np.uint8)
    p = cv2.cvtColor(cv2.applyColorMap(p, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    p[~assigned] = 35
    agree = np.full(assigned.shape + (3,), 90, np.uint8)
    agree[sim & real] = (40, 200, 70)
    agree[real & ~sim] = (60, 110, 255)
    agree[sim & ~real] = (235, 60, 50)
    agree[~assigned] = 35
    up = lambda x: cv2.resize(np.flipud(x), None, fx=GRID_SCALE, fy=GRID_SCALE, interpolation=cv2.INTER_NEAREST)
    return up(p), up(agree)


def label(img: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    bar = np.zeros((height, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def pad_to(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] >= w:
        return img[:, :w]
    return np.hstack([img, np.zeros((img.shape[0], w - img.shape[1], 3), np.uint8)])


def even(img: np.ndarray) -> np.ndarray:
    return img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]


# ====================================================================== recording
class _Writer:
    """imageio writer that remembers its last frame (to hold it at the end of an episode)."""

    def __init__(self, writer):
        self._w, self.last_frame = writer, None

    def append_data(self, img):
        self.last_frame = img
        self._w.append_data(img)

    def close(self):
        self._w.close()


class Recorder:
    """Renders and writes one video per environment (clip) of one run while the policy is replayed."""

    def __init__(self, env, run_label: str, out_dir: Path, fps: int):
        self.env, self.run_label, self.out_dir = env, run_label, out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        B = env.num_envs
        # one recorded env per clip (Genesis needs 2 envs at least: a single clip gets a spare copy)
        clip_of = env.env_clip.cpu().numpy()
        self.running = np.array([i == np.argmax(clip_of == c) for i, c in enumerate(clip_of)])
        self.writers = [_Writer(imageio.get_writer(out_dir / f"{env.clip_names[clip_of[i]]}.mp4", fps=fps,
                                                   codec="libx264", quality=7, macro_block_size=1))
                        if self.running[i] else None for i in range(B)]
        self.traj: list[dict[str, list]] = [{} for _ in range(B)]
        self.lookat = None
        self.front_offset = None
        self.tp = np.zeros(B)

    def _cameras(self) -> None:
        env = self.env
        mid = env.target_wrist_pos.mean(1)                                                     # (B, 3)
        head = env.ref_at("head_pos", env.t)
        if self.lookat is None:
            self.lookat = mid.clone()
            fwd = mid - head
            fwd[:, 2] = 0
            fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self.front_offset = 0.75 * fwd + torch.tensor([0.0, 0.0, 0.35], device=env.device)
        self.lookat = 0.85 * self.lookat + 0.15 * mid
        for i, (ego, front) in enumerate(env.cameras):
            if self.running[i]:
                ego.set_pose(pos=head[i], lookat=self.lookat[i], up=UP)
                front.set_pose(pos=self.lookat[i] + self.front_offset[i], lookat=self.lookat[i], up=UP)

    def frame(self, status: dict[int, str] | None = None) -> None:
        """Render the current state of every running env (and of the envs that just ended)."""
        env, status = self.env, status or {}
        draw_reference(env)
        self._cameras()
        m = env.common_metrics()
        for i in np.nonzero(self.running)[0]:
            views = [cam.render()[0] for cam in env.cameras[i]]
            top = np.hstack(views)
            panels = []
            for h, side in enumerate(("left", "right")):
                p, a = glove_panel(env, i, h)
                panels += [label(p, f"{side}: glove"), label(a, f"{side}: sim vs glove")]
            gap = np.zeros((panels[0].shape[0], 8, 3), np.uint8)
            bottom = pad_to(np.hstack(sum([[x, gap] for x in panels], [])), top.shape[1])
            legend = label(np.zeros((0, top.shape[1], 3), np.uint8),
                           "green spheres: reference | glove: green both, blue glove only, red sim only")
            tp, ns, nr = (m[k][i].item() for k in ("tac_tp", "tac_sim", "tac_real"))
            f1 = 2 * tp / max(ns + nr, 1) if nr + ns > 0 else float("nan")
            t, start, end = int(env.t[i]), int(env.start[i]), int(env.horizon[i])
            text = (f"{self.run_label} | {env.clip_names[int(env.env_clip[i])]} | frame {t - start}/{end - start}"
                    f" | contact F1 {f1:.2f} | obj err {100 * m['e_obj'][i].item():.1f} cm")
            if i in status:
                text += f" | {status[i]}"
            img = label(np.vstack([top, bottom, legend]), text, height=26)
            self.writers[i].append_data(even(img))
            self._log(i, f1)

    def _log(self, i: int, f1: float) -> None:
        env, tr = self.env, self.traj[i]
        for k, v in dict(t=env.t[i], wrist_pos=env.base_pos[i], target_wrist_pos=env.target_wrist_pos[i],
                         keypoints=env.keypoints[i], target_keypoints=env.target_keypoints[i],
                         object_pos=env.object_pos[i], target_object_pos=env.target_object_pos[i],
                         object_quat=env.object_quat[i], hand_dof_pos=env.hand_dof_pos[i],
                         sim_contact=env.tactile_distance[i] < env._args.tactile.probe_radius,  # noqa: SLF001
                         glove_contact=env.target_contact[i]).items():
            tr.setdefault(k, []).append(v.cpu().numpy())
        tr.setdefault("contact_f1", []).append(f1)

    def on_step(self, terminated: torch.Tensor, truncated: torch.Tensor, infos: dict) -> None:
        """Env step callback: the state is still the one the episode ended in."""
        done = ((terminated | truncated).cpu().numpy()) & self.running
        term = infos.get("termination", {})
        status = {}
        for i in np.nonzero(done)[0]:
            ok = bool(truncated[i]) or bool(term.get("succeeded", torch.zeros_like(terminated))[i])
            why = [k for k in STATUS_KEYS if k in term and bool(term[k][i])]
            if ok:
                e = infos["metrics"]["e_obj_end"][i].item()
                status[i] = f"END OF CLIP, {'objects placed' if e < PLACE_TOL else f'object off by {100 * e:.0f} cm'}"
            else:
                status[i] = "FAILED: " + ",".join(why)
        self.frame(status)
        for i in np.nonzero(done)[0]:
            for _ in range(15):                     # hold the last frame half a second
                self.writers[i].append_data(self.writers[i].last_frame)
            self.running[i] = False

    def close(self) -> list[Path]:
        paths = []
        for i, w in enumerate(self.writers):
            if w is None:
                continue
            w.close()
            name = self.env.clip_names[int(self.env.env_clip[i])]
            np.savez_compressed(self.out_dir / f"{name}.npz", **{k: np.stack(v) for k, v in self.traj[i].items()})
            paths.append(self.out_dir / f"{name}.mp4")
        return paths


@torch.no_grad()
def record_run(exp_name: str, checkpoint, clips, out_dir: Path, res, fps: int, device: str) -> list[Path]:
    method, env_args, algo_args, ckpt = load_run(exp_name, checkpoint, clips)
    env = create_env(env_args, num_envs=max(2, count_clips(env_args)), device=device, eval_mode=True,
                     camera_res=res)
    policy = load_policy(env, algo_args, ckpt, device)
    run_label = f"{method} ({exp_name}, {ckpt.stem})"
    rec = Recorder(env, run_label, out_dir / exp_name, fps)
    env.reset()
    obs, _ = env.get_observations()
    rec.frame()
    env.step_callback = rec.on_step
    print(f"{run_label}: replaying {int(rec.running.sum())} clips")
    while rec.running.any():
        action, _ = policy(obs, deterministic=True)
        obs, *_ = env.step(action)
    env.step_callback = None
    return rec.close()


# ====================================================================== human demonstration
def human_frames(clip_path: Path, size: tuple[int, int]) -> list[np.ndarray]:
    """HumanTouch head-camera frames of a clip, one per simulation frame (clip metadata `source_frames`:
    the 60 Hz dataset frame of each 30 Hz clip frame), resized to `size` (w, h)."""
    meta = pickle.loads(clip_path.read_bytes())["metadata"]
    video = head_video_path(Path(meta["tactile_mapping"]).parents[2], int(meta["episode"]))  # <task>/meta/humantouch/...
    if not video.exists():
        print(f"  no head-camera video {video}")
        return []
    return read_head_frames(video, meta["source_frames"], size)


def human_row(frames: list[np.ndarray], k: int, clip: dict, width: int, size: tuple[int, int]) -> np.ndarray:
    """Head-camera frame k with the clip's annotations next to it."""
    w, h = size
    img = frames[min(k, len(frames) - 1)] if frames else np.zeros((h, w, 3), np.uint8)
    info = np.zeros((h, max(width - w, 0), 3), np.uint8)
    n = clip["frames"]
    lines = ["human demonstration", "(HumanTouch head camera)", "", clip["name"],
             f"frame {min(k, n - 1)}/{n - 1}" + ("   (clip over)" if k >= n else ""), ""]
    lines += [t.replace("_", " ") for t in clip["types"]] + [""]
    for side in ("left", "right"):
        obj = clip.get(side)
        if obj is None:
            lines.append(f"{side}: no object")
            continue
        a, b = obj["grasp"]
        lines.append(f"{side}: grasp {a}-{b}" + ("  << GRASPING" if a <= k < b else ""))
        lines.append(f"   cylinder r {100 * obj['radius']:.1f} cm, l {100 * obj['length']:.1f} cm")
    for j, text in enumerate(lines):
        cv2.putText(info, text, (10, 24 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 230, 120) if "GRASPING" in text else (230, 230, 230), 1, cv2.LINE_AA)
    return pad_to(np.hstack([img, info]), width)


def stack_videos(columns: list[list[Path]], clip_dir: Path, out_dir: Path, fps: int) -> None:
    """One video per clip: the human head-camera video, then the runs top to bottom. The video lasts as
    long as the clip or the longest episode; a run whose episode ended holds its last frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {c["name"]: c for c in json.loads((clip_dir / "index.json").read_text())["clips"]}
    for paths in zip(*columns):
        name = paths[0].stem
        readers = [imageio.get_reader(p) for p in paths]
        iters = [iter(r) for r in readers]
        last = [next(it) for it in iters]
        width = max(x.shape[1] for x in last)
        size = (width * 2 // 3 // 2 * 2, width * 2 // 3 * 9 // 16 // 2 * 2)   # 16:9, two thirds of the width
        human = human_frames(clip_dir / f"{name}.pkl", size)
        clip = index[name]
        writer = imageio.get_writer(out_dir / f"{name}.mp4", fps=fps, codec="libx264", quality=7, macro_block_size=1)
        k, alive = 0, True
        while alive or k < clip["frames"]:
            top = label(human_row(human, k, clip, width, size), "", height=6)
            writer.append_data(even(np.vstack([top] + [pad_to(x, width) for x in last])))
            k += 1
            alive = False
            for j, it in enumerate(iters):
                try:
                    last[j] = next(it)
                    alive = True
                except StopIteration:
                    pass
        writer.close()
        for r in readers:
            r.close()
        print(f"  {out_dir / f'{name}.mp4'}")


# ====================================================================== interactive viewer
@torch.no_grad()
def view_run(exp_name: str, checkpoint, clips, device: str) -> None:
    method, env_args, algo_args, ckpt = load_run(exp_name, checkpoint, clips)
    env = create_env(env_args, num_envs=max(2, count_clips(env_args)), device=device, eval_mode=True,
                     show_viewer=True)
    policy = load_policy(env, algo_args, ckpt, device)
    print(f"{method} ({exp_name}, {ckpt.stem}): environment 0 is {env.clip_names[int(env.env_clip[0])]}; Ctrl-C to stop")
    try:
        while True:
            env.reset()
            obs, _ = env.get_observations()
            mid, head = env.target_wrist_pos[0].mean(0), env.ref_at("head_pos", env.t)[0]
            env.scene.viewer.set_camera_pose(pos=head.cpu().numpy(), lookat=mid.cpu().numpy())
            done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            while not done.all():
                tic = time.time()
                draw_reference(env)
                action, _ = policy(obs, deterministic=True)
                obs, _, _, terminated, truncated, _ = env.step(action)
                done |= (terminated | truncated)[:, 0]
                time.sleep(max(0.0, env.dt - (time.time() - tic)))
    except KeyboardInterrupt:
        pass


def main(exp_names: str, checkpoint: str | None = None, clips: str | None = None, name: str | None = None,
         res: str = "480,360", fps: int = 30, viewer: bool = False, compose_only: bool = False,
         device: str = "cuda") -> None:
    runs = [e for e in str(exp_names).split(",") if e]
    if viewer:
        view_run(runs[0], checkpoint, clips, device)
        return
    out = REPO / "outputs" / "videos" / (name or "_vs_".join(runs))
    res_t = tuple(int(x) for x in str(res).split(","))
    clip_dir = Path(load_run(runs[0], checkpoint, clips)[1].data.clip_dir)
    if compose_only:
        names = clips.split(",") if clips else sorted(p.stem for p in (out / runs[0]).glob("*.mp4"))
        columns = [[out / e / f"{n}.mp4" for n in names] for e in runs]
    else:
        columns = [record_run(e, checkpoint, clips, out, res_t, fps, device) for e in runs]
    print("videos (human demonstration + " + ", ".join(runs) + "):")
    stack_videos(columns, clip_dir, out, fps)


if __name__ == "__main__":
    script, env_ov, _, _ = split_overrides(parse_cli(sys.argv[1:]), MAIN_ARGS)
    if "exp_names" not in script or env_ov:
        raise SystemExit(__doc__)
    main(**{k: (str(v) if k in ("exp_names", "clips", "checkpoint", "res", "name") else v) for k, v in script.items()})
