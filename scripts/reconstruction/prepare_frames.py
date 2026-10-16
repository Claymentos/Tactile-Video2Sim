#!/usr/bin/env python3
"""Perception stage 1: the frames, camera intrinsics (and sensor depth) every later stage reads.

Samples the episode's head-camera video at --fps, resizes it to --height and, for HumanTouch, undistorts
it with the recorded calibration (k1 = 0.055), so that the pinhole intrinsics written next to the frames
are exact. TouchScale records registered depth (depth_head.mkv, mm): it is written for the same frames.
EgoTouch gives no intrinsics: a nominal field of view (tv2s_data.egotouch.NOMINAL_HFOV_DEG) is written,
flagged `intrinsics_estimated`, and register_camera.py fits the focal length; its glove masks (masks.npz)
are written as sam3/hands.npz (what segment_objects_sam3.py --hands gives the other datasets; an episode
without masks needs that step).
The scene file of the episode (<data>/scene/<key>.json, semantic_parsing.py) is required: it lists the
objects the later stages segment and reconstruct.

Writes <data>/perception/<key>/:
  frames/000000.jpg ...   the sampled frames
  frames.json             dataset, key, video, scene file, frame_index (video frame of every sampled
                          frame), times_s, size, intrinsics, depth source
  intrinsics.yaml         fu, fv, pu, pv (RecGen's format)
  depth/000000.png ...    uint16 depth in mm (TouchScale's sensor; HumanTouch: estimate_depth_moge.py)
and prints the folder on its last line.

    python scripts/reconstruction/prepare_frames.py --dataset humantouch --data data/humantouch --root ~/HumanTouch/X001 --episode 2
    python scripts/reconstruction/prepare_frames.py --dataset touchscale --data data/touchscale --root ~/TouchScale --episode 5a1ae97d
    python scripts/reconstruction/prepare_frames.py --dataset egotouch --data data/egotouch --root ~/EgoTouch --episode pick_up_bottle__20260314_000533_921
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def humantouch_camera(root: Path, episode: int) -> dict:
    from tv2s_data.clip_video import head_video_path
    from tv2s_data.humantouch import load_episode_meta

    cal = load_episode_meta(root, episode)["camera_calibration"]
    intr = next(v for k, v in cal.items() if "intrinsics" in k)
    m = intr["camera_matrix"]
    return dict(key=f"episode_{episode:06d}", video=head_video_path(root, episode), fps=60.0,
                size=(intr["image_width"], intr["image_height"]), K=(m["fx"], m["fy"], m["cx"], m["cy"]),
                dist=np.array(intr["distortion_coefficients"]["values"]), depth_video=None)


def touchscale_camera(root: Path, episode: str) -> dict:
    from tv2s_data import touchscale as ts

    eps = ts.select_episodes(root, episode)
    if len(eps) != 1:
        raise SystemExit(f"{len(eps)} TouchScale episodes match {episode!r}")
    ep = eps[0]
    rgb = json.loads(ep.path("head_param.json").read_text())["rgb_camera"]["intrinsic"]
    return dict(key=ep.id, video=ep.video, fps=ts.FPS, size=(1280, 720), K=(rgb["fx"], rgb["fy"], rgb["ppx"], rgb["ppy"]),
                dist=None, depth_video=ep.path("depth_head.mkv"))


def egotouch_camera(root: Path, episode: str) -> dict:
    from tv2s_data import egotouch as et

    eps = et.select_episodes(root, episode)
    if len(eps) != 1:
        raise SystemExit(f"{len(eps)} EgoTouch episodes match {episode!r}")
    ep = eps[0]
    W, H = et.video_size(ep)
    fx = (W / 2) / math.tan(math.radians(et.NOMINAL_HFOV_DEG) / 2)
    return dict(key=ep.key, video=ep.video, fps=et.FPS, size=(W, H), K=(fx, fx, W / 2, H / 2), dist=None,
                depth_video=None, estimated=True, hand_masks=lambda: et.glove_masks(ep))


def read_frames(path: Path, size: tuple[int, int], keep: np.ndarray, pix_fmt: str = "rgb24", bpp: int = 3,
                interp: str = "bicubic"):
    """Frames `keep` of a video, resized to `size` (w, h) by ffmpeg."""
    w, h = size
    every = int(keep[1] - keep[0]) if len(keep) > 1 else 1
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt=pix_fmt, bpp=bpp, output_params=[
        "-vf", f"select='not(mod(n\\,{every}))',scale={w}:{h}:flags={interp}", "-vsync", "0"])
    next(gen)
    dtype, ch = (np.uint16, 1) if pix_fmt == "gray16le" else (np.uint8, 3)
    for i, buf in enumerate(gen):
        if i >= len(keep):
            break
        a = np.frombuffer(buf, dtype)
        yield a.reshape(h, w) if ch == 1 else a.reshape(h, w, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("humantouch", "touchscale", "egotouch"), required=True)
    ap.add_argument("--data", required=True, help="prepared folder: scene/ read, perception/<key>/ written")
    ap.add_argument("--root", required=True, help="the dataset folder")
    ap.add_argument("--episode", required=True, help="HumanTouch: index; TouchScale: id or id prefix; EgoTouch: key")
    ap.add_argument("--fps", type=float, default=10.0, help="sampled frames per second")
    ap.add_argument("--height", type=int, default=720, help="height of the sampled frames (px)")
    args = ap.parse_args()
    data, root = Path(args.data), Path(args.root).expanduser()
    cam = {"humantouch": lambda: humantouch_camera(root, int(args.episode)), "touchscale": lambda: touchscale_camera(root, args.episode),
           "egotouch": lambda: egotouch_camera(root, args.episode)}[args.dataset]()
    scene = data / "scene" / f"{cam['key']}.json"
    if not scene.exists():
        raise SystemExit(f"no scene file {scene}: run scripts/reconstruction/semantic_parsing.py first")
    out = data / "perception" / cam["key"]
    for sub in ("frames", "depth"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)

    reader = imageio_ffmpeg.read_frames(str(cam["video"]))
    n_video = int(round(next(reader)["duration"] * cam["fps"]))
    reader.close()
    step = max(1, int(round(cam["fps"] / args.fps)))
    keep = np.arange(0, n_video, step)
    W0, H0 = cam["size"]
    s = args.height / H0
    size = (int(round(W0 * s)) // 2 * 2, args.height)
    fx, fy, cx, cy = (v * s for v in cam["K"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    n = 0
    for i, img in enumerate(read_frames(cam["video"], size, keep)):
        if cam["dist"] is not None:                      # distortion coefficients are scale-free: same K
            img = cv2.undistort(img, K, cam["dist"])
        cv2.imwrite(str(out / "frames" / f"{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n = i + 1
    keep = keep[:n]
    if cam["depth_video"] is not None:
        for i, d in enumerate(read_frames(cam["depth_video"], size, keep, "gray16le", 2, "neighbor")):
            cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), d)

    hm = cam["hand_masks"]() if cam.get("hand_masks") is not None else None
    if hm is not None:                                   # the dataset's glove masks: sam3/hands.npz
        hm = hm[:, np.clip(keep, 0, hm.shape[1] - 1)]
        hm = np.stack([[cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in track]
                       for track in hm])
        (out / "sam3").mkdir(exist_ok=True)
        np.savez_compressed(out / "sam3" / "hands.npz", masks=np.packbits(hm, axis=-1), width=size[0],
                            prompt="dataset glove masks (left, right)", prompt_frame=0)
    (out / "intrinsics.yaml").write_text(f"fu: {fx}\nfv: {fy}\npu: {cx}\npv: {cy}\n")
    meta = dict(dataset=args.dataset, key=cam["key"], video=str(cam["video"]), scene=str(scene.resolve()),
                video_fps=cam["fps"], sampled_fps=cam["fps"] / step, frame_index=keep.tolist(),
                times_s=(keep / cam["fps"]).round(4).tolist(), size=list(size),
                intrinsics=dict(fx=fx, fy=fy, cx=cx, cy=cy), fov_x_deg=math.degrees(2 * math.atan(size[0] / (2 * fx))),
                undistorted=cam["dist"] is not None, intrinsics_estimated=bool(cam.get("estimated", False)),
                depth="sensor" if cam["depth_video"] is not None else None)
    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    print(f"{cam['key']}: {n} frames at {meta['sampled_fps']:g} fps, {size[0]}x{size[1]}, fx {fx:.1f}"
          + (", sensor depth" if meta["depth"] else ""))
    print(out.resolve())


if __name__ == "__main__":
    main()
