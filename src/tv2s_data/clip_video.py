"""Head-camera video of a clip: the HumanTouch frames of its window, with what the clip says about them.

A clip's metadata gives `source_frames`, the 60 Hz dataset frame of each of its frames (30 Hz): the video
is read at those frames, so it is synchronised frame by frame with the clip (and the simulation). Under
the image, a panel shows the clip's caption and, per hand, the object Gemini says it manipulates (scene
clips) next to the grasp the glove measured, with a timeline of the glove grasp and the current frame.
"""

import json
import logging
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from tv2s_data.humantouch import SIDES

HEAD_CAMERA = "observation.images.cam_head"
PANEL_H = 120
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)   # warns on every resized read


def head_video_path(dataset_root: str | Path, episode: int) -> Path:
    """The head-camera video of an episode (LeRobot layout of meta/info.json)."""
    root = Path(dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    return root / info["video_path"].format(episode_chunk=episode // info["chunks_size"], video_key=HEAD_CAMERA,
                                           episode_index=episode)


def read_head_frames(video: Path, source_frames: np.ndarray, size: tuple[int, int], fps: float = 60.0) -> list[np.ndarray]:
    """Frames of `video` at `source_frames` (frame indices at `fps`), resized to `size` (w, h)."""
    src, (w, h) = np.asarray(source_frames), size
    gen = imageio_ffmpeg.read_frames(str(video), input_params=["-ss", f"{src[0] / fps:.4f}"],
                                     output_params=["-vf", f"scale={w}:{h}", "-frames:v", str(src[-1] - src[0] + 1)])
    next(gen)                                                            # stream metadata
    frames = [np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in gen]
    return [frames[min(f - src[0], len(frames) - 1)] for f in src] if frames else []


def _text(img: np.ndarray, text: str, row: int, color=(230, 230, 230)) -> None:
    cv2.putText(img, text, (8, 18 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _panel(clip: dict, k: int, w: int, fps: float) -> np.ndarray:
    meta = clip["metadata"]
    n = int(meta["end"] - meta["start"])
    p = np.full((PANEL_H, w, 3), 25, np.uint8)
    _text(p, f"{meta.get('name', '')}   frame {k}/{n - 1}   t = {meta['source_frames'][k] / fps:.2f} s", 0)
    caption = meta["captions"][0] if meta.get("scene") else "  ".join(meta["types"])   # annotation captions: Chinese
    _text(p, caption[: w // 7], 1, (200, 200, 120))
    x0, x1 = 300, w - 10
    for i, side in enumerate(SIDES):
        hand = clip["hands"][side]
        obj, sem = hand.get("object"), hand.get("semantic")
        # HumanTouch clips: the grasp of the proxy object; TouchScale clips (no object): the glove contact
        grasp = obj["grasp_interval"] if obj is not None else hand.get("grasp_interval")
        holding = grasp is not None and grasp[0] <= k < grasp[1]
        what = "" if sem is None else (f"{sem['object']} ({sem['from']} > {sem['to']})" if sem["object"] != "none" else "-")
        _text(p, f"{side[0].upper()}: {what}"[:40], 2 + i, (120, 230, 120) if holding else (230, 230, 230))
        # timeline: glove grasp (green), current frame (white)
        y = 30 + 20 * (2 + i)
        cv2.rectangle(p, (x0, y - 10), (x1, y - 2), (70, 70, 70), -1)
        if grasp is not None:
            g0, g1 = (x0 + (x1 - x0) * np.array(grasp) / max(n - 1, 1)).astype(int)
            cv2.rectangle(p, (g0, y - 10), (g1, y - 2), (60, 170, 60), -1)
        cx = int(x0 + (x1 - x0) * k / max(n - 1, 1))
        cv2.line(p, (cx, y - 13), (cx, y + 1), (255, 255, 255), 2)
    _text(p, "green bar: glove grasp / contact, white: current frame", 4, (60, 170, 60))
    return p


def write_clip_video(clip: dict, video: Path, out: Path, height: int = 360, fps: float = 60.0) -> int:
    """The clip's head-camera frames with its panel, at the clip rate; returns the number of frames."""
    w = int(round(height * 16 / 9)) // 2 * 2
    frames = read_head_frames(video, clip["metadata"]["source_frames"], (w, height), fps)
    if not frames:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(out), (w, height + PANEL_H), fps=float(clip["metadata"]["hz"]),
                                         codec="libx264", output_params=["-crf", "23"], macro_block_size=1)
    writer.send(None)
    for k, img in enumerate(frames):
        writer.send(np.ascontiguousarray(np.vstack([img, _panel(clip, k, w, fps)])))
    writer.close()
    return len(frames)
