#!/usr/bin/env python3
"""Semantic parsing of a manipulation video with Gemini (Benchmark-Genesis' Stage 1, extended to clips).

Input: one video (e.g. a HumanTouch head-camera episode). Output: one JSON file describing the scene,
made in two Gemini calls on the same uploaded video:
  objects  every object of the workspace with a stable id and what SAM3 needs to segment it: a text
           prompt (`description`) and a time where it is clearly visible (`visible_at_s`); plus a
           material, a shape and a size (physical properties, proxy / RecGen checks);
  clips    hints to cut the video: one clip per object transfer (reach, grasp, move, release), with a
           caption and, per hand, the object manipulated, where it comes from and where it goes, and
           the objects visible during the clip. Objects are referred to by their ids (a schema enum);
  hands    which hands manipulate objects in the demonstration (`hands_used`: left, right or both), a
           third call; it sets the simulated hands (scripts/rl/prepare_reference.py: one hand used, a
           one-hand environment). The clips' per-hand objects give the same answer (`hands_in_clips`);
           a disagreement is printed. --hands-only adds it to an existing scene file.
Gemini samples the video at --fps: clip times are hints to within ~1/fps s, not frame-accurate
(scripts/data/cut_clips.py pads them; the glove gives the exact grasp inside).

Run in an env with google-genai and imageio-ffmpeg (the `sam3d` conda env); reads GEMINI_API_KEY.

    python scripts/reconstruction/semantic_parsing.py --video ~/HumanTouch/X001/videos/chunk-000/observation.images.cam_head/episode_000002.mp4 \
        --out data/humantouch/scene/episode_000002.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

MATERIALS = ["ceramic", "porcelain", "glass", "plastic", "metal", "aluminum", "steel", "wood", "paper",
             "cardboard", "rubber", "fabric", "foam"]
SHAPES = ["cylinder", "box", "sphere", "flat", "container", "irregular"]
ACTIONS = ["none", "grasp_and_move", "hold_static", "push_or_touch", "support"]

OBJECTS_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "description": "one sentence: the workspace and the overall task"},
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "unique, lowercase, underscores, e.g. 'red_cup'"},
                "description": {"type": "string", "description": "visually grounded text prompt for a promptable "
                                                                 "segmentation model, e.g. 'small red plastic cup'"},
                "visible_at_s": {"type": "number", "description": "a time (s) where the object is fully visible, "
                                                                  "not occluded by the hands"},
                "material": {"type": "string", "enum": MATERIALS},
                "shape": {"type": "string", "enum": SHAPES},
                "size_cm": {"type": "array", "items": {"type": "number"}, "description": "approximate extent, 3 values"},
                "is_container": {"type": "boolean"},
                "mass_g": {"type": "number", "description": "best guess of the object's mass in grams, as it is in the video"},
                "is_static": {"type": "boolean", "description": "true if it never moves (table, fixed furniture)"},
                "initial_location": {"type": "string", "description": "e.g. 'in the basket', 'on the table, left'"},
            },
            "required": ["id", "description", "visible_at_s", "material", "shape", "size_cm", "is_container", "mass_g",
                         "is_static", "initial_location"],
        }},
    },
    "required": ["scene", "objects"],
}

OBJECTS_PROMPT = """The video is egocentric: the camera is on the head of a person wearing tactile gloves who
manipulates objects on a workspace (table, counter; their left hand is normally on the left of the image).
List every distinct object of the workspace, containers included (basket, box, tray), plus the support
surface (e.g. 'table'). Ignore the
hands, gloves and the person. Give each object a unique id and the requested attributes; its description
must let a segmentation model find it alone in a frame (colour, material, shape, position if ambiguous).
Respond ONLY with JSON matching the schema."""

CLIPS_PROMPT = """Same video, {duration:.1f} s long. The scene contains these objects (use only these ids):
{objects}

Split the video into manipulation clips: one clip per object transfer, from the moment a hand starts
reaching for the object to the moment it has released it and moved away. A bimanual transfer (e.g. both
hands lifting the basket) is a single clip. No clip for idle periods or for looking at the scene. Clips
are chronological and do not overlap. For each clip give its start and end (seconds), a one-sentence
caption, and for each hand (the person's left / right hand, not the image side) its action, the object it
manipulates, where the object comes from and where it goes (ids, 'none' if not applicable), plus the ids
of every object visible during the clip. Respond ONLY with JSON matching the schema."""


HANDS_SCHEMA = {
    "type": "object",
    "properties": {
        "hands_used": {"type": "string", "enum": ["left", "right", "both"],
                       "description": "the hands that touch and manipulate objects during the demonstration"},
        "reason": {"type": "string", "description": "one sentence"},
    },
    "required": ["hands_used", "reason"],
}

HANDS_PROMPT = """Same video. Which of the person's hands (their own left / right, not the image side) touch and
manipulate the objects during the demonstration: only the left hand, only the right hand, or both? A hand that
only rests, stays idle or is out of view does not count; a hand that holds, steadies or pushes an object does.
Respond ONLY with JSON matching the schema."""


def hands_in_clips(clips: list[dict]) -> str:
    """left / right / both / none: the hands the clips give an object to manipulate."""
    used = [s for s in ("left", "right") if any(c[f"{s}_hand"]["object"] != "none" for c in clips)]
    return {0: "none", 1: used[0] if used else "none", 2: "both"}[len(used)]


def clips_schema(ids: list[str]) -> dict:
    ref = {"type": "string", "enum": ["none"] + ids}
    hand = {"type": "object",
            "properties": {"action": {"type": "string", "enum": ACTIONS}, "object": ref, "from": ref, "to": ref},
            "required": ["action", "object", "from", "to"]}
    clip = {"type": "object",
            "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"},
                           "caption": {"type": "string"},
                           "left_hand": hand, "right_hand": hand,
                           "objects_present": {"type": "array", "items": {"type": "string", "enum": ids}}},
            "required": ["start_s", "end_s", "caption", "left_hand", "right_hand", "objects_present"]}
    return {"type": "object", "properties": {"clips": {"type": "array", "items": clip}}, "required": ["clips"]}


def small_video(src: Path, dst: Path, height: int) -> float:
    """H.264 copy at `height` px (the HumanTouch source is 1080p HEVC); returns the duration (s)."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-crf", "28", "-an", str(dst)], check=True)
    reader = imageio_ffmpeg.read_frames(str(dst))
    duration = float(next(reader)["duration"])
    reader.close()
    return duration


def upload(client: genai.Client, path: Path):
    f = client.files.upload(file=str(path))
    while f.state.name == "PROCESSING":
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"upload of {path} failed: {f.state}")
    return f


def ask(client: genai.Client, model: str, parts: list, schema: dict) -> dict:
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                         temperature=0.0)
    for attempt in range(5):
        try:
            return json.loads(client.models.generate_content(model=model, contents=parts, config=config).text)
        except genai_errors.ServerError as e:
            print(f"attempt {attempt + 1}/5 failed ({e}); retrying...", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini unavailable")


def uploaded_video(client: genai.Client, src: Path, fps: float, height: int):
    """(video part, uploaded file, duration s)"""
    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "video.mp4"
        duration = small_video(src, small, height)
        f = upload(client, small)
    return types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                      video_metadata=types.VideoMetadata(fps=fps)), f, duration


def ask_hands(client: genai.Client, model: str, video, hint: str, clips: list[dict]) -> dict:
    """hands_used (Gemini), its reason, and hands_in_clips (from the clips' per-hand objects)."""
    h = ask(client, model, [video, HANDS_PROMPT + hint], HANDS_SCHEMA)
    in_clips = hands_in_clips(clips)
    if in_clips not in (h["hands_used"], "none"):
        print(f"warning: hands_used {h['hands_used']} but the clips give objects to: {in_clips}", file=sys.stderr)
    return dict(hands_used=h["hands_used"], hands_reason=h["reason"], hands_in_clips=in_clips)


def parse_video(client: genai.Client, src: Path, model: str, fps: float, height: int, context: str = "") -> dict:
    video, f, duration = uploaded_video(client, src, fps, height)
    try:
        hint = f"\n\nContext from the dataset (may be partial): {context}" if context else ""
        scene = ask(client, model, [video, OBJECTS_PROMPT + hint], OBJECTS_SCHEMA)
        ids = [o["id"] for o in scene["objects"]]
        prompt = CLIPS_PROMPT.format(duration=duration, objects=json.dumps(scene["objects"], indent=1)) + hint
        clips = ask(client, model, [video, prompt], clips_schema(ids))["clips"]
        hands = ask_hands(client, model, video, hint, clips)
    finally:
        client.files.delete(name=f.name)          # uploads otherwise stay 48 h on Google's side
    for c in sorted(clips, key=lambda c: c["start_s"]):
        hs = [c[f"{s}_hand"] for s in ("left", "right")]
        c["objects_manipulated"] = sorted({h["object"] for h in hs} - {"none"})
        c["objects_used"] = sorted({h[k] for h in hs for k in ("object", "from", "to")} - {"none"})
    return dict(video=str(src), duration_s=round(duration, 3), model=model, sampled_fps=fps, context=context,
                scene=scene["scene"], **hands, objects=scene["objects"], clips=sorted(clips, key=lambda c: c["start_s"]))


def add_hands(client: genai.Client, scene: dict, model: str, fps: float, height: int) -> dict:
    """An existing scene with hands_used added (its objects and clips unchanged)."""
    video, f, _ = uploaded_video(client, Path(scene["video"]), fps, height)
    try:
        hint = f"\n\nContext from the dataset (may be partial): {scene['context']}" if scene.get("context") else ""
        hands = ask_hands(client, model, video, hint, scene["clips"])
    finally:
        client.files.delete(name=f.name)
    out = {k: v for k, v in scene.items() if k not in ("objects", "clips")}
    return dict(out, **hands, objects=scene["objects"], clips=scene["clips"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the manipulation video")
    ap.add_argument("--out", required=True, help="the scene JSON to write")
    ap.add_argument("--model", default="gemini-flash-lite-latest",
                    help="gemini-flash-latest has been persistently 503-overloaded in testing; "
                         "gemini-flash-lite-latest and gemini-3.6-flash both work reliably")
    ap.add_argument("--fps", type=float, default=2.0, help="video frames per second Gemini samples")
    ap.add_argument("--height", type=int, default=480, help="height of the uploaded video (px)")
    ap.add_argument("--context", default="", help="what is known about the video (e.g. the dataset's task "
                                                  "and items), given to Gemini with both prompts")
    ap.add_argument("--hands-only", action="store_true",
                    help="--out exists: only ask which hands are used and add it (objects and clips kept)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set")
    src, out = Path(args.video).expanduser(), Path(args.out)
    client = genai.Client(api_key=api_key)
    if args.hands_only:
        scene = add_hands(client, dict(json.loads(out.read_text()), video=str(src)), args.model, args.fps, args.height)
    else:
        scene = parse_video(client, src, args.model, args.fps, args.height, args.context)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, indent=2, ensure_ascii=False))

    print(f"{scene['scene']}\nhands used: {scene['hands_used']} ({scene['hands_reason']}); in the clips: {scene['hands_in_clips']}"
          f"\nobjects:")
    for o in scene["objects"]:
        print(f"  {o['id']:20s} {o['description']}  (visible at {o['visible_at_s']:.1f} s)")
    print("clips:")
    for c in scene["clips"]:
        hands = "  ".join(f"{s[0].upper()}: {c[f'{s}_hand']['object']} ({c[f'{s}_hand']['from']} -> "
                          f"{c[f'{s}_hand']['to']})" for s in ("left", "right"))
        print(f"  {c['start_s']:5.1f}-{c['end_s']:5.1f} s  {hands}  | {c['caption']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
