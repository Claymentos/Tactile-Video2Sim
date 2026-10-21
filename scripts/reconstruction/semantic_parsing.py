#!/usr/bin/env python3
"""Semantic parsing of a manipulation video with a vision-language model (Benchmark-Genesis' Stage 1,
extended to clips): Gemini (API) or Qwen-VL (local, --backend qwen).

Input: one video (e.g. a HumanTouch head-camera episode). Output: one JSON file describing the scene,
made in three calls on the same video:
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
The model samples the video at --fps: clip times are hints to within ~1/fps s, not frame-accurate
(scripts/data/cut_clips.py pads them; the glove gives the exact grasp inside).

Backends (same prompts, same schemas, same output):
  gemini  Google's API, the answer constrained to the schema. An env with google-genai and imageio-ffmpeg (the
          `sam3d` conda env); reads GEMINI_API_KEY. Free-tier quotas and 503 overloads are per model (--model).
  qwen    Qwen-VL run locally with transformers (default Qwen/Qwen3-VL-4B-Instruct, ~9 GB downloaded once from
          Hugging Face, ~10 GB of GPU memory): no key, no quota. The schema is given in the prompt, the answer
          checked against it and asked again with the errors when it does not match. An env with transformers
          >= 4.57, qwen-vl-utils and a video reader (the `lerobot` conda env). A small model: check the hands
          (left / right of the person) and the clip times it gives.

    python scripts/reconstruction/semantic_parsing.py --video ~/HumanTouch/X001/videos/chunk-000/observation.images.cam_head/episode_000002.mp4 \
        --out data/humantouch/scene/episode_000002.json
    python scripts/reconstruction/semantic_parsing.py --backend qwen --video <video> --out <scene.json>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tv2s_data.scene import SUPPORT_WORDS  # noqa: E402

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


def schema_errors(x, schema: dict, path: str = "$") -> list[str]:
    """What in `x` does not match `schema` (the JSON-schema subset of this file: types, enums, required)."""
    t = schema.get("type")
    ok = {"object": dict, "array": list, "string": str, "boolean": bool}.get(t)
    if t == "number":
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return [f"{path}: a number expected"]
    elif ok is not None and not isinstance(x, ok):
        return [f"{path}: {t} expected"]
    if "enum" in schema and x not in schema["enum"]:
        return [f"{path}: {x!r} is not one of {schema['enum']}"]
    errs = []
    if t == "object":
        errs += [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in x]
        for k, sub in schema.get("properties", {}).items():
            if k in x:
                errs += schema_errors(x[k], sub, f"{path}.{k}")
    elif t == "array" and "items" in schema:
        for i, v in enumerate(x):
            errs += schema_errors(v, schema["items"], f"{path}[{i}]")
    return errs


class Gemini:
    """Google's API: the video uploaded once per parse, answers constrained to the schema."""

    def __init__(self, model: str) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY not set")
        self.model, self.client = model, genai.Client(api_key=api_key)

    @contextlib.contextmanager
    def video(self, src: Path, fps: float, height: int):
        """(video part, duration s); the upload is deleted afterwards (it would stay 48 h on Google's side)."""
        from google.genai import types

        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "video.mp4"
            duration = small_video(src, small, height)
            f = self.client.files.upload(file=str(small))
        while f.state.name == "PROCESSING":
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        if f.state.name != "ACTIVE":
            raise RuntimeError(f"upload of {src} failed: {f.state}")
        try:
            yield types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                             video_metadata=types.VideoMetadata(fps=fps)), duration
        finally:
            self.client.files.delete(name=f.name)

    def ask(self, video, prompt: str, schema: dict, check=None) -> dict:
        from google.genai import errors as genai_errors
        from google.genai import types

        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                             temperature=0.0)
        for attempt in range(5):
            try:
                return json.loads(self.client.models.generate_content(model=self.model, contents=[video, prompt],
                                                                      config=config).text)
            except genai_errors.ServerError as e:
                print(f"attempt {attempt + 1}/5 failed ({e}); retrying...", file=sys.stderr)
                time.sleep(2 ** attempt)
        raise RuntimeError("Gemini unavailable")


class Qwen:
    """Qwen-VL run locally (transformers): the video read at `fps` (at most `max_frames` frames), the schema in
    the prompt, the answer parsed and checked, and asked again with the errors (up to `tries` times)."""

    def __init__(self, model: str, max_frames: int = 64, max_new_tokens: int = 4096, tries: int = 3) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"loading {model} (downloaded from Hugging Face the first time)...", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(model)
        self.net = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda").eval()
        self.max_frames, self.max_new_tokens, self.tries = max_frames, max_new_tokens, tries

    @contextlib.contextmanager
    def video(self, src: Path, fps: float, height: int):
        """(video message part, duration s): an H.264 copy at `height` px, read at `fps`."""
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "video.mp4"
            duration = small_video(src, small, height)
            fps = min(fps, self.max_frames / max(duration, 1e-3))
            yield {"type": "video", "video": str(small), "fps": fps, "max_pixels": 640 * 480}, duration

    def _generate(self, messages: list, sample: bool) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True,
                                                           return_video_metadata=True)
        metadata = None
        if videos is not None:
            videos, metadata = map(list, zip(*videos))
        inputs = self.processor(text=[text], images=images, videos=videos, video_metadata=metadata, do_resize=False,
                                return_tensors="pt", **video_kwargs).to(self.net.device)
        with torch.no_grad():
            out = self.net.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=sample,
                                    **({"temperature": 0.3, "top_p": 0.9} if sample else {}))
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def ask(self, video, prompt: str, schema: dict, check=None) -> dict:
        """`check`: the answer -> what else is wrong with it (asked again too)."""
        prompt = (f"{prompt}\n\nThe JSON schema of the answer:\n{json.dumps(schema)}\n"
                  "Answer with one JSON object only, no comment, no markdown.")
        messages = [{"role": "user", "content": [video, {"type": "text", "text": prompt}]}]
        for attempt in range(self.tries):
            reply = self._generate(messages, sample=attempt > 0)
            m = re.search(r"\{.*\}", reply, re.S)
            try:
                answer = json.loads(m.group(0)) if m else None
                errs = schema_errors(answer, schema) if answer is not None else ["no JSON object in the answer"]
                errs = errs or (check(answer) if check else [])
            except json.JSONDecodeError as e:
                answer, errs = None, [f"invalid JSON ({e})"]
            if not errs:
                return answer
            print(f"attempt {attempt + 1}/{self.tries}: the answer does not match the schema ({'; '.join(errs[:5])})",
                  file=sys.stderr)
            messages = messages[:1] + [
                {"role": "assistant", "content": [{"type": "text", "text": reply}]},
                {"role": "user", "content": [{"type": "text", "text": "This answer does not match the schema: "
                                              + "; ".join(errs[:10]) + ". Answer again with the corrected JSON only."}]}]
        raise RuntimeError(f"{self.tries} answers did not match the schema")


def support_listed(scene: dict) -> list[str]:
    """The camera registration segments the support surface (tv2s_data.scene.support_id): an object named after
    one (table, desk, ...), static."""
    if any(any(w in o["id"] for w in SUPPORT_WORDS) and o["is_static"] for o in scene["objects"]):
        return []
    return [f"the support surface the objects stand on is missing: list it as an object too, its id containing one "
            f"of {list(SUPPORT_WORDS)} (e.g. 'table'), with is_static true; objects the hands move have is_static false"]


def ask_hands(vlm, video, hint: str, clips: list[dict]) -> dict:
    """hands_used (the model), its reason, and hands_in_clips (from the clips' per-hand objects)."""
    h = vlm.ask(video, HANDS_PROMPT + hint, HANDS_SCHEMA)
    in_clips = hands_in_clips(clips)
    if in_clips not in (h["hands_used"], "none"):
        print(f"warning: hands_used {h['hands_used']} but the clips give objects to: {in_clips}", file=sys.stderr)
    return dict(hands_used=h["hands_used"], hands_reason=h["reason"], hands_in_clips=in_clips)


def parse_video(vlm, src: Path, model: str, fps: float, height: int, context: str = "") -> dict:
    with vlm.video(src, fps, height) as (video, duration):
        hint = f"\n\nContext from the dataset (may be partial): {context}" if context else ""
        scene = vlm.ask(video, OBJECTS_PROMPT + hint, OBJECTS_SCHEMA, check=support_listed)
        ids = [o["id"] for o in scene["objects"]]
        prompt = CLIPS_PROMPT.format(duration=duration, objects=json.dumps(scene["objects"], indent=1)) + hint
        clips = vlm.ask(video, prompt, clips_schema(ids))["clips"]
        hands = ask_hands(vlm, video, hint, clips)
    moved = {c[f"{s}_hand"]["object"] for c in clips for s in ("left", "right")} - {"none"}
    for o in scene["objects"]:
        if o["is_static"] and o["id"] in moved:           # a hand manipulates it in a clip: not static
            print(f"{o['id']}: manipulated in a clip, is_static set to false", file=sys.stderr)
            o["is_static"] = False
    for c in sorted(clips, key=lambda c: c["start_s"]):
        hs = [c[f"{s}_hand"] for s in ("left", "right")]
        c["objects_manipulated"] = sorted({h["object"] for h in hs} - {"none"})
        c["objects_used"] = sorted({h[k] for h in hs for k in ("object", "from", "to")} - {"none"})
    return dict(video=str(src), duration_s=round(duration, 3), model=model, sampled_fps=fps, context=context,
                scene=scene["scene"], **hands, objects=scene["objects"], clips=sorted(clips, key=lambda c: c["start_s"]))


def add_hands(vlm, scene: dict, fps: float, height: int) -> dict:
    """An existing scene with hands_used added (its objects and clips unchanged)."""
    with vlm.video(Path(scene["video"]), fps, height) as (video, _):
        hint = f"\n\nContext from the dataset (may be partial): {scene['context']}" if scene.get("context") else ""
        hands = ask_hands(vlm, video, hint, scene["clips"])
    out = {k: v for k, v in scene.items() if k not in ("objects", "clips")}
    return dict(out, **hands, objects=scene["objects"], clips=scene["clips"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the manipulation video")
    ap.add_argument("--out", required=True, help="the scene JSON to write")
    ap.add_argument("--backend", choices=("gemini", "qwen"), default="gemini",
                    help="gemini: Google's API (GEMINI_API_KEY); qwen: Qwen-VL run locally (transformers)")
    ap.add_argument("--model", default=None,
                    help="gemini (default gemini-flash-lite-latest): a model of the API, e.g. gemini-3.6-flash, "
                         "gemini-3-flash-preview; qwen (default Qwen/Qwen3-VL-4B-Instruct): a Hugging Face "
                         "Qwen-VL, e.g. Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max-frames", type=int, default=64, help="qwen: at most this many frames (fps lowered)")
    ap.add_argument("--fps", type=float, default=2.0, help="video frames per second the model samples")
    ap.add_argument("--height", type=int, default=480, help="height of the uploaded video (px)")
    ap.add_argument("--context", default="", help="what is known about the video (e.g. the dataset's task "
                                                  "and items), given to Gemini with both prompts")
    ap.add_argument("--hands-only", action="store_true",
                    help="--out exists: only ask which hands are used and add it (objects and clips kept)")
    args = ap.parse_args()

    src, out = Path(args.video).expanduser(), Path(args.out)
    if args.backend == "qwen":
        model = args.model or "Qwen/Qwen3-VL-4B-Instruct"
        vlm = Qwen(model, max_frames=args.max_frames)
    else:
        model = args.model or "gemini-flash-lite-latest"
        vlm = Gemini(model)
    if args.hands_only:
        scene = add_hands(vlm, dict(json.loads(out.read_text()), video=str(src)), args.fps, args.height)
    else:
        scene = parse_video(vlm, src, model, args.fps, args.height, args.context)
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
