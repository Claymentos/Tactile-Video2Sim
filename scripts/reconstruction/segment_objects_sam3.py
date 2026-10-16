#!/usr/bin/env python3
"""Perception stage 2: a SAM3 video mask of every object of the scene file, on the sampled frames.

For each object of the episode's scene file (semantic_parsing.py), SAM3 is prompted with its text
description on the sampled frame nearest to `visible_at_s` (the time Gemini saw it unoccluded), then the
detection is propagated forwards and backwards through the whole video. A text prompt can match several
instances: the best-scoring one on the prompt frame is followed (if the tracker renumbers it, the
instance tracked on most frames). If the description finds nothing, shorter prompts are tried
(prompt_variants: the description before "with", the id, its last word, colour + shape; or --prompt
id=text), then the same prompts around visible_at_s (up to --search-s) and at the start and end of
every Gemini clip the object appears in (Gemini's visible_at_s can be a time it is hidden, e.g. in the
basket). A mask overlapping an object already segmented (IoU > --max-overlap on the prompt frame) is
rejected: objects are segmented in the scene file's order. The frames are loaded once; the session is reset between objects. --objects reruns some objects
and keeps the others' results.

Reads <perception>/frames, frames.json (prepare_frames.py). Writes <perception>/sam3/:
  <object_id>.npz       masks (T, H, W) bool (np.packbits along W: unpack with segmentation_masks()),
                        scores (T,), prompt_frame, prompt
  overlay.mp4           every object's mask over the frames, one colour per object
  summary.json          per object: prompt, prompt frame, frames with a mask, mean area (px)

Run in the `sam3` conda env (facebook/sam3).
    python scripts/reconstruction/segment_objects_sam3.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import functools
import json
import os
from pathlib import Path

import cv2
import numpy as np

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


def segmentation_masks(path: Path) -> np.ndarray:
    """(T, H, W) bool masks of one object's npz ((N, T, H, W) for hands.npz)."""
    d = np.load(path)
    return np.unpackbits(d["masks"], axis=-1, count=int(d["width"])).astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def prompt_variants(o: dict, override: str | None = None) -> list[str]:
    """Text prompts for one scene object, most specific first: SAM3 misses some long descriptions
    ('pink cylinder container with peach image on it') that a shorter one finds."""
    if override:
        return [override]
    base = o["description"].split(" with ")[0].split(",")[0].strip()
    words = o["id"].split("_")
    return list(dict.fromkeys([o["description"], base, " ".join(words), words[-1], f"{base.split()[0]} {o['shape']}"]))


def propagate(predictor, session: str) -> dict[int, dict[int, tuple[np.ndarray, float]]]:
    """Every instance SAM3 tracks, both ways from the prompt: instance -> frame -> (packed mask, score)."""
    tracks: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session,
                                                  propagation_direction="both")):
        o, t = r["outputs"], r["frame_index"]
        for i, oid in enumerate(o["out_obj_ids"]):
            m = np.asarray(o["out_binary_masks"][i], bool)
            if m.any():
                tracks.setdefault(int(oid), {})[t] = (np.packbits(m, axis=-1), float(o["out_probs"][i]))
    return tracks


def segment_hands(predictor, session: str, prompts: list[str], n_frames: int, shape: tuple[int, int],
                  max_tracks: int = 4) -> tuple[np.ndarray, str, int] | None:
    """Every instance of a hand prompt (both hands, possibly split tracks): the first prompt and frame
    (middle, quarters, start) where SAM3 sees two instances. Returns (masks (N, T, H, W) packed along W,
    longest tracks first), the prompt and the frame, or None."""
    for frame in (n_frames // 2, n_frames // 4, 3 * n_frames // 4, 0):
        for text in prompts:
            predictor.handle_request(dict(type="reset_session", session_id=session))
            out = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=frame,
                                                text=text, obj_id=1))["outputs"]
            if len(out["out_obj_ids"]) < 2:
                continue
            tracks = propagate(predictor, session)
            keep = sorted(tracks, key=lambda k: -len(tracks[k]))[:max_tracks]
            masks = np.zeros((len(keep), n_frames, shape[0], (shape[1] + 7) // 8), np.uint8)
            for n, k in enumerate(keep):
                for t, (packed, _) in tracks[k].items():
                    masks[n, t] = packed
            return masks, text, frame
    return None


def segment(predictor, session: str, frame: int, text: str | None, n_frames: int, shape: tuple[int, int],
            taken: dict[str, np.ndarray], max_overlap: float, points: np.ndarray | None = None):
    """Prompt `text` on `frame`, follow its best instance through the video: (masks (T, H, W), scores (T,)),
    else the reason: None if the prompt detects nothing, the id of the object it lands on if its mask
    overlaps one of `taken` (object id -> mask on `frame`), checked before propagating."""
    predictor.handle_request(dict(type="reset_session", session_id=session))
    prompt = dict(text=text)
    if points is not None:                                # positive clicks (x, y) in pixels
        import torch
        rel = points / [shape[1], shape[0]]
        prompt = dict(points=torch.tensor(rel, dtype=torch.float32), point_labels=torch.ones(len(points), dtype=torch.int32))
    out = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=frame, obj_id=1,
                                        **prompt))["outputs"]
    if len(out["out_obj_ids"]) == 0:
        return None
    best = int(np.argmax(out["out_probs"]))
    target = int(out["out_obj_ids"][best])
    first = np.asarray(out["out_binary_masks"][best], bool)
    hit = [i for i, m in taken.items() if iou(first, m) > max_overlap]
    if hit:
        return hit[0]
    tracks = propagate(predictor, session)
    if not tracks:
        return None
    if target not in tracks:                              # renumbered by the tracker: the longest track
        target = max(tracks, key=lambda k: len(tracks[k]))
    masks = np.zeros((n_frames, *shape), bool)
    scores = np.zeros(n_frames, np.float32)
    for t, (packed, p) in tracks[target].items():
        masks[t] = np.unpackbits(packed, axis=-1, count=shape[1]).astype(bool)
        scores[t] = p
    return masks, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (prepare_frames.py)")
    ap.add_argument("--objects", default=None, help="comma list of object ids (default: every object)")
    ap.add_argument("--prompt", action="append", default=[], metavar="ID=TEXT",
                    help="text prompt for an object instead of its Gemini description (repeatable)")
    ap.add_argument("--point", action="append", default=[], metavar="ID=X,Y[;X,Y...]@FRAME",
                    help="click prompt for an object a text prompt misses (thin objects): pixels on a sampled frame")
    ap.add_argument("--max-overlap", type=float, default=0.5,
                    help="reject a mask with this IoU with an object already segmented (prompt frame)")
    ap.add_argument("--search-s", type=float, default=3.0, help="look this far around visible_at_s")
    ap.add_argument("--hands", action="store_true",
                    help="segment the hands instead of the objects (sam3/hands.npz, for register_camera.py)")
    ap.add_argument("--hand-prompts", default="hand,glove,black glove", help="tried in order with --hands")
    ap.add_argument("--no-offload", action="store_true", help="keep the frames on the GPU (~12 MB each)")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    scene = json.loads(Path(meta["scene"]).read_text())
    objects = [o for o in scene["objects"] if not args.objects or o["id"] in args.objects.split(",")]
    times = np.asarray(meta["times_s"])
    W, H = meta["size"]
    T = len(times)
    out = root / "sam3"
    out.mkdir(exist_ok=True)

    os.environ.setdefault("TQDM_DISABLE", "1")             # one progress bar per frame otherwise
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()
    if not args.no_offload:
        predictor.model.init_state = functools.partial(predictor.model.init_state, offload_video_to_cpu=True)
    session = predictor.handle_request(dict(type="start_session", resource_path=str(root / "frames")))["session_id"]

    if args.hands:
        result = segment_hands(predictor, session, args.hand_prompts.split(","), T, (H, W))
        predictor.handle_request(dict(type="close_session", session_id=session))
        predictor.shutdown()
        if result is None:
            raise SystemExit(f"no prompt of {args.hand_prompts} sees two hands")
        masks, text, frame = result
        np.savez_compressed(out / "hands.npz", masks=masks, width=W, prompt=text, prompt_frame=frame)
        n = [(m.any(-1).any(-1)).sum() for m in masks]
        print(f"hands: '{text}' on frame {frame}: {len(masks)} tracks on {n} frames -> {out / 'hands.npz'}")
        return

    step = max(1, int(round(meta["sampled_fps"])))               # one try per second around visible_at_s
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() and args.objects else {}
    for o in objects:                                        # rerun objects are segmented anew
        summary.pop(o["id"], None)
    nearest = lambda s: int(np.argmin(np.abs(times - s)))
    for o in objects:
        t0 = nearest(o["visible_at_s"])
        tries = [t0] + [t for d in range(step, int(args.search_s * meta["sampled_fps"]) + 1, step)
                        for t in (t0 - d, t0 + d) if 0 <= t < T]
        for c in scene["clips"]:                                  # where the object is in view and moves
            if o["id"] in c["objects_present"] or o["id"] in c.get("objects_manipulated", []):
                tries += [nearest(c["start_s"]), nearest(c["end_s"])]
        tries = list(dict.fromkeys(tries))
        prompts = prompt_variants(o, dict(p.split("=", 1) for p in args.prompt).get(o["id"]))
        clicks = dict(p.split("=", 1) for p in args.point).get(o["id"])
        if clicks:                                                # a click prompt replaces the text search
            xy, frame = clicks.split("@")
            pts = np.array([[float(v) for v in c.split(",")] for c in xy.split(";")])
            tries, prompts = [int(frame)], [None]
        # masks of the objects already segmented: a prompt that lands on one of them (a generic
        # 'container' finding the basket) is rejected and the search goes on
        others = {i: segmentation_masks(out / f"{i}.npz") for i, v in summary.items()
                  if i != o["id"] and v.get("prompt") and (out / f"{i}.npz").exists()}
        result = None
        for t in tries:
            for text in prompts:
                result = segment(predictor, session, t, text, T, (H, W), {i: m[t] for i, m in others.items()},
                                 args.max_overlap, pts if text is None else None)
                if isinstance(result, str):
                    print(f"  {o['id']:20s} '{text}' on frame {t} is {result}, rejected", flush=True)
                    result = None
                if result is not None:
                    break
            if result is not None:
                break
        if result is None:
            print(f"  {o['id']:20s} not found ({prompts}, frames {tries[0]}..)")
            summary[o["id"]] = dict(prompt=None)
            (out / f"{o['id']}.npz").unlink(missing_ok=True)       # a mask of an earlier run is stale
            continue
        masks, scores = result
        text = text if text is not None else f"click {pts.astype(int).tolist()}"
        np.savez_compressed(out / f"{o['id']}.npz", masks=np.packbits(masks, axis=-1), width=W, scores=scores,
                            prompt_frame=t, prompt=text)
        area = masks.sum((1, 2))
        summary[o["id"]] = dict(prompt=text, prompt_frame=t, prompt_time_s=float(times[t]),
                                frames_with_mask=int((area > 0).sum()), frames=T,
                                mean_area_px=float(area[area > 0].mean()) if (area > 0).any() else 0.0)
        print(f"  {o['id']:20s} '{text}' on frame {t} ({times[t]:.1f} s): mask on {(area > 0).sum()}/{T} frames",
              flush=True)
        summary_path.write_text(json.dumps(summary, indent=1))
    predictor.handle_request(dict(type="close_session", session_id=session))
    predictor.shutdown()
    summary_path.write_text(json.dumps(summary, indent=1))

    # overlay video: every object, one colour each
    writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["sampled_fps"], (W, H))
    all_masks = {i: segmentation_masks(out / f"{i}.npz") for i, s in summary.items() if s["prompt"]}
    for t in range(T):
        img = cv2.imread(str(root / "frames" / f"{t:06d}.jpg"))
        for k, (i, m) in enumerate(all_masks.items()):
            c = COLORS[k % len(COLORS)][::-1]
            img[m[t]] = (0.5 * img[m[t]] + 0.5 * np.array(c)).astype(np.uint8)
            if m[t].any():
                ys, xs = np.nonzero(m[t])
                cv2.putText(img, i, (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cv2.putText(img, f"{times[t]:.1f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    n_ok = sum(bool(summary[o["id"]]["prompt"]) for o in objects)
    print(f"{n_ok}/{len(objects)} objects segmented ({len(all_masks)} in the overlay) -> {out}")


if __name__ == "__main__":
    main()
