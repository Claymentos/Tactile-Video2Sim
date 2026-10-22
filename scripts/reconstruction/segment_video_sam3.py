#!/usr/bin/env python3
"""SAM3 video segmentation of one object, prompted with text or with points on one frame (launch/reconstruct.sh).

The prompt goes on --frame_idx; the detection is propagated through the whole video. The best-scoring instance of
each frame is kept.

Writes, in <video's folder>/video_segmentation/ (or --output_dir):
  masks/frame_000000_masks/<obj_id>.png   a binary mask per frame
  overlays/frame_000000.png               the mask over the frame
  tracked_<obj_id>_<prompt>.mp4           the overlays as a video
With --instances, every instance SAM3 tracks is kept too (masks/instances_<obj_id>.npz: obj_ids (N,), masks (T, N, H, W)
bool, probs (T, N)): a text prompt such as "black glove" finds both hands, one SAM3 id each.

Run in the `sam3` env:
    python scripts/reconstruction/segment_video_sam3.py --video <recon>/<task>_rgb.mp4 --text hammer --obj_id hammer --frame_idx 0
    python scripts/reconstruction/segment_video_sam3.py --video ... --points "287,75;549,163" --point_labels "1;1" --obj_id table
"""

import argparse
import os
import subprocess

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="video (.mp4) or folder of frames")
    parser.add_argument("--text", default=None, help="text prompt")
    parser.add_argument("--points", default=None, help="point prompts 'x1,y1;x2,y2;...' in pixels")
    parser.add_argument("--point_labels", default=None, help="'1;0;1;...' (1 positive, 0 negative)")
    parser.add_argument("--obj_id", required=True, help="name of the saved masks, e.g. 'table'")
    parser.add_argument("--frame_idx", type=int, default=0, help="frame the prompt is on")
    parser.add_argument("--output_dir", default=None, help="default: <video's folder>/video_segmentation")
    parser.add_argument("--offload-video", action="store_true",
                        help="keep the decoded frames in CPU RAM: they are ~12 MB each on the GPU at SAM3's 1008 px, "
                        "so a few hundred frames exhaust a 16 GB card before tracking starts")
    parser.add_argument("--instances", action="store_true",
                        help="also keep every tracked instance (masks/instances_<obj_id>.npz), not only the best")
    return parser.parse_args()


def load_video_frames(video_path):
    """RGB frames of a video or a folder of images."""
    if video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    paths = [os.path.join(video_path, p) for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in exts]
    try:
        paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        paths.sort()
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]


def get_highest_score_obj(outputs):
    """(obj_id, score, mask) of a frame output's best-scoring object, Nones if there is none."""
    obj_ids, probs, masks = outputs["out_obj_ids"], outputs["out_probs"], outputs["out_binary_masks"]
    if len(obj_ids) == 0:
        return None, None, None
    best = np.argmax(probs)
    return obj_ids[best], probs[best], masks[best]


def overlay_mask_on_frame(frame, mask, color=(30, 144, 255), alpha=0.5):
    overlay = frame.copy()
    overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.uint8)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def main():
    args = parse_args()
    assert args.text is not None or args.points is not None, "give --text or --points"
    if args.output_dir is None:
        video_dir = (os.path.dirname(args.video) or ".") if args.video.endswith(".mp4") else args.video
        args.output_dir = os.path.join(video_dir, "video_segmentation")
    prompt_name = args.text.replace(" ", "_") if args.text is not None else "points"
    masks_base_dir = os.path.join(args.output_dir, "masks")
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(masks_base_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    import torch
    from sam3.model_builder import build_sam3_video_predictor

    print("Loading SAM3 model...")
    predictor = build_sam3_video_predictor()
    if args.offload_video:
        import functools

        model = predictor.model
        model.init_state = functools.partial(model.init_state, offload_video_to_cpu=True)
    print(f"Starting session on: {args.video}")
    session_id = predictor.handle_request(dict(type="start_session", resource_path=args.video))["session_id"]
    sam3_obj_id = 1                       # SAM3 wants an integer id; the string one names the files

    video_frames = None
    if args.text is not None:
        print(f"Adding text prompt: '{args.text}' on frame {args.frame_idx}")
        response = predictor.handle_request(dict(type="add_prompt", session_id=session_id, frame_index=args.frame_idx,
                                                 text=args.text, obj_id=sam3_obj_id))
    else:
        points = [[float(c) for c in pt.split(",")] for pt in args.points.split(";")]
        labels = [int(v) for v in args.point_labels.split(";")]
        assert len(points) == len(labels), "as many labels as points"
        video_frames = load_video_frames(args.video)
        h, w = video_frames[0].shape[:2]
        print(f"Adding point prompt on frame {args.frame_idx}: points={points}, labels={labels}")
        response = predictor.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=args.frame_idx,
            points=torch.tensor([[x / w, y / h] for x, y in points], dtype=torch.float32),
            point_labels=torch.tensor(labels, dtype=torch.int32), obj_id=sam3_obj_id))

    best_obj_id, best_score, _ = get_highest_score_obj(response["outputs"])
    if best_obj_id is not None:
        print(f"Prompt frame {args.frame_idx}: best obj_id={best_obj_id}, score={best_score:.4f}")
    else:
        print("No objects detected on the prompt frame.")

    print("Propagating through video...")
    outputs_per_frame = {}
    for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]

    if video_frames is None:
        video_frames = load_video_frames(args.video)
    h, w = video_frames[0].shape[:2]
    out_video_path = os.path.join(args.output_dir, f"tracked_{args.obj_id}_{prompt_name}.mp4")
    print(f"Saving {len(outputs_per_frame)} frames to {args.output_dir}/...")
    for frame_idx in sorted(outputs_per_frame):
        _, _, best_mask = get_highest_score_obj(outputs_per_frame[frame_idx])
        frame_mask_dir = os.path.join(masks_base_dir, f"frame_{frame_idx:06d}_masks")
        os.makedirs(frame_mask_dir, exist_ok=True)
        mask_uint8 = (best_mask * 255).astype(np.uint8) if best_mask is not None else np.zeros((h, w), np.uint8)
        cv2.imwrite(os.path.join(frame_mask_dir, f"{args.obj_id}.png"), mask_uint8)
        if frame_idx < len(video_frames):
            frame = video_frames[frame_idx]
            overlay = overlay_mask_on_frame(frame, best_mask) if best_mask is not None else frame
            cv2.imwrite(os.path.join(overlay_dir, f"frame_{frame_idx:06d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if args.instances:
        ids = sorted({int(i) for o in outputs_per_frame.values() for i in o["out_obj_ids"]})
        frames = sorted(outputs_per_frame)
        masks = np.zeros((len(frames), len(ids), h, w), bool)
        probs = np.zeros((len(frames), len(ids)), np.float32)
        for t, frame_idx in enumerate(frames):
            o = outputs_per_frame[frame_idx]
            for i, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], o["out_probs"]):
                masks[t, ids.index(int(i))], probs[t, ids.index(int(i))] = m, p
        path = os.path.join(masks_base_dir, f"instances_{args.obj_id}.npz")
        np.savez_compressed(path, obj_ids=np.array(ids), frames=np.array(frames), masks=masks, probs=probs)
        print(f"{len(ids)} instances ({', '.join(f'id {i}: {masks[:, k].any((1, 2)).mean():.0%} of frames' for k, i in enumerate(ids))}) -> {path}")

    print(f"Encoding tracked video with H264: {out_video_path}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", os.path.join(overlay_dir, "frame_%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out_video_path], check=True)
    predictor.handle_request(dict(type="close_session", session_id=session_id))
    predictor.shutdown()
    print(f"Done: {masks_base_dir}/frame_XXXXXX_masks/{args.obj_id}.png, {out_video_path}")


if __name__ == "__main__":
    main()
