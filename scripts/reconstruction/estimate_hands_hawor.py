#!/usr/bin/env python3
"""Perception: both hands as MANO from the video with HaWoR (videos without a hand mocap, and the check of it).

HaWoR (do-as-i-do/reconstruction/modules/HaWoR, run as Benchmark-Genesis' run_hawor.py does) regresses
MANO per hand over time. Its YOLO hand detector finds neither HumanTouch's black gloves nor reliably
TouchScale's, so the detections are given to it: the hand boxes come from the SAM3 hand masks
(segment_objects_sam3.py --hands), interpolated between the sampled frames, each track's side from its
mean image position (in an egocentric view the left hand is on the left), written where HaWoR looks for
its own (tracks_<start>_<end>/model_{boxes,tracks}.npy), so it skips its detection.

HaWoR runs on the video at --fps (30: its temporal model's rate), 720 px high and undistorted like the
sampled frames (HumanTouch's calibration), with --static_camera: the hands come out in each frame's camera
frame, and the camera poses we already have (camera.npz, interpolated to these frames) place them in the
world -- no SLAM.

Writes <perception>/hawor/hands.npz: frame_index (video frames), times_s, per side <side>_joints_cam and
<side>_joints_world (T, 21, 3) (MediaPipe / OpenPose order), <side>_valid (T,), <side>_vertices_cam
(T, 778, 3), and the MANO parameters (<side>_rot, <side>_hand_pose, <side>_betas, <side>_trans).

Run in the `hawor` env (HAWOR_DIR: the checkout).
    python scripts/reconstruction/estimate_hands_hawor.py --perception data/humantouch/perception/episode_000002
"""

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_objects_sam3 import segmentation_masks  # noqa: E402

HAWOR_DIR = Path(os.environ.get("HAWOR_DIR", Path.home() / "do-as-i-do/reconstruction/modules/HaWoR"))
SIDES = ("left", "right")


def hand_boxes(tracks: np.ndarray, times: np.ndarray, out_times: np.ndarray, margin: float):
    """Per side, (T_out, 4) xyxy boxes (NaN where no hand) from SAM3 hand tracks (N, T, H, W): each track's
    side by its mean x; per sampled frame the union of a side's masks; linear in time in between."""
    T, W = tracks.shape[1], tracks.shape[3]
    sides = {}
    for n, tr in enumerate(tracks):
        xs = [np.nonzero(m.any(0))[0].mean() for m in tr if m.any()]
        if xs:
            sides.setdefault("left" if np.mean(xs) < W / 2 else "right", []).append(n)
    out = {}
    for side in SIDES:
        b = np.full((T, 4), np.nan)
        for t in range(T):
            m = np.zeros(tracks.shape[2:], bool)
            for n in sides.get(side, []):
                m |= tracks[n, t]
            if m.sum() > 50:
                ys, xs = np.nonzero(m)
                w, h = xs.max() - xs.min(), ys.max() - ys.min()
                b[t] = [xs.min() - margin * w, ys.min() - margin * h, xs.max() + margin * w, ys.max() + margin * h]
        ok = np.isfinite(b[:, 0])
        res = np.full((len(out_times), 4), np.nan)
        if ok.sum() >= 2:
            for k in range(4):
                res[:, k] = np.interp(out_times, times[ok], b[ok, k], left=np.nan, right=np.nan)
            # no box across gaps longer than 0.5 s between visible samples
            nearest = np.min(np.abs(out_times[:, None] - times[ok][None]), axis=1)
            res[nearest > 0.5] = np.nan
        out[side] = res
    return out, sides


def interp_poses(T_world_cam: np.ndarray, times: np.ndarray, out_times: np.ndarray) -> np.ndarray:
    """Camera poses at out_times: positions linear, rotations slerped, between the sampled frames."""
    from scipy.spatial.transform import Rotation, Slerp

    ok = np.isfinite(T_world_cam).all((1, 2))
    t, P = times[ok], T_world_cam[ok]
    q = np.clip(out_times, t[0], t[-1])
    res = np.tile(np.eye(4), (len(out_times), 1, 1))
    res[:, :3, :3] = Slerp(t, Rotation.from_matrix(P[:, :3, :3]))(q).as_matrix()
    for k in range(3):
        res[:, k, 3] = np.interp(q, t, P[:, k, 3])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key> (frames, SAM3 hands, camera)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate HaWoR runs at")
    ap.add_argument("--margin", type=float, default=0.15, help="box padding (share of its size)")
    args = ap.parse_args()
    root = Path(args.perception).resolve()
    meta = json.loads((root / "frames.json").read_text())
    work = root / "hawor"
    seq = work / "video"                                   # HaWoR's sequence folder for <work>/video.mp4
    shutil.rmtree(seq, ignore_errors=True)
    (seq / "extracted_images").mkdir(parents=True)

    # frames at --fps, as the sampled ones (size, undistortion)
    step = max(1, int(round(meta["video_fps"] / args.fps)))
    cap = cv2.VideoCapture(meta["video"])
    W, H = meta["size"]
    K = np.array([[meta["intrinsics"]["fx"], 0, meta["intrinsics"]["cx"]], [0, meta["intrinsics"]["fy"], meta["intrinsics"]["cy"]],
                  [0, 0, 1.0]])
    dist = None
    if meta.get("undistorted"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from tv2s_data.humantouch import load_episode_meta

        root_ds = Path(meta["video"]).parents[3]
        cal = load_episode_meta(root_ds, int(meta["key"].split("_")[1]))["camera_calibration"]
        dist = np.array(next(v for k, v in cal.items() if "intrinsics" in k)["distortion_coefficients"]["values"])
    frame_index, i = [], 0
    while True:
        ok, img = cap.read()                                # BGR
        if not ok:
            break
        if i % step == 0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            if dist is not None:
                img = cv2.undistort(img, K, dist)
            cv2.imwrite(str(seq / "extracted_images" / f"{len(frame_index):06d}.png"), img)
            frame_index.append(i)
        i += 1
    frame_index = np.array(frame_index)
    out_times = frame_index / meta["video_fps"]
    T = len(frame_index)

    # detections from the SAM3 hand masks, in HaWoR's track format
    boxes, sides = hand_boxes(segmentation_masks(root / "sam3" / "hands.npz"), np.asarray(meta["times_s"]), out_times, args.margin)
    tracks = {}
    for side, tid in (("left", 5000), ("right", 10000)):
        tracks[tid] = [dict(frame=t, det=True, det_box=np.r_[boxes[side][t], 1.0][None], det_handedness=np.array([float(side == "right")]))
                       for t in range(T) if np.isfinite(boxes[side][t]).all()]
    tdir = seq / f"tracks_0_{T}"
    tdir.mkdir()
    np.save(tdir / "model_boxes.npy", np.array([], dtype=object))
    np.save(tdir / "model_tracks.npy", np.array(tracks, dtype=object))
    print(f"{T} frames at {meta['video_fps'] / step:g} fps; SAM3 hand tracks per side {sides}; boxes on "
          + ", ".join(f"{s} {np.isfinite(boxes[s][:, 0]).mean():.0%}" for s in SIDES), flush=True)

    # HaWoR, as Benchmark-Genesis' run_hawor.py runs it (static camera: no SLAM)
    import torch

    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
    (work / "video.mp4").touch()                          # placeholder: HaWoR finds the frames already extracted
    os.chdir(HAWOR_DIR)
    sys.path.insert(0, str(HAWOR_DIR))
    from lib.vis.renderer import Renderer

    Renderer.render_multiple = lambda self, *a, **k: (np.zeros((self.height, self.width, 3), np.uint8),
                                                     np.zeros((self.height, self.width), bool))
    sys.argv = ["demo.py", "--video_path", str(work / "video.mp4"), "--vis_mode", "cam",
                "--img_focal", str(meta["intrinsics"]["fx"]), "--static_camera"]
    try:
        runpy.run_path(str(HAWOR_DIR / "demo.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:                                # the visualisation at the end may fail; the npz is written before
        if not (seq / "all_hand_meshes.npz").exists():
            raise
        print(f"(HaWoR's visualisation failed: {type(e).__name__}; the hands are written)")

    d = np.load(seq / "all_hand_meshes.npz")
    cam = root / "camera.npz"
    Twc = interp_poses(np.load(cam)["T_world_cam"], np.asarray(meta["times_s"]), out_times) if cam.exists() else None
    res = dict(frame_index=frame_index, times_s=out_times)
    for s in SIDES:
        j = d[f"{s}_joints"]
        valid = np.asarray(d[f"{s}_valid"]).astype(bool).reshape(-1)[:T]
        has_box = np.isfinite(boxes[s][:, 0])
        res[f"{s}_joints_cam"], res[f"{s}_vertices_cam"] = j, d[f"{s}_vertices"]
        res[f"{s}_valid"] = valid & has_box
        for k in ("rot", "hand_pose", "betas", "trans"):
            res[f"{s}_{k}"] = d[f"{s}_{k}"]
        if Twc is not None:
            res[f"{s}_joints_world"] = np.einsum("tij,tkj->tki", Twc[:, :3, :3], j) + Twc[:, None, :3, 3]
    np.savez_compressed(work / "hands.npz", **res)
    print(f"HaWoR hands: " + ", ".join(f"{s} valid on {res[f'{s}_valid'].mean():.0%}" for s in SIDES) + f" -> {work / 'hands.npz'}")


if __name__ == "__main__":
    main()
