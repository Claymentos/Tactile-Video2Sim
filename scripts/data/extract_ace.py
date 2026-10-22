#!/usr/bin/env python3
"""ACE-Data-0 Tactile takes -> EgoTouch-shaped episode folders (tv2s_data/ace.py), what launch/reconstruct.sh and
scripts/rl/prepare_reference.py read.

Per take, in <out>/<key>/:
  chest.mp4           the egocentric video (link to the dataset's egocap/stereo0)
  pressure_grids.npz  both gloves on EgoTouch's 21x21 grid at every video frame (EgoTouch's keys: <side>_pressure_grid,
                      tactile_max, ...), baseline-free
  glove_raw.npz       the same frames' raw cells: <side>_raw (T, 256) baseline-free, <side>_baseline (256,),
                      <side>_imu (T, 4) xyzw, <side>_contact (T,) the dataset's flag; time_s, fps
  take.json           key, shard, index, fps, frames, task and sub-tasks
--download first fetches the take's files from the hub (gated: `huggingface-cli login` with an approved account).

Run in the `lerobot` env (pandas, huggingface_hub, cv2):
    python scripts/data/extract_ace.py --root ~/ACE-Data-0 --takes t4_000078 --download
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.ace import (FPS, REPO_ID, SIDES, TACTILE_MAX, glove_grid, load_glove, load_glove_map,  # noqa: E402
                           read_captions, remove_baseline, resample_max, take_from_key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ACE_ROOT", "~/ACE-Data-0"), help="ACE-Data-0 download")
    ap.add_argument("--takes", required=True, help="comma-separated keys t<shard>_<index>")
    ap.add_argument("--out", default=str(REPO / "data" / "ace" / "episodes"))
    ap.add_argument("--download", action="store_true", help="fetch the takes' files from the hub first")
    args = ap.parse_args()
    glove_map = load_glove_map()
    for key in args.takes.split(","):
        take = take_from_key(args.root, key)
        if args.download:
            from huggingface_hub import hf_hub_download

            for f in take.hub_files():
                hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=str(take.root))
        for f in [take.video] + [take.tactile(s) for s in SIDES]:
            if not f.exists():
                raise SystemExit(f"missing {f} (--download)")

        cap = cv2.VideoCapture(str(take.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_times = np.arange(T) / fps                   # the video starts at the take's t = 0 (meta/takes)
        out = Path(args.out) / key
        out.mkdir(parents=True, exist_ok=True)
        link = out / "chest.mp4"
        link.unlink(missing_ok=True)
        link.symlink_to(take.video.resolve())

        grids = dict(tactile_max=np.array(TACTILE_MAX), bend_max=np.array(TACTILE_MAX), grid_size=np.array(21),
                     num_frames=np.array(T), separate_normalization=np.array(False))
        raw_out = dict(time_s=frame_times, fps=np.array(fps))
        for s in SIDES:
            g = load_glove(take, s)
            raw, base = remove_baseline(g["raw"])
            at_frames = resample_max(g["time_s"], raw, frame_times)
            grids[f"{s}_pressure_grid"] = glove_grid(at_frames, s, glove_map)
            grids[f"baseline_corrected_{s}"] = np.array(True)
            # the IMU and the flag at the next sample (a max of unit quaternions is not one)
            near = np.clip(np.searchsorted(g["time_s"], frame_times), 0, len(g["time_s"]) - 1)
            imu = g["imu"][near]
            raw_out.update({f"{s}_raw": at_frames, f"{s}_baseline": base, f"{s}_imu": imu,
                            f"{s}_contact": g["contact"][near]})
            on = (at_frames > 5).sum(1)
            lost = sorted(set(np.flatnonzero(raw.max(0) > 5)) - set(glove_map[s][1]))
            print(f"{key} {s}: {len(g['time_s'])} samples ({len(g['time_s']) / np.ptp(g['time_s']):.0f} Hz) -> {T} frames; "
                  f"cells above 5 on {np.count_nonzero(on)}/{T} frames (peak {on.max()} cells); baseline removed from "
                  f"{np.count_nonzero(base > 0)} cells (max {base.max():.0f}); raw cells off the grid that fired: {lost}")
        np.savez_compressed(out / "pressure_grids.npz", **grids)
        np.savez_compressed(out / "glove_raw.npz", **raw_out)
        cap = read_captions(take)
        (out / "take.json").write_text(json.dumps(dict(key=key, shard=take.shard, index=take.index, fps=fps, frames=T,
                                                       video=str(take.video), **cap), indent=1))
        print(f"{key}: {T} frames at {fps:g} fps, task: {cap['task'] or '-'} -> {out}")


if __name__ == "__main__":
    main()
