#!/usr/bin/env python3
"""The object's measured poses and keyframes added to an existing reference.npz, its targets left as they are.

prepare_reference.py writes them since 2026-10-28 (keyframe_object_trajectory): obj_pos_measured / obj_rot_measured
(T, 3) / (T, 3, 3), each frame's pose solved again from the CoTracker points off the gloves; obj_inliers (T,) their
RANSAC inliers; obj_kf (T,) the keyframes, reliable (>= --min-inliers inliers among >= --min-visible points off the
gloves) and consistent with their reliable neighbours (tv2s_data.recon_reference.consistent_keyframes). This script
gives references made before the same keys, from their reference_source.json's reconstruction, so that
eval_rl_recon.py can score the object at the keyframes.

Run in the `genesis` env:
    python scripts/rl/add_object_keyframes.py data/references/<name> [data/references/<name2> ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tv2s_data.recon_reference import consistent_keyframes, measure_object_poses  # noqa: E402
from tv2s_data.reference import contact_holds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("references", nargs="+", help="reference folders (with reference_source.json, reference_info.json)")
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--min-visible", type=int, default=40)
    args = ap.parse_args()
    for d in map(Path, args.references):
        src = json.loads((d / "reference_source.json").read_text())
        obj = json.loads((d / "reference_info.json").read_text())["object"].replace(" ", "_")
        z = dict(np.load(d / "reference.npz", allow_pickle=True))
        if "source_frame" in z:
            print(f"{d}: retimed (retime_reference.py): run this on its source reference first")
            continue
        pos, rot, inl, _, _ = measure_object_poses(z, Path(src["recon"]), obj, Path(src["episode"]), args.min_visible)
        kf = consistent_keyframes(pos, rot, inl >= args.min_inliers)
        z.update(obj_pos_measured=pos, obj_rot_measured=rot, obj_inliers=inl, obj_kf=kf)
        np.savez(d / "reference.npz", **z)
        held = np.zeros(len(kf), bool)
        for a, b in contact_holds(z["real_contact"] > 0):
            held[a:b] = True
        print(f"{d.name}: {kf.sum()}/{len(kf)} keyframes ({(inl >= args.min_inliers).sum()} reliable), "
              f"{(kf & held).sum()}/{held.sum()} of the held frames, {(kf & ~held).sum()}/{(~held).sum()} at rest")


if __name__ == "__main__":
    main()
