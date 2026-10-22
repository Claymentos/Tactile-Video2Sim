#!/usr/bin/env python3
"""RecGen's own `scripts/run_inference.py` (one object mesh from RGB + metric depth + mask), with its mask erosion
made optional (launch/reconstruct.sh).

RecGen erodes the object mask (5x5, one iteration) before reconstructing, to drop pixels that bleed onto the
background. On a thin part that removes the part: a screwdriver's shaft is 3-4 px wide at 640x480, the erosion
keeps 44% of its 775-pixel mask and none of the shaft, and RecGen returned an 8 cm handle for a ~15 cm screwdriver.
`run_inference.py` has no flag for it, so this patches `recgen_inference.generate` and runs the script unchanged,
passing every other argument through.

Run in the `recgen` env (RECGEN_DIR: the RecGen checkout, default ~/recgen):
    python scripts/reconstruction/reconstruct_mesh_recgen.py --no-mask-erosion --rgb all_frames/000000.png \
        --depth depth_moge/000000.png --mask video_segmentation/masks/frame_000000_masks/hammer.png \
        --intrinsics intrinsics.yaml --checkpoint recgen_base.multiview_stereo --out recgen_out_moge/hammer --name hammer
"""

import os
import runpy
import sys

RECGEN_DIR = os.environ.get("RECGEN_DIR", os.path.expanduser("~/recgen"))


def main():
    argv = sys.argv[1:]
    erosion = "--no-mask-erosion" not in argv
    argv = [a for a in argv if a != "--no-mask-erosion"]
    # relative paths are the caller's: RecGen runs from its own checkout
    argv = [os.path.abspath(a) if os.path.exists(a) or (i and argv[i - 1] == "--out") else a for i, a in enumerate(argv)]
    sys.path.insert(0, RECGEN_DIR)
    import recgen_inference

    if not erosion:
        original = recgen_inference.generate

        def generate(*args, **kwargs):
            kwargs["mask_erosion_enabled"] = False
            return original(*args, **kwargs)

        # run_inference.py does `from recgen_inference import generate` when it runs, so replacing the attribute
        # first is enough
        recgen_inference.generate = generate
        print("[recgen] mask erosion disabled")
    os.chdir(RECGEN_DIR)
    sys.argv = ["scripts/run_inference.py", *argv]
    runpy.run_path(os.path.join(RECGEN_DIR, "scripts", "run_inference.py"), run_name="__main__")


if __name__ == "__main__":
    main()
