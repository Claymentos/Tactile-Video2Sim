#!/usr/bin/env python3
"""A reference's quality card before any training: <reference dir>/quality.json, and the exit status (--enforce).

Rejects (the RL learnt nothing on any of these, arm1 + readme1, 15 references, 2026-10-26):
  contact   the reference hand's grasp has no anchor in hand_contact.json (hand_object_contact.py found no depth that
            brings the hand to the object: the hand then misses it, and the policy never grasps; 4 / 4 RL failures)
  slip      the median slip of the grasp-pose candidates (prepare_reference.py's select_grasp_pose, the retargeted
            hand without residual) above --max-slip-cm: the object falls out of the hand (>= 28 cm on 8 references,
            1 RL success; <= 8 cm on 7, 5 successes)
  wrist     the wrist higher than --max-wrist-z above the table (a mocap hand put on the wrong glove)
Reported only: the depth's rescale to the hands (scale.json), the object's size against the scene's size_cm, the
scene's source (a VLM, or the manifest), the object's reliably tracked frames.

Run in the `genesis` env:
    python scripts/rl/check_reference.py --recon data/egotouch/recon/<key> --reference data/references/<name> [--enforce]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="the reconstruction folder (launch/reconstruct.sh)")
    ap.add_argument("--reference", required=True, help="the reference folder (prepare_reference.py --out)")
    ap.add_argument("--scene", default=None, help="the scene JSON (semantic_parsing.py), for the report")
    ap.add_argument("--max-slip-cm", type=float, default=10.0)
    ap.add_argument("--max-wrist-z", type=float, default=1.0, help="m above the table")
    ap.add_argument("--enforce", action="store_true", help="exit 1 when rejected")
    args = ap.parse_args()
    recon, out = Path(args.recon), Path(args.reference)
    ref = np.load(out / "reference.npz", allow_pickle=True)
    side = str(ref["side"])
    checks, report = {}, {}

    contact = recon / "hand_contact.json"
    if contact.exists():
        s = json.loads(contact.read_text())["sides"].get(side, {})
        grasp = [a for a in s.get("anchors", []) if a["kind"] == "grasp"]
        checks["contact"] = dict(ok=bool(grasp), notes=s.get("notes", []))
        if grasp:
            checks["contact"].update(lam=round(grasp[0]["lam_applied"], 3), gap_cm=grasp[0]["gap_cm"],
                                     gap_after_cm=grasp[0].get("gap_after_cm"))
    else:
        checks["contact"] = dict(ok=False, notes=["no hand_contact.json (HAND_CONTACT=1 not run)"])

    if "grasp_candidates_slip" in ref.files:
        slip = float(np.median(ref["grasp_candidates_slip"])) * 100
        checks["slip"] = dict(ok=slip <= args.max_slip_cm, median_cm=round(slip, 1), max_cm=args.max_slip_cm)
    else:
        checks["slip"] = dict(ok=True, notes=["no grasp_candidates_slip (no candidate test): not checked"])

    z = ref["wrist_pos"][:, 2]
    checks["wrist"] = dict(ok=bool(z.max() <= args.max_wrist_z), median_m=round(float(np.median(z)), 3),
                           max_m=round(float(z.max()), 3))

    scale = recon / "depth_moge" / "scale.json"
    report["depth_rescale"] = json.loads(scale.read_text()).get("scale") if scale.exists() else 1.0
    report["object_measured_frames"] = round(float(np.mean(ref["obj_measured"])), 2) if "obj_measured" in ref.files else None
    if args.scene and Path(args.scene).exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        import trimesh
        from tv2s_data.scene import manipulated_object

        sc = json.loads(Path(args.scene).read_text())
        o = manipulated_object(sc)
        report["scene"] = dict(model=sc.get("model"), object=o["id"], hands_used=sc.get("hands_used"))
        if o.get("size_cm"):
            ext = np.sort(trimesh.load(out / "object_visual.obj", force="mesh").extents * 100)[::-1]
            report["object_size_cm"] = dict(mesh=[round(float(e), 1) for e in ext], scene=sorted(o["size_cm"], reverse=True),
                                            ratio_longest=round(float(ext[0] / max(max(o["size_cm"]), 1e-3)), 2))

    rejected = [k for k, c in checks.items() if not c["ok"]]
    card = dict(accepted=not rejected, rejected_by=rejected, checks=checks, report=report)
    (out / "quality.json").write_text(json.dumps(card, indent=1))
    print(f"quality: {'accepted' if not rejected else 'REJECTED by ' + ', '.join(rejected)} -> {out / 'quality.json'}")
    for k, c in checks.items():
        print(f"  {k:8s} {'ok ' if c['ok'] else 'BAD'} {json.dumps({x: v for x, v in c.items() if x != 'ok'})}")
    print(f"  report   {json.dumps(report)}")
    if args.enforce and rejected:
        sys.exit(1)


if __name__ == "__main__":
    main()
