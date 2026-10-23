#!/usr/bin/env python3
"""The hands' depth corrected so that the glove touches the object when it grasps it and lets it go
(launch/reconstruct.sh HAND_CONTACT=1, after the object's trajectory).

The hands and the object are both placed with MoGe's depth, and MoGe's near / far depth is not consistent: on
pick_up_screwdriver/103227 the left glove is 14 cm in front of the screwdriver it touches (the depth under the glove
89 cm, under the screwdriver 102 cm; the hand's size in the image puts it at ~75 cm), so the reference hand never
reaches the object and its target jumps 20 cm into the palm at the grasp. The image constrains a hand across its
camera ray, not along it; the glove says when it touches. So, per hand, at the first and last touch of each hold:

  touch frames    the hold (tv2s_data.reference.contact_holds: >= 5 glove channels) extended back and forward while
                  >= --touch-contacts channels fire (a light first contact before the firm grip, the last after it);
  object there    at rest: the robust mean of the object's measured poses in the stretch before the hold (its first
                  --rest-frames, away from the approaching hand) or after it (its last); the release is used only if
                  the tracked object is already within --rest-tol of that rest pose at the last touch (otherwise it is
                  still carried, or tracked on the glove);
  depth factor    the hand scaled about the camera centre of that frame, p -> c + lam (p - c): it moves along its rays
                  only, so its projection in the image is unchanged; lam, nearest 1 in [--lam-min, --lam-max], puts its
                  bone points --finger-radius from the object's surface (median over the touch frame and its two
                  neighbours);

and lam is interpolated linearly between a hand's anchors, held before the first and after the last (a hand without
an anchor is left as it is). MoGe's error varies over the clip, so one lam can push the hand through the table
(pick_up_wrench/094952: the grasp asks 1.12, the table allows 1.06 twenty frames later): in every frame lam is also
kept under the one that puts the hand's lowest joint --table-clearance above the table (ground_plane_moge.json),
that bound min-filtered and smoothed over time; a hand already under the table is not moved up here
(prepare_reference.py's clearance lift does it). The object, the table and the camera are not moved: the table
defines the simulated world and the object rests on it.

Reads (and keeps on its first run) mocap_joints_cotracker_raw.npz, the placement of place_mocap_hands.py or
place_video_hands.py, so that it can be run again; rewrites mocap_joints_cotracker.npz (joints and bone points) and
writes hand_contact.json. Then scripts/rl/prepare_reference.py --overwrite (the retargeting reads the hands).

Run in the `genesis` env:
    python scripts/reconstruction/hand_object_contact.py --recon data/egotouch/recon/<key> [--object screwdriver]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.recon_reference import load_real_channels, taxels_path  # noqa: E402
from tv2s_data.reference import contact_holds  # noqa: E402

SIDES = ("left", "right")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="launch/reconstruct.sh output folder")
    ap.add_argument("--object", default=None, help="object id (default: the one with a trajectory and a mesh)")
    ap.add_argument("--episode", default=None, help="the episode folder with pressure_grids.npz (default <recon>/source)")
    ap.add_argument("--scene", default=None, help="scene JSON: only the hands of its hands_used are corrected")
    ap.add_argument("--sides", default=None, help="comma list of the hands to correct (default: the scene's, else both)")
    ap.add_argument("--touch-contacts", type=int, default=2, help="glove channels of a touch around a hold")
    ap.add_argument("--finger-radius", type=float, default=0.01, help="bone points to the object's surface at a touch (m)")
    ap.add_argument("--rest-frames", type=int, default=15, help="measured frames the rest pose is averaged over")
    ap.add_argument("--rest-tol", type=float, default=0.04, help="release: the tracked object this near its rest pose (m)")
    ap.add_argument("--table-clearance", type=float, default=0.008, help="lowest joint above the table (m); < 0: off")
    ap.add_argument("--lam-min", type=float, default=0.7)
    ap.add_argument("--lam-max", type=float, default=1.5)
    ap.add_argument("--faces", type=int, default=4000, help="the object's mesh decimated to this for the distances")
    return ap.parse_args()


def rest_pose(frames, rot, trans, centre, ok, n_frames, first: bool):
    """The object's rest pose over `frames` (a stretch between holds), as the transform (R, t) of its posed mesh: the
    mean rotation and the median position of its centre over its first (or last) `n_frames` measured poses; None
    without any."""
    good = frames[ok[frames]]
    if len(good) == 0:
        return None
    pick = good[:n_frames] if first else good[-n_frames:]
    R = Rotation.from_matrix(rot[pick]).mean().as_matrix()
    p = np.median(np.einsum("tij,j->ti", rot[pick], centre) + trans[pick], 0)
    return R, p - R @ centre


def hand_object_gap(points, centre, lam, R, t, query):
    """Smallest distance (m) of the hand's points, scaled by each `lam` about the camera centre, to the object's
    surface at pose (R, t): (len(lam),)."""
    P = centre + lam[:, None, None] * (points - centre)[None]                    # (L, N, 3)
    local = (P.reshape(-1, 3) - t) @ R                                           # into the mesh's frame
    _, d, _ = query(local)
    return d.reshape(len(lam), -1).min(1)


def solve_lam(points, centre, R, t, query, radius, lam_min, lam_max):
    """The depth factor nearest 1 putting the hand `radius` from the object, or None (never that near / far)."""
    grid = np.linspace(lam_min, lam_max, int(round((lam_max - lam_min) / 0.005)) + 1)
    f = hand_object_gap(points, centre, grid, R, t, query) - radius
    roots = []
    for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:])):
        a, b = grid[i], grid[i + 1]
        for _ in range(20):                                                      # bisection on the crossing
            m = 0.5 * (a + b)
            fm = hand_object_gap(points, centre, np.array([m]), R, t, query)[0] - radius
            if np.sign(fm) == np.sign(f[i]):
                a = m
            else:
                b = m
        roots.append(0.5 * (a + b))
    return min(roots, key=lambda x: abs(x - 1)) if roots else None


def table_bound(joints, centres, plane, clearance, window=15, sigma=3.0):
    """(T,) the largest lam keeping every joint `clearance` above the table plane (n . p + d = height, n unit, up),
    min-filtered over `window` frames and smoothed, never above the frame's own bound; inf where a hand moved along
    its rays does not go down."""
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d

    n, d = plane
    down = np.einsum("tkj,j->tk", joints - centres[:, None], n)                 # height gained per unit of lam
    base = centres @ n + d
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(down < 0, (clearance - base[:, None]) / down, np.inf).min(1)
    finite = np.isfinite(lam)
    if not finite.any():
        return lam
    filled = np.where(finite, lam, lam[finite].max())
    smooth = gaussian_filter1d(minimum_filter1d(filled, size=window, mode="nearest"), sigma, mode="nearest")
    return np.where(finite, np.minimum(smooth, lam), np.inf)


def main() -> None:
    args = parse_args()
    recon = Path(args.recon).resolve()
    episode = Path(args.episode).expanduser().resolve() if args.episode else (recon / "source").resolve()
    objs = [p.name[: -len("_traj_cotracker.npz")] for p in sorted(recon.glob("*_traj_cotracker.npz"))]
    objs = [o for o in objs if (recon / "recgen_out_moge" / o / "posed_mesh.obj").exists()]
    if not args.object and len(objs) != 1:
        raise SystemExit(f"--object: one of {objs}")
    obj = args.object or objs[0]

    sides = list(SIDES)
    if args.sides:
        sides = args.sides.split(",")
    elif args.scene and Path(args.scene).exists():
        used = json.loads(Path(args.scene).read_text()).get("hands_used")
        sides = list(SIDES) if used in (None, "both") else [used]

    raw = recon / "mocap_joints_cotracker_raw.npz"
    if not raw.exists():
        shutil.copy(recon / "mocap_joints_cotracker.npz", raw)
    hands = dict(np.load(raw, allow_pickle=True))
    cam = np.load(recon / "camera_traj_cotracker.npz")
    centres = -np.einsum("tji,tj->ti", cam["rotation"], cam["translation"])     # camera t's centre, reference camera
    traj = np.load(recon / f"{obj}_traj_cotracker.npz")
    Rt, tt = traj["rotation"], traj["translation"]
    ok = traj["valid"].astype(bool) & traj["measured"].astype(bool)
    mesh = trimesh.load(recon / "recgen_out_moge" / obj / "posed_mesh.obj", force="mesh", process=False)
    if len(mesh.faces) > args.faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args.faces)
    query = trimesh.proximity.ProximityQuery(mesh).on_surface
    centre = np.asarray(mesh.centroid)
    obj_centre = np.einsum("tij,j->ti", Rt, centre) + tt                       # the object's centre per frame
    T = min(len(Rt), len(hands["left_joints"]))
    plane = None
    if args.table_clearance >= 0 and (recon / "ground_plane_moge.json").exists():
        gp = json.loads((recon / "ground_plane_moge.json").read_text())
        normal = np.asarray(gp["normal_camera"], dtype=float)
        plane = (normal / np.linalg.norm(normal), float(gp["offset"]) / np.linalg.norm(normal))

    report = {"object": obj, "sides": {}}
    for side in sides:
        tax = dict(np.load(taxels_path(side)))
        _, contact = load_real_channels(episode / "pressure_grids.npz", side, tax)
        n = contact[:T].sum(1)
        holds = contact_holds(contact[:T])
        pts, valid = hands[f"{side}_vertices"], hands[f"{side}_valid"].astype(bool)
        anchors, notes = [], []
        bounds = [0] + [x for h in holds for x in h] + [T]                      # rest stretches between the holds
        for k, (a, b) in enumerate(holds):
            first = a
            while first > 0 and n[first - 1] >= args.touch_contacts:
                first -= 1
            last = b - 1
            while last + 1 < T and n[last + 1] >= args.touch_contacts:
                last += 1
            before = np.arange(bounds[2 * k], a)
            after = np.arange(b, bounds[2 * k + 3])
            for kind, t, frames, from_start in (("grasp", first, before, True), ("release", last, after, False)):
                pose = rest_pose(frames, Rt, tt, centre, ok, args.rest_frames, from_start)
                if pose is None:
                    notes.append(f"{kind} at {t}: the object is never measured at rest there")
                    continue
                R0, p0 = pose
                away = np.linalg.norm(obj_centre[t] - (R0 @ centre + p0))
                if kind == "release" and away > args.rest_tol:
                    notes.append(f"release at {t}: the tracked object is still {away * 100:.1f} cm "
                                 f"from its rest pose (carried, or tracked on the glove): not used")
                    continue
                lams, gaps = [], []
                for s in (t - 1, t, t + 1):
                    if 0 <= s < T and valid[s]:
                        gaps.append(hand_object_gap(pts[s], centres[s], np.array([1.0]), R0, p0, query)[0])
                        lam = solve_lam(pts[s], centres[s], R0, p0, query, args.finger_radius, args.lam_min, args.lam_max)
                        if lam is not None:
                            lams.append(lam)
                if not lams:
                    notes.append(f"{kind} at {t}: no depth in [{args.lam_min}, {args.lam_max}] brings the hand to the object")
                    continue
                lam = float(np.median(lams))
                dist = float(np.median(np.linalg.norm(pts[t] - centres[t], axis=1)))
                anchors.append(dict(kind=kind, frame=int(t), lam=lam, gap_cm=round(float(np.median(gaps)) * 100, 1),
                                    shift_cm=round((lam - 1) * dist * 100, 1), _pose=(R0, p0)))
        anchors.sort(key=lambda x: x["frame"])
        if anchors:
            lam_t = np.interp(np.arange(len(pts)), [x["frame"] for x in anchors], [x["lam"] for x in anchors])
        else:
            lam_t = np.ones(len(pts))
        c = centres[np.minimum(np.arange(len(pts)), len(centres) - 1)][:, None]
        capped = 0
        if plane is not None and anchors:
            bound = table_bound(hands[f"{side}_joints"], c[:, 0], plane, args.table_clearance)
            # the table only holds the hand up: a correction towards the camera (lam < 1) is kept as it is
            new = np.maximum(np.minimum(lam_t, bound), np.minimum(lam_t, 1.0))
            capped = int(np.count_nonzero(new < lam_t - 1e-4))
            lam_t = new
            for x in anchors:
                t = x["frame"]
                x["lam_applied"] = float(lam_t[t])
                x["gap_after_cm"] = round(float(hand_object_gap(pts[t], centres[t], np.array([lam_t[t]]), *x.pop("_pose"),
                                                                query)[0]) * 100, 1)
        for key in (f"{side}_joints", f"{side}_vertices"):
            hands[key] = c + lam_t[:, None, None] * (hands[key] - c)
        for x in anchors:
            x.pop("_pose", None)
        report["sides"][side] = dict(holds=[[int(a), int(b)] for a, b in holds], anchors=anchors, notes=notes,
                                     lam_min=float(lam_t.min()), lam_max=float(lam_t.max()), frames_held_by_table=capped)
        print(f"[contact] {side}: holds {[(int(a), int(b)) for a, b in holds]}")
        for x in anchors:
            after = (f"; with the table: x {x['lam_applied']:.3f}, {x['gap_after_cm']:.1f} cm from the object"
                     if "lam_applied" in x and abs(x["lam_applied"] - x["lam"]) > 1e-4 else "")
            print(f"  {x['kind']} at frame {x['frame']}: hand {x['gap_cm']:.1f} cm from the object at rest -> "
                  f"depth x {x['lam']:.3f} ({x['shift_cm']:+.1f} cm along the camera ray){after}")
        if anchors:
            print(f"  depth x {lam_t.min():.3f}-{lam_t.max():.3f} over the clip; held up by the table on {capped} frames")
        for note in notes:
            print(f"  {note}")
        if not anchors:
            print("  no anchor: the hand is left as placed")

    np.savez(recon / "mocap_joints_cotracker.npz", **hands)
    (recon / "hand_contact.json").write_text(json.dumps(report, indent=1))
    print(f"[contact] -> {recon / 'mocap_joints_cotracker.npz'} (placement before: {raw.name}), {recon / 'hand_contact.json'}")


if __name__ == "__main__":
    main()
