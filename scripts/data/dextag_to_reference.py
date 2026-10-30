#!/usr/bin/env python3
"""A DexTaG demonstration (dextag_trajectories.zip: <object>/pickles_filtered/<stem>.pkl) -> the rl_recon reference
(reference.npz, tv2s_data/reference.py), to train this repository's RL on DexTaG's measured hand-object data instead
of a video reconstruction: if it reproduces the demonstration there, the EgoTouch failures come from the
reconstruction, not from the RL.

DexTaG's data (README, single_hand_retargeting_env_tactile_map.py), 60 Hz, z up, the table at z = 0:
  hand_trajectory      wrist_positions, wrist_rotations_aa: the world pose of the Wuji palm_link; dof_positions
                       (T, 20) Wuji joints finger1_joint1 .. finger5_joint4, the order of our Wuji model
  object_trajectory    pose_matrices (T, 4, 4) of the scanned mesh <obj_id>_collision.obj (FoundationPose, RGB-D)
  mano_reference       the human fingertips (here: the retargeting residual only)
  tactile_map          (T, 24, 32) the WUJI glove, raw 0-511; DexTaG drops < 60 as noise and divides by 60

Written here:
  hand      the palm pose and joints as they are, on the virtual base; ref_keypoints the Wuji FK (PIP, DIP, tip) and
            human_kp the same FK in the OpenPose layout (the reference is the Wuji hand, as in DexTaG)
  object    the mesh centred on its centroid, the trajectory of that centroid, measured on every frame (obj_kf all
            true: obj_reward=keyframes and dense see the same targets)
  glove     one channel per glove pixel DexTaG maps a hand point to (tactile_pixel_mapping_v5.json), its taxels the
            points of full_hand_tactile_v5.json carried into our URDF's link frames; real_raw = raw / 2 (our
            glove_scale 30 per observation unit = DexTaG's / 60), contact at raw >= 60
  holds     the in-palm object pose: the chordal mean of palm^-1 object over the frames the glove holds the object
            above the table (grasp_R_palm, grasp_p_palm; grasp_frames the in-hand start candidates)
  camera    cam_pos, a fixed head position (--head): DexTaG's camera is overhead and its pose is not in the data;
            place_arm puts the Panda's base towards it
--object-offset moves the whole object trajectory (m, world): the hand-object error a reconstruction makes, to
measure how much of it the RL absorbs.

Run in the `genesis` env:
    python scripts/data/dextag_to_reference.py --pkl ~/DexTaG/data/dextag_trajectories/hammer_mixed/pickles_filtered/20260201_165843_353314.pkl \\
        --out data/references/dextag_hammer_165843
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.recon_reference import HUMAN_KP  # noqa: E402
from tv2s_env.robots.wuji_model import FINGERS, KEYPOINT_PARTS, WujiHand, base_coordinates, mean_rotation  # noqa: E402

DEXTAG = Path.home() / "DexTaG"
NOISE = 60                                    # DexTaG: raw readings below this are noise
REGION = {f"finger{i + 1}": f for i, f in enumerate(FINGERS)}


def dextag_taxels(side: str, dextag: Path, out: Path) -> Path:
    """wuji_taxels_<side>_dextag.npz: DexTaG's tactile points on our Wuji links, one channel per glove pixel.
    DexTaG's URDF (wujihand-urdf) and ours describe the same links in frames that differ by a constant transform
    (the thumb's by a half turn), estimated at the zero pose."""
    if out.exists():
        return out
    mapping = json.loads((dextag / "assets/robot/wujihand-urdf/tactile_pixel_mapping_v5.json").read_text())
    rows, cols = mapping["image_shape"]
    ours, theirs = WujiHand(side), WujiHand(side, str(dextag / "assets/robot/wujihand-urdf/urdf" / f"{side}.urdf"))
    q0 = np.zeros(20)
    fk_o, fk_t = ours.fk(q0), theirs.fk(q0)
    link, pos, normal, pixel = [], [], [], []
    for name, points in mapping["per_link_mapping"].items():
        mine = f"{side}_{name}"
        X = np.linalg.inv(fk_o[mine]) @ fk_t[name]                       # their link frame -> ours
        P = np.array([p["local_pos"] for p in points.values()]) @ X[:3, :3].T + X[:3, 3]
        mesh = ours.collision_mesh(mine)
        _, _, tri = trimesh.proximity.closest_point(mesh, P)
        link += [mine] * len(P)
        pos.append(P)
        normal.append(mesh.face_normals[tri])
        pixel += [tuple(p["pixel"]) for p in points.values()]
    pos, normal = np.concatenate(pos), np.concatenate(normal)
    used = sorted(set(pixel))
    channel = np.array([used.index(p) for p in pixel])
    region = np.array([REGION.get(l.split("_")[1], "palm") for l in link])
    np.savez(out, link=np.array(link), pos=pos, normal=normal, channel=channel,
             raw_ids=np.array([r * cols + c for r, c in used]), cell_rc=np.array(pixel), region=region,
             side=side, grid=np.array([rows, cols]))
    print(f"taxels: {len(pos)} points, {len(used)} glove pixels of {rows}x{cols} -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True, help="a DexTaG demonstration")
    ap.add_argument("--out", required=True, help="folder of reference.npz")
    ap.add_argument("--dextag", default=str(DEXTAG), help="DexTaG checkout (its Wuji URDF and tactile mapping)")
    ap.add_argument("--fps-div", type=int, default=2, help="keep every n-th frame (60 Hz -> 30 Hz, EgoTouch's rate)")
    ap.add_argument("--head", type=float, nargs=3, default=(0.0, 0.0, 0.6), help="cam_pos (m, world)")
    ap.add_argument("--object-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="m, added to the object trajectory")
    ap.add_argument("--min-contacts", type=int, default=5, help="glove channels of a held frame")
    ap.add_argument("--min-lift", type=float, default=0.01, help="m: the held object's lowest point above the table")
    args = ap.parse_args()

    pkl, out = Path(args.pkl).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pickle.load(open(pkl, "rb"))
    side, obj_id = str(d["metadata"]["hand_side"]), str(d["metadata"]["obj_id"])
    s = slice(None, None, args.fps_div)
    h = d["hand_trajectory"]
    wrist_pos = np.asarray(h["wrist_positions"], float)[s]
    wrist_rot = Rotation.from_rotvec(np.asarray(h["wrist_rotations_aa"], float)[s]).as_matrix()
    qpos = np.asarray(h["dof_positions"], float)[s]
    pose = np.asarray(d["object_trajectory"]["pose_matrices"], float)[s]
    tac = np.asarray(d["tactile_map"])[s]
    T = len(qpos)
    fps = 60.0 / args.fps_div

    # object: centred mesh, trajectory of the centre
    mesh = trimesh.load(pkl.parent / f"{obj_id}_collision.obj", force="mesh", process=False)
    centre = np.asarray(mesh.centroid)
    mesh.vertices = np.asarray(mesh.vertices) - centre
    mesh.export(out / "object_visual.obj")
    mesh.export(out / "object_collision.obj")
    obj_rot = pose[:, :3, :3]
    obj_pos = pose[:, :3, 3] + obj_rot @ centre + np.asarray(args.object_offset)
    V = mesh.vertices
    zmin = np.einsum("vj,tij->tvi", V, obj_rot)[..., 2].min(1) + obj_pos[:, 2]
    print(f"{pkl.stem}: {T} frames at {fps:g} Hz, {side} hand, {obj_id} extents {np.round(mesh.extents * 100, 1)} cm, "
          f"lowest point at frame 0: z = {zmin[0] * 100:.1f} cm")

    # hand
    hand = WujiHand(side)
    R0 = mean_rotation(wrist_rot)
    base_q = base_coordinates(R0, wrist_pos, wrist_rot)
    B = np.tile(np.eye(4), (T, 1, 1))
    B[:, :3, :3], B[:, :3, 3] = wrist_rot, wrist_pos
    fk = hand.fk(qpos, B)
    keypoint_links = [hand.link_name(f, p) for f in FINGERS for p in KEYPOINT_PARTS]
    ref_kp = np.stack([fk[l][:, :3, 3] for l in keypoint_links], 1)
    human_kp = np.zeros((T, 21, 3))
    human_kp[:, 0] = wrist_pos
    for f in FINGERS:
        human_kp[:, 4 * FINGERS.index(f) + 1] = fk[hand.link_name(f, "link2")][:, :3, 3]
    human_kp[:, HUMAN_KP] = ref_kp
    mano = np.stack([d["mano_reference"]["finger_joints"][f"{f}_tip"] for f in ("thumb", "index", "middle", "ring", "pinky")], 1)[s]
    err = np.linalg.norm(ref_kp[:, 2::3] - mano, axis=-1)
    print(f"retargeting residual (Wuji tips vs MANO): {np.round(err.mean(0) * 1e3, 1)} mm; lowest hand keypoint "
          f"z = {ref_kp[..., 2].min() * 100:.1f} cm")

    # glove
    taxels = dextag_taxels(side, Path(args.dextag).expanduser(), REPO / "assets" / "tactile" / f"wuji_taxels_{side}_dextag.npz")
    tz = np.load(taxels)
    cols = int(tz["grid"][1])
    rc = np.stack(divmod(tz["raw_ids"], cols), 1)
    raw = np.clip(tac[:, rc[:, 0], rc[:, 1]], 0, 511).astype(np.float32)
    real_contact = raw >= NOISE
    real_raw = np.where(real_contact, raw / 2, 0).astype(np.float32)
    n = real_contact.sum(1)

    # in-palm object pose over the frames the glove holds the object off the table
    held = np.flatnonzero((n >= args.min_contacts) & (zmin > args.min_lift))
    if not len(held):
        raise SystemExit(f"no frame with >= {args.min_contacts} contacts and the object above the table")
    rel_R = np.einsum("tji,tjk->tik", wrist_rot[held], obj_rot[held])
    rel_p = np.einsum("tji,tj->ti", wrist_rot[held], obj_pos[held] - wrist_pos[held])
    grasp_R = mean_rotation(rel_R)
    grasp_p = np.median(rel_p, 0)
    spread = np.degrees(Rotation.from_matrix(np.einsum("ji,tjk->tik", grasp_R, rel_R)).magnitude())
    print(f"glove: {real_contact.shape[1]} channels, contact on {np.count_nonzero(n)} / {T} frames, peak {n.max()}; "
          f"held off the table on {len(held)} frames ({held[0]}-{held[-1]}); object in the palm: spread "
          f"{np.median(spread):.0f} deg / {np.median(np.linalg.norm(rel_p - grasp_p, axis=1)) * 100:.1f} cm median")

    # the measure the EgoTouch references fail on: the reference hand's distance to the resting object at the grasp
    first = int(np.flatnonzero(n >= args.min_contacts)[0])
    rest = trimesh.Trimesh(V @ obj_rot[0].T + obj_pos[0], mesh.faces, process=False)
    win = np.arange(max(0, first - 10), min(T, first + 11), 5)
    gap = [np.abs(trimesh.proximity.signed_distance(rest, ref_kp[t])).min() for t in win]
    print("reference hand to the resting object around the first grasp (frame %d): %s cm"
          % (first, " / ".join(f"t{t} {g * 100:.1f}" for t, g in zip(win, gap))))

    ref = dict(
        fps=fps, side=side, C=np.eye(3), shift=np.zeros(3),
        wrist_pos=wrist_pos, wrist_rot=wrist_rot, qpos=qpos, joint_names=np.array(hand.dof_names),
        hand_valid=np.ones(T, bool), base_R0=R0, base_q=base_q, human_kp=human_kp, ref_keypoints=ref_kp,
        keypoint_links=np.array(keypoint_links), human_kp_index=np.array(HUMAN_KP), hand_dz=np.zeros(T),
        obj_pos=obj_pos, obj_rot=obj_rot, obj_valid=np.ones(T, bool), obj_rest_z=max(0.0, -zmin[0]),
        cam_pos=np.tile(np.asarray(args.head, float), (T, 1)),
        real_contact=real_contact, real_raw=real_raw, taxels=str(taxels.resolve()),
        object_visual=str((out / "object_visual.obj").resolve()),
        object_collision=str((out / "object_collision.obj").resolve()),
        grasp_R_palm=grasp_R, grasp_p_palm=grasp_p, grasp_frames=held,
        obj_kf=np.ones(T, bool), obj_pos_measured=obj_pos, obj_rot_measured=obj_rot, obj_measured=np.ones(T, bool),
        obj_weight=np.ones(T), obj_margin_scale=np.ones(T), obj_target="dense",
    )
    np.savez(out / "reference.npz", **ref)
    (out / "reference_info.json").write_text(json.dumps(dict(
        source="dextag", pkl=str(pkl.resolve()), object=obj_id, side=side, frames=T, fps=fps,
        object_offset=list(args.object_offset), head=list(args.head)), indent=2))
    print("->", out / "reference.npz")


if __name__ == "__main__":
    main()
