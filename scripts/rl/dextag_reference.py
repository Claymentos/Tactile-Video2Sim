#!/usr/bin/env python3
"""A DexTaG demonstration (its pickles_filtered/<clip>.pkl: the right hand retargeted onto the Wuji Hand, the object's
mocap pose, the glove's 24 x 32 pressure map, at 60 Hz) as a reference.npz for the rl_recon environment
(rl_recon_env.py, read by tv2s_data/reference.py), so that the residual policy runs on DexTaG's data.

  hand     DexTaG's wrist (its palm_link pose) and Wuji joints, kept as they are; DexTaG's Wuji URDF (v5) differs from
           wuji-retargeting's in some joint frames, so the hand is DexTaG's own URDF, side-prefixed under
           assets/robot/dextag_wuji (train with --env.hands.wuji_dir=assets/robot/dextag_wuji)
  glove    one channel per glove pixel that has tactile points on the Wuji (DexTaG's tactile_pixel_mapping_v5.json,
           431 pixels), the points at DexTaG's positions on its links (assets/tactile/dextag_taxels_right.npz);
           contact where the map reads >= --contact-threshold (DexTaG's noise floor, 60)
  object   DexTaG's scanned collision mesh, centred on its centroid, at its mocap pose on every frame
  human    DexTaG keeps MANO's fingertips only: the human keypoints are the Wuji's, with MANO's tips at the tips
  camera   none: a fixed virtual camera in front of the scene (renders; the object error's along/across split, so
           train with equal margins, --env.obj_margin_along=0.02 --env.obj_scale_along=0.05 --env.obj_depth_discount=1)
  keyframes the object poses are FoundationPose estimates: as for EgoTouch, the frames consistent with their
           neighbours (+-3 frames, 2 cm, 10 deg), one every 5 frames (obj_kf, obj_pos/rot_measured; --env.obj_reward=keyframes)
  holds    the glove's (tv2s_data/reference.py: from 5 channels on), anchored at the object's median pose in the
           palm while it is lifted (the mocap measures it, no tactile grasp search)

    python scripts/rl/dextag_reference.py --pkl ../dextag_data/hammer_mixed/pickles_filtered/20260201_142337_939403.pkl \\
        --mesh data/dextag/meshes/hammer_scanned_collision.obj --out data/references/dt_hammer_20260201_142337_939403
"""

import argparse
import json
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.recon_reference import consistent_keyframes  # noqa: E402
from tv2s_data.reference import contact_holds  # noqa: E402
from tv2s_env.robots.wuji_model import FINGERS, KEYPOINT_PARTS, WujiHand, base_coordinates, default_urdf, mean_rotation  # noqa: E402

DEXTAG = REPO.parent / "DexTaG" / "assets" / "robot" / "wujihand-urdf"
WUJI_DIR = REPO / "assets" / "robot" / "dextag_wuji"
TAXELS = REPO / "assets" / "tactile" / "dextag_taxels_{side}.npz"
HUMAN_KP = [4 * f + k for f in range(5) for k in (2, 3, 4)]       # recon_reference.py's: PIP, DIP, tip of each finger
REGION = {"1": "thumb", "2": "index", "3": "middle", "4": "ring", "5": "little"}


def write_hand_urdf(side: str) -> Path:
    """DexTaG's Wuji URDF with wuji-retargeting's naming (<side>_ before every link and joint), its meshes copied."""
    src = DEXTAG / "urdf" / f"{side}.urdf"
    out = Path(default_urdf(side, str(WUJI_DIR)))
    out.parent.mkdir(parents=True, exist_ok=True)
    meshes = WUJI_DIR / "meshes" / side
    if not meshes.exists():
        shutil.copytree(DEXTAG / "meshes" / side, meshes)
    root = ET.parse(src).getroot()
    for el in root.iter():
        for key in ("name", "link"):
            if el.tag in ("link", "joint", "parent", "child") and el.get(key) is not None:
                el.set(key, f"{side}_{el.get(key)}")
        if el.tag == "mesh":
            el.set("filename", str((meshes / Path(el.get("filename")).name).resolve()))
    ET.ElementTree(root).write(out)
    return out


def write_taxels(side: str) -> Path:
    """The glove's points on the Wuji links, one channel per mapped pixel (row-major order)."""
    m = json.loads((DEXTAG / "tactile_pixel_mapping_v5.json").read_text())
    W = int(m["image_shape"][1])
    link, pos, rc = [], [], []
    for name, pts in m["per_link_mapping"].items():
        for p in pts.values():
            link.append(f"{side}_{name}"), pos.append(p["local_pos"]), rc.append(p["pixel"])
    rc = np.array(rc, int)
    pixels = np.unique(rc[:, 0] * W + rc[:, 1])
    channel = np.searchsorted(pixels, rc[:, 0] * W + rc[:, 1])
    region = np.array(["palm" if "palm" in l else REGION[l.split("finger")[1][0]] for l in link])
    out = Path(str(TAXELS).format(side=side))
    np.savez(out, link=np.array(link), pos=np.array(pos, float), normal=np.zeros((len(link), 3)), hit=np.ones(len(link), bool),
             s=np.full(len(link), np.nan), channel=channel, raw_ids=pixels, region=region, cell_rc=rc,
             mano_bone=np.zeros(len(link), int))
    print(f"glove: {len(link)} points on {len(set(link))} links, {len(pixels)} channels -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--mesh", required=True, help="DexTaG's <object>_scanned_collision.obj")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="DexTaG records at 60 Hz; every (60 / fps)-th frame is kept")
    ap.add_argument("--contact-threshold", type=float, default=60.0)
    ap.add_argument("--hand-table-clearance", type=float, default=0.008)
    ap.add_argument("--lift", type=float, default=0.03, help="the object counts as lifted this far above its rest height")
    ap.add_argument("--kf-window", type=int, default=3)
    ap.add_argument("--kf-max-cm", type=float, default=2.0)
    ap.add_argument("--kf-max-deg", type=float, default=10.0)
    ap.add_argument("--kf-every", type=int, default=5, help="at most one keyframe every this many frames (EgoTouch: ~1 in 5)")
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    side = str(d["metadata"]["hand_side"])
    step = int(round(60.0 / args.fps))
    sl = slice(None, None, step)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_hand_urdf(side)
    taxels = write_taxels(side)
    tax = np.load(taxels)

    # hand
    h = d["hand_trajectory"]
    wrist_pos = np.asarray(h["wrist_positions"], float)[sl]
    wrist_rot = Rotation.from_rotvec(np.asarray(h["wrist_rotations_aa"], float)[sl]).as_matrix()
    qpos = np.asarray(h["dof_positions"], float)[sl]
    T = len(qpos)
    hand = WujiHand(side, urdf=default_urdf(side, str(WUJI_DIR)))
    keypoint_links = [hand.link_name(f, p) for f in FINGERS for p in KEYPOINT_PARTS]
    tips = np.stack([np.asarray(d["mano_reference"]["finger_joints"][f"{f}_tip"], float)[sl]
                     for f in ("thumb", "index", "middle", "ring", "pinky")], 1)

    def keypoints(wp):
        B = np.tile(np.eye(4), (T, 1, 1))
        B[:, :3, :3], B[:, :3, 3] = wrist_rot, wp
        fk = hand.fk(qpos, B)
        return np.stack([fk[l][:, :3, 3] for l in keypoint_links], 1)

    ref_kp = keypoints(wrist_pos)
    err = np.linalg.norm(ref_kp[:, 2::3] - tips, axis=-1)
    print(f"{Path(args.pkl).name}: {side} hand, {T} frames at {args.fps:g} Hz; Wuji tips vs MANO tips: "
          f"mean {err.mean() * 1e3:.1f} mm (check of the wrist frame and the joint order)")
    # the hand clears the table (recon_reference.py's "frame" mode): each frame lifted by what it needs, spread
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    need = np.maximum(0.0, args.hand_table_clearance - ref_kp[..., 2].min(1))
    hand_dz = np.maximum(gaussian_filter1d(maximum_filter1d(need, size=31, mode="nearest"), 5.0, mode="nearest"), need)
    wrist_pos = wrist_pos + hand_dz[:, None] * np.r_[0, 0, 1]
    ref_kp, tips = keypoints(wrist_pos), tips + hand_dz[:, None, None] * np.r_[0, 0, 1]
    print(f"hand lifted by {hand_dz.max() * 100:.1f} cm max on {np.count_nonzero(hand_dz > 1e-4)}/{T} frames to clear the table")
    R0 = mean_rotation(wrist_rot)
    human_kp = np.zeros((T, 21, 3))
    human_kp[:, 0] = wrist_pos
    human_kp[:, HUMAN_KP] = ref_kp
    human_kp[:, [4 * f + 4 for f in range(5)]] = tips

    # glove
    tm = np.asarray(d["tactile_map"], float)[sl].reshape(T, -1)
    real_raw = tm[:, tax["raw_ids"]].astype(np.float32)
    real_contact = real_raw >= args.contact_threshold
    n = real_contact.sum(1)
    print(f"glove: contact on {np.count_nonzero(n)}/{T} frames, peak {n.max()} channels, "
          f"{(tm >= args.contact_threshold).sum() - real_contact.sum()} contact pixels off the mapped ones")

    # object
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    centre = np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid)
    local = mesh.copy()
    local.vertices = np.asarray(mesh.vertices) - centre
    local.export(out_dir / "object_visual.obj")
    local.export(out_dir / "object_collision.obj")
    P = np.asarray(d["object_trajectory"]["pose_matrices"], float)[sl]
    obj_rot = P[:, :3, :3]
    obj_pos = np.einsum("tij,j->ti", obj_rot, centre) + P[:, :3, 3]
    z = np.einsum("vj,tij->tvi", local.vertices[::max(1, len(local.vertices) // 2000)], obj_rot)[..., 2] + obj_pos[:, None, 2]
    lowest = z.min(1)
    print(f"object: {len(local.faces)} faces, extents {np.round(local.extents * 100, 1)} cm, lowest point at frame 0 "
          f"z = {lowest[0] * 100:.1f} cm, highest centre {obj_pos[:, 2].max() * 100:.1f} cm")

    # holds anchored at the measured in-palm pose while lifted
    holds = contact_holds(real_contact)
    lifted = lowest > lowest[0] + args.lift
    in_hold = np.zeros(T, bool)
    for a, b in holds:
        in_hold[a:b] = True
    frames = np.flatnonzero(in_hold & lifted)
    if not len(frames):
        raise SystemExit("the object is never lifted inside a glove hold")
    R_palm = np.einsum("tji,tjk->tik", wrist_rot[frames], obj_rot[frames])
    p_palm = np.einsum("tji,tj->ti", wrist_rot[frames], obj_pos[frames] - wrist_pos[frames])
    grasp_R, grasp_p = mean_rotation(R_palm), np.median(p_palm, 0)
    spread = np.linalg.norm(p_palm - grasp_p, axis=1)
    print(f"holds {holds}; in-palm pose over {len(frames)} lifted frames: spread {np.median(spread) * 100:.1f} cm median, "
          f"{spread.max() * 100:.1f} cm max")

    # keyframes: DexTaG's object poses are FoundationPose estimates, not measurements; as for EgoTouch, the frames
    # that agree with their neighbours (recon_reference.consistent_keyframes), one every --kf-every frames
    kf = consistent_keyframes(obj_pos, obj_rot, np.ones(T, bool), window=args.kf_window, max_cm=args.kf_max_cm,
                              max_deg=args.kf_max_deg)
    ok = np.flatnonzero(kf)
    print(f"keyframes: {len(ok)}/{T} frames consistent with their neighbours", end="")
    kf[:] = False
    last = -10 ** 9
    for t in ok:
        if t - last >= args.kf_every:
            kf[t], last = True, t
    print(f", {kf.sum()} kept (one every {args.kf_every} frames), {np.count_nonzero(kf & in_hold)} inside holds, "
          f"{np.count_nonzero(kf & lifted)} with the object lifted")

    # a fixed virtual camera 0.7 m in front of the object's start, 0.5 m up, looking at it (OpenCV axes)
    target = obj_pos[0]
    eye = target + np.array([-0.7, 0.0, 0.5])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    cam_R = np.stack([right, np.cross(fwd, right), fwd], 1)            # columns: x right, y down, z forward
    out = dict(
        fps=args.fps, side=side, wrist_pos=wrist_pos, wrist_rot=wrist_rot, qpos=qpos, joint_names=np.array(hand.dof_names),
        hand_valid=np.ones(T, bool), base_R0=R0, base_q=base_coordinates(R0, wrist_pos, wrist_rot),
        human_kp=human_kp, ref_keypoints=ref_kp, keypoint_links=np.array(keypoint_links), human_kp_index=np.array(HUMAN_KP),
        hand_dz=hand_dz, obj_pos=obj_pos, obj_rot=obj_rot, obj_valid=np.ones(T, bool), obj_rest_z=-lowest[0],
        cam_pos=np.tile(eye, (T, 1)), cam_rot=np.tile(cam_R, (T, 1, 1)), fov_y=60.0, image_size=np.array([640, 480]),
        frames_dir=str(out_dir / "no_frames"), obj_centre_cam=centre,
        real_contact=real_contact, real_raw=real_raw, taxels=str(taxels.resolve()),
        object_visual=str((out_dir / "object_visual.obj").resolve()), object_collision=str((out_dir / "object_collision.obj").resolve()),
        grasp_R_palm=grasp_R.astype(np.float32), grasp_p_palm=grasp_p.astype(np.float32), grasp_frames=frames,
        obj_kf=kf, obj_pos_measured=np.where(kf[:, None], obj_pos, np.nan), obj_rot_measured=np.where(kf[:, None, None], obj_rot, np.nan),
        obj_inliers=np.zeros(T, int),
    )
    np.savez(out_dir / "reference.npz", **out)
    (out_dir / "reference_source.json").write_text(json.dumps(dict(
        dataset="dextag", pkl=str(Path(args.pkl).resolve()), mesh=str(Path(args.mesh).resolve()), fps=args.fps,
        object=str(d["metadata"]["obj_id"]), side=side, frames=T, wuji_dir=str(WUJI_DIR)), indent=2))
    print("->", out_dir / "reference.npz")


if __name__ == "__main__":
    main()
