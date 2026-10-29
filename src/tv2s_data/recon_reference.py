"""The RL reference of one reconstruction (launch/reconstruct.sh), Benchmark-Genesis' rl_recon chain
(prepare_reference.py, mask_axes.py, tactile_grasp_pose.py), run by scripts/rl/prepare_reference.py:

  build_reference     everything in the simulated world: z up, the fitted table at z = 0, the object centred in x, y.
                      The object's mesh centred on its centre of mass (object_visual.obj) and decimated for the
                      collisions (object_collision.obj); its CoTracker trajectory; the retargeted Wuji hand
                      (retarget_mocap_wuji.py) on a virtual base; the GT mocap joints; the glove's channels
                      (max over a channel's cells, contact above 5 raw units). The hand is lifted so that its lowest
                      GT joint clears the table by a finger radius: the mocap placement and the table plane come
                      from different depth biases (on the screwdriver the GT hand reaches 6 cm under the table).
  add_mask_axes       the object's long axis in each frame from its SAM3 mask (PCA, signed towards CoTracker's
                      projected axis): CoTracker's orientation drifts 29-58 deg off the masks while the object is in
                      the fist, the masks do not.
  add_grasp_pose      the in-palm object pose that best explains the glove over the grasp window, the hand held at
                      its reference: Digital-Tactile-Twin's optimize_object_pose.py loss on the Wuji taxels (real
                      contacts pulled into the probe radius, non-contacts pushed out, hand penetration) plus the
                      video (the long axis in the image, the position across the camera ray); the best candidates
                      kept for the physics check (scripts/rl/prepare_reference.py: select_grasp_pose).

reference.npz: the keys of Benchmark-Genesis' reference (tv2s_data/reference.py reads it).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from tv2s_env.robots.wuji_model import FINGERS, KEYPOINT_PARTS, WujiHand, base_coordinates, mean_rotation

REPO = Path(__file__).resolve().parents[2]
TAXELS = REPO / "assets" / "tactile"                    # wuji_taxels_<side>.npz: the glove's taxels on the Wuji links
# OpenPose index of the human joint each Wuji keypoint corresponds to: finger f (0 thumb) -> 4f+2..4f+4
HUMAN_KP = [4 * f + k for f in range(5) for k in (2, 3, 4)]


def taxels_path(side: str) -> Path:
    """The glove's taxels carried from MANO onto the Wuji links (Benchmark-Genesis' map_tactile_to_wuji.py)."""
    return TAXELS / f"wuji_taxels_{side}.npz"


def camera_to_world(up_cam: np.ndarray) -> np.ndarray:
    """Rotation taking OpenCV camera coordinates to a z-up world, `up_cam` the up direction in the camera."""
    z_axis = np.asarray(up_cam, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    forward = np.array([0.0, 0.0, 1.0])
    y_axis = forward - forward.dot(z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    return np.stack([np.cross(y_axis, z_axis), y_axis, z_axis])


def world_frame(ground_plane_json: Path, anchor_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(C, shift): p_world = C p_cam + shift, z up with the fitted table at z = 0 and the anchor points (the object
    mesh) centred in x, y."""
    plane = json.loads(Path(ground_plane_json).read_text())
    normal = np.asarray(plane["normal_camera"], dtype=float)
    offset = float(plane["offset"]) / np.linalg.norm(normal)
    C = camera_to_world(normal / np.linalg.norm(normal))
    shift = np.r_[-(anchor_cam @ C.T)[:, :2].mean(axis=0), offset]
    return C, shift


def load_real_channels(pressure_npz: Path, hand: str, taxels: dict, thr_raw: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel glove signal (T, C) in raw 8-bit units (max over the channel's grid cells) and the contacts."""
    z = np.load(pressure_npz)
    G = z[f"{hand}_pressure_grid"] * float(z["tactile_max"])
    r, c = taxels["cell_rc"].T
    cells = G[:, r, c]
    ch = np.stack([np.nanmax(cells[:, taxels["channel"] == i], 1) for i in range(len(taxels["raw_ids"]))], 1)
    return ch, ch > thr_raw


def rot6d_to_rotmat(x):
    """6D rotation representation (Zhou et al. 2019) -> rotation matrix (..., 3, 3)."""
    import torch

    a, b = x[..., :3], x[..., 3:]
    e1 = torch.nn.functional.normalize(a, dim=-1)
    e2 = torch.nn.functional.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], dim=-1)


def object_long_axis(V: np.ndarray) -> np.ndarray:
    """The mesh's principal axis, towards the end farther from the centre (a screwdriver's shaft tip)."""
    axis = np.linalg.eigh((V - V.mean(0)).T @ (V - V.mean(0)))[1][:, -1]
    return axis if (V @ axis).max() > -(V @ axis).min() else -axis


def build_reference(recon: Path, episode: Path, obj: str, side: str, wuji_ref: Path, out_dir: Path,
                    collision_faces: int = 4000, contact_threshold: float = 5.0, hand_table_clearance: float = 0.008,
                    clearance_mode: str = "global") -> Path:
    """<out_dir>/reference.npz and the object meshes (module docstring). clearance_mode "frame": each frame lifted by
    what it needs, spread over +-15 frames and smoothed, instead of one lift for the whole clip."""
    recon, out_dir = Path(recon), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(recon / "recgen_out_moge" / obj / "posed_mesh.obj", force="mesh", process=False)
    C, shift = world_frame(recon / "ground_plane_moge.json", np.asarray(mesh.vertices))
    to_w = lambda p: p @ C.T + shift

    # object: centred mesh, trajectory of its centre of mass; object frame axes = world axes at frame 0
    centre = np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid)
    local = mesh.copy()
    local.vertices = (np.asarray(mesh.vertices) - centre) @ C.T
    local.export(out_dir / "object_visual.obj")
    coll = local.simplify_quadric_decimation(face_count=collision_faces) if len(local.faces) > collision_faces else local
    coll.export(out_dir / "object_collision.obj")
    track = np.load(recon / f"{obj}_traj_cotracker.npz")
    Rt, tt = track["rotation"], track["translation"]
    obj_rot = np.einsum("ij,tjk,lk->til", C, Rt, C)
    obj_pos = to_w(np.einsum("tij,j->ti", Rt, centre) + tt)
    obj_valid = track["valid"].astype(bool)
    z0 = (local.vertices @ obj_rot[0].T + obj_pos[0])[:, 2].min()
    print(f"object: {len(local.faces)} faces ({len(coll.faces)} collision), extents {np.round(local.extents * 100, 1)} cm, "
          f"lowest point at frame 0: z = {z0 * 100:.1f} cm")

    cam = np.load(recon / "camera_traj_cotracker.npz")
    cam_pos = to_w(-np.einsum("tji,tj->ti", cam["rotation"], cam["translation"]))
    cam_rot = np.einsum("ij,tkj->tik", C, cam["rotation"])                  # camera axes (OpenCV) in the world
    K = yaml.safe_load((recon / "intrinsics.yaml").read_text())
    width, height = int(round(2 * float(K["pu"]))), int(round(2 * float(K["pv"])))
    fov_y = float(np.degrees(2 * np.arctan(height / (2 * float(K["fv"])))))

    # hand: retargeted Wuji reference + GT human joints
    ref = np.load(wuji_ref)
    q = ref["wrist_quat"]
    wrist_rot = np.einsum("ij,tjk->tik", C, Rotation.from_quat(np.c_[q[:, 1:], q[:, :1]]).as_matrix())
    wrist_pos = to_w(ref["wrist_pos"])
    qpos = ref["qpos"]
    hand_valid = ref["valid"].astype(bool)
    R0 = mean_rotation(wrist_rot[hand_valid])
    base_q = base_coordinates(R0, wrist_pos, wrist_rot)
    print(f"virtual base: max |Euler| {np.round(np.degrees(np.abs(base_q[:, 3:]).max(0)), 1)} deg about the mean wrist orientation")
    human_kp = to_w(np.load(recon / "mocap_joints_cotracker.npz")[f"{side}_joints"])

    taxels = taxels_path(side)
    tax = dict(np.load(taxels))
    real_raw, real_contact = load_real_channels(Path(episode) / "pressure_grids.npz", side, tax, thr_raw=contact_threshold)

    # hand-table consistency: lift the hand (never the object or the table) so that its lowest joint clears the table
    hand_dz = np.zeros(len(human_kp))
    if hand_table_clearance >= 0:
        lowest = human_kp[..., 2].min(1)
        if clearance_mode == "global":
            hand_dz[:] = max(0.0, hand_table_clearance - lowest[hand_valid].min())
        else:
            from scipy.ndimage import gaussian_filter1d, maximum_filter1d
            need = np.where(hand_valid, np.maximum(0.0, hand_table_clearance - lowest), 0.0)
            hand_dz = np.maximum(gaussian_filter1d(maximum_filter1d(need, size=31, mode="nearest"), 5.0, mode="nearest"), need)
        wrist_pos = wrist_pos + hand_dz[:, None] * np.r_[0, 0, 1]
        human_kp = human_kp + hand_dz[:, None, None] * np.r_[0, 0, 1]
        base_q = base_coordinates(R0, wrist_pos, wrist_rot)
    print(f"hand lifted ({clearance_mode}) by {hand_dz.max() * 100:.1f} cm max, {np.median(hand_dz) * 100:.1f} cm "
          f"median, on {np.count_nonzero(hand_dz > 1e-4)}/{len(hand_dz)} frames, to clear the table")

    hand = WujiHand(side)
    assert list(hand.dof_names) == [str(n) for n in ref["joint_names"]]
    B = np.tile(np.eye(4), (len(qpos), 1, 1))
    B[:, :3, :3], B[:, :3, 3] = wrist_rot, wrist_pos
    fk = hand.fk(qpos, B)
    keypoint_links = [hand.link_name(f, p) for f in FINGERS for p in KEYPOINT_PARTS]
    ref_kp = np.stack([fk[l][:, :3, 3] for l in keypoint_links], 1)
    err = np.linalg.norm(ref_kp - human_kp[:, HUMAN_KP], axis=-1)[hand_valid]
    print(f"retargeting residual (Wuji PIP/DIP/tip vs GT human): mean {err.mean() * 1e3:.1f} mm, tips {err[:, 2::3].mean() * 1e3:.1f} mm")

    T = min(len(qpos), len(Rt), len(real_raw))
    n = real_contact.sum(1)
    print(f"GT tactile ({side}): {real_contact.shape[1]} channels, contact on {np.count_nonzero(n)} / {T} frames, "
          f"frames {np.flatnonzero(n)[0]}-{np.flatnonzero(n)[-1]}, peak {n.max()} channels")
    out = dict(
        fps=float(ref["fps"]), side=side, C=C, shift=shift,
        wrist_pos=wrist_pos[:T], wrist_rot=wrist_rot[:T], qpos=qpos[:T], joint_names=ref["joint_names"],
        hand_valid=hand_valid[:T], base_R0=R0, base_q=base_q[:T],
        human_kp=human_kp[:T], ref_keypoints=ref_kp[:T], keypoint_links=np.array(keypoint_links),
        human_kp_index=np.array(HUMAN_KP), hand_dz=hand_dz,
        obj_pos=obj_pos[:T], obj_rot=obj_rot[:T], obj_valid=obj_valid[:T], obj_rest_z=-z0,
        cam_pos=cam_pos[:T], cam_rot=cam_rot[:T], fov_y=fov_y, image_size=np.array([width, height]),
        frames_dir=str((recon / "all_frames").resolve()), obj_centre_cam=centre,
        real_contact=real_contact[:T], real_raw=real_raw[:T], taxels=str(taxels.resolve()),
        object_visual=str((out_dir / "object_visual.obj").resolve()),
        object_collision=str((out_dir / "object_collision.obj").resolve()),
    )
    path = out_dir / "reference.npz"
    np.savez(path, **out)
    (out_dir / "reference_info.json").write_text(json.dumps(dict(
        recon_dir=str(recon.resolve()), episode=str(Path(episode).resolve()), object=obj, side=side, frames=int(T)), indent=2))
    print("->", path)
    return path


def add_mask_axes(reference: Path, recon: Path, obj: str, min_pixels: int = 150, min_elongation: float = 3.0) -> None:
    """mask_axis_2d (T, 2) unit 2D axis (image x right, y down) signed towards the shaft tip, mask_axis_valid (T,),
    and cam_R (T, 3, 3), cam_t (T, 3), cam_K (4,) (reference camera -> camera of frame t, fu fv pu pv), so that the
    environment can project the simulated object. Frames whose mask is small or not elongated are invalid."""
    ref = dict(np.load(reference, allow_pickle=True))
    recon = Path(recon)
    cam = np.load(recon / "camera_traj_cotracker.npz")
    K = yaml.safe_load((recon / "intrinsics.yaml").read_text())
    C, shift = ref["C"], ref["shift"]
    T = len(ref["qpos"])
    axis = object_long_axis(np.asarray(trimesh.load(str(ref["object_collision"]), process=False).vertices))

    def project(Pw, t):
        x = ((Pw - shift) @ C) @ cam["rotation"][t].T + cam["translation"][t]
        return np.c_[K["fu"] * x[:, 0] / x[:, 2] + K["pu"], K["fv"] * x[:, 1] / x[:, 2] + K["pv"]]

    out, valid, disagree = np.zeros((T, 2)), np.zeros(T, bool), []
    for t in range(T):
        m = cv2.imread(str(recon / f"video_segmentation/masks/frame_{t:06d}_masks/{obj}.png"), 0)
        if m is None or (m > 128).sum() < min_pixels:
            continue
        ys, xs = np.nonzero(m > 128)
        P = np.c_[xs, ys].astype(float)
        w, v = np.linalg.eigh(np.cov((P - P.mean(0)).T))
        if np.sqrt(w[-1] / max(w[0], 1e-9)) < min_elongation:
            continue
        a = v[:, -1]
        q = project(np.stack([ref["obj_pos"][t], ref["obj_pos"][t] + ref["obj_rot"][t] @ axis * 0.08]), t)
        cot = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0])
        a = a if a @ cot >= 0 else -a
        out[t], valid[t] = a, True
        disagree.append(np.degrees(np.arccos(np.clip(a @ cot, -1, 1))))
    if disagree:
        print(f"mask axes: {valid.sum()} / {T} usable frames; CoTracker vs mask median {np.median(disagree):.0f} deg "
              f"(p90 {np.percentile(disagree, 90):.0f})")
    else:
        print(f"mask axes: no usable frame (the mask is never elongated enough): the reward uses CoTracker's axis")
    ref.update(mask_axis_2d=out, mask_axis_valid=valid, cam_R=cam["rotation"][:T], cam_t=cam["translation"][:T],
               cam_K=np.array([K["fu"], K["fv"], K["pu"], K["pv"]], float))
    np.savez(reference, **ref)


def add_grasp_pose(reference: Path, min_contacts: int = 5, radius: float = 0.004, tau: float = 0.0015,
                   w_neg: float = 0.5, w_pen: float = 20.0, starts: int = 128, w_axis: float = 0.02,
                   w_across: float = 2.0, iters: int = 600, keep: int = 32, cotracker_axis: bool = False,
                   device: str = "cuda") -> None:
    """grasp_R_palm, grasp_p_palm (object -> palm), grasp_frames, grasp_f1, and the `keep` best candidates
    (grasp_candidates_*) for the physics check. Over the grasp window (>= min_contacts glove contacts) the hand is at
    its reference configuration and one object pose in the palm frame is optimised from `starts` starts (CoTracker's
    orientation relative to the palm, perturbed; a quarter fully random):
      real contact    softplus((d - r) / tau)   pulls the channel into the probe radius
      real no-contact softplus((r - d) / tau)   pushes it out (w_neg)
      penetration     relu(-sdf) of points on the hand's collision hulls (w_pen)
      video           w_axis (1 - cos) of the long axis' angle in the image (the SAM3 masks' axis, or CoTracker's),
                      w_across relu(lateral offset - 2 cm)^2 across the camera ray against CoTracker
    The contacts alone leave the orientation free (a reversed screwdriver touches the same taxels), hence the video
    terms, restricted to what the video measures well (not the depth, not the spin about the shaft)."""
    import torch
    import torch.nn.functional as F

    from tv2s_env.tactile.mesh_sdf import MeshDistance

    dev = device
    ref = dict(np.load(reference, allow_pickle=True))
    tax = np.load(str(ref["taxels"]), allow_pickle=True)
    hand = WujiHand(str(ref["side"]))
    y_all = ref["real_contact"] > 0
    frames = np.flatnonzero(y_all.sum(1) >= min_contacts)
    print(f"grasp window: {len(frames)} frames ({frames[0]}-{frames[-1]})")

    # taxels and hand surface points in the PALM frame, per frame (reference finger configuration)
    T = hand.fk(ref["qpos"][frames])
    P = np.stack([np.einsum("fij,j->fi", T[l][:, :3, :3], p) + T[l][:, :3, 3] for l, p in zip([str(l) for l in tax["link"]], tax["pos"])], 1)
    surf = []
    for link in hand.movable_links():
        hulls = hand.collision_hulls(link)
        if hulls:
            pts = np.concatenate([h.sample(40) for h in hulls])
            surf.append(np.einsum("fij,nj->fni", T[link][:, :3, :3], pts) + T[link][:, None, :3, 3])
    Tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    P, S = Tt(P), Tt(np.concatenate(surf, 1))                                  # (F, N, 3), (F, M, 3)
    y = Tt(y_all[frames])                                                      # (F, C)
    channel = torch.as_tensor(tax["channel"], device=dev)
    C = int(y.shape[1])
    sdf = MeshDistance(str(ref["object_collision"]), torch.device(dev), voxel=0.001, signed=True)
    axis = object_long_axis(np.asarray(trimesh.load(str(ref["object_collision"]), force="mesh").vertices))
    Rw, pw = Tt(ref["wrist_rot"][frames]), Tt(ref["wrist_pos"][frames])           # palm -> world
    ray = Tt(ref["obj_pos"][frames] - ref["cam_pos"][frames])
    ray = ray / ray.norm(dim=1, keepdim=True)
    image_axis = lambda a: a - (a * ray).sum(-1, keepdim=True) * ray               # not normalised: foreshortening
    ref_axis = image_axis(Tt(ref["obj_rot"][frames]) @ Tt(axis))
    use_mask = "mask_axis_2d" in ref and not cotracker_axis and ref["mask_axis_valid"][frames].any()
    if use_mask:
        mask_axis, mask_ok = Tt(ref["mask_axis_2d"][frames]), Tt(ref["mask_axis_valid"][frames].astype(float))
        cam_R, cam_t, (fu, fv, pu, pv) = Tt(ref["cam_R"][frames]), Tt(ref["cam_t"][frames]), ref["cam_K"]
        C_w, shift_w = Tt(ref["C"]), Tt(ref["shift"])

        def pixel_axis(centre, a_world):                                         # (B, F, 3) each
            pts = torch.stack([centre, centre + 0.08 * a_world], -2)
            x = torch.einsum("bfkj,fij->bfki", (pts - shift_w) @ C_w, cam_R) + cam_t[None, :, None]
            q = torch.stack([fu * x[..., 0] / x[..., 2] + pu, fv * x[..., 1] / x[..., 2] + pv], -1)
            return F.normalize(q[:, :, 1] - q[:, :, 0], dim=-1)
    ref_pos = Tt(ref["obj_pos"][frames])
    valid = Tt(ref["obj_valid"][frames].astype(float))

    cell_contact = y[:, channel] > 0
    centroid = (P * cell_contact[..., None]).sum((0, 1)) / cell_contact.sum().clamp_min(1)
    B = starts
    g = torch.Generator(device=dev).manual_seed(0)
    rel = np.einsum("fji,fjk->fik", ref["wrist_rot"][frames], ref["obj_rot"][frames])
    U, _, Vt = np.linalg.svd(rel.sum(0))
    R_rel = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    base6 = Tt(R_rel[:, :2].T.reshape(6))
    noise = 0.3 * torch.randn(B, 6, generator=g, device=dev)
    noise[3 * B // 4:] = 3.0 * torch.randn(B - 3 * B // 4, 6, generator=g, device=dev)
    rot6 = (base6[None] + noise).requires_grad_(True)
    trans = (centroid + 0.02 * torch.randn(B, 3, generator=g, device=dev)).requires_grad_(True)
    opt = torch.optim.Adam([rot6, trans], lr=3e-3)
    r = radius

    def losses():
        R = rot6d_to_rotmat(rot6)                                              # (B, 3, 3) object -> palm
        to_obj = lambda X: torch.einsum("bji,bfnj->bfni", R, X[None] - trans[:, None, None])
        d_cell = sdf(to_obj(P))                                                # (B, F, N)
        d = torch.full((B, len(frames), C), float("inf"), device=dev).scatter_reduce(
            2, channel.expand(B, len(frames), -1), d_cell, reduce="amin")
        pos = (F.softplus((d - r) / tau) * tau * y).sum((1, 2)) / y.sum()
        neg = (F.softplus((r - d) / tau) * tau * (1 - y)).sum((1, 2)) / (1 - y).sum()
        pen = torch.relu(-sdf(to_obj(S[::4]))).mean((1, 2))                  # every 4th frame: memory
        a_world = torch.einsum("fij,bjk,k->bfi", Rw, R, Tt(axis))
        c_world = pw[None] + torch.einsum("fij,bj->bfi", Rw, trans)
        if use_mask:
            cos = (pixel_axis(c_world, a_world) * mask_axis[None]).sum(-1)
            axis_cost = ((1 - cos) * mask_ok).sum(1) / mask_ok.sum()
        else:
            diff = (image_axis(a_world) - ref_axis[None]).norm(dim=-1)
            cos = 1 - 2 * (diff / 2).clamp(max=1) ** 2
            axis_cost = ((1 - cos) * valid).sum(1) / valid.sum()
        err = c_world - ref_pos[None]
        across = (err - (err * ray[None]).sum(-1, keepdim=True) * ray[None]).norm(dim=-1)
        across_cost = (torch.relu(across - 0.02) ** 2 * valid).sum(1) / valid.sum()
        total = pos + w_neg * neg + w_pen * pen + w_axis * axis_cost + w_across * across_cost
        return total, d, pen, cos, across

    for _ in range(iters):
        loss = losses()[0]
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        loss, d, pen, cos, across = losses()
        sim = d < r
        tp = (sim & (y[None] > 0)).sum((1, 2)).float()
        prec = tp / sim.sum((1, 2)).clamp_min(1)
        rec = tp / y.sum()
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
        best = int(torch.argmin(loss))
        R = rot6d_to_rotmat(rot6)[best].cpu().numpy()
        p = trans[best].detach().cpu().numpy()
    has_target = (mask_ok if use_mask else valid) > 0          # elsewhere cos is 0 by construction
    ang = torch.rad2deg(torch.acos(cos[best][has_target].clamp(-1, 1)))
    print(f"axis target: {'SAM3 masks' if use_mask else 'CoTracker'}; video agreement: long axis in the image "
          f"{ang.median():.0f} deg median (p90 {ang.quantile(0.9):.0f}), lateral offset {across[best].median() * 100:.1f} cm median")
    print(f"best of {B} starts: contact P {prec[best]:.2f} R {rec[best]:.2f} F1 {f1[best]:.2f}, mean hand penetration "
          f"{pen[best] * 1e3:.2f} mm; object centre {np.round(p * 100, 1)} cm in the palm frame "
          f"(F1 over starts: median {f1.median():.2f}, max {f1.max():.2f})")
    order = torch.argsort(loss)[:keep].cpu().numpy()
    ang_all = torch.rad2deg(torch.acos(cos[:, has_target].clamp(-1, 1))).median(1).values
    ref.update(grasp_candidates_R=rot6d_to_rotmat(rot6)[order].detach().cpu().numpy(),
               grasp_candidates_p=trans[order].detach().cpu().numpy(),
               grasp_candidates_loss=loss[order].cpu().numpy(), grasp_candidates_f1=f1[order].cpu().numpy(),
               grasp_candidates_axis_deg=ang_all[order].cpu().numpy(),
               grasp_candidates_across=across.median(1).values[order].cpu().numpy(),
               grasp_R_palm=R, grasp_p_palm=p, grasp_frames=frames, grasp_f1=float(f1[best]))
    np.savez(reference, **ref)


def measure_object_poses(ref: dict, recon: Path, obj: str, episode: Path, min_visible: int = 40):
    """The object's pose in each frame, solved again (Kabsch + RANSAC on CoTracker's lifted points, no smoothing) from
    the points off both gloves' masks (EgoTouch masks.npz): CoTracker's points slide onto the glove that hides the
    object and stay on it, and a frame is then pulled towards the hand. Frames with fewer than `min_visible` such
    points keep the composed (Kalman) trajectory and 0 inliers. Returns world (pos (T, 3), rot (T, 3, 3)), the RANSAC
    inliers (T,), the points off the gloves (T,) and the fraction of the visible points on a glove (T,; NaN where none
    is visible)."""
    T = len(ref["qpos"])
    C, shift, centre = ref["C"], ref["shift"], ref["obj_centre_cam"]
    comp = np.load(recon / f"{obj}_traj_composed_cotracker.npz")
    Rt, tt = comp["rotation"][:T], comp["translation"][:T]
    rot_w = np.einsum("ij,tjk,lk->til", C, Rt, C)
    pos_w = (np.einsum("tij,j->ti", Rt, centre) + tt) @ C.T + shift
    # each frame's pose solved again from the points off the gloves: CoTracker's points slide onto the glove that
    # hides the object and stay on it, and a frame is then pulled towards the hand
    from tv2s_data.rigid_poses import kabsch_ransac

    pts = np.load(recon / f"{obj}_points_cotracker.npz")
    P3, vis, tracks = pts["points_3d"][:T], pts["visible"][:T], pts["tracks"][:T]
    ref_frame = int(pts["ref_frame"])
    off = vis.copy()
    if (episode / "masks.npz").exists():
        masks = np.load(episode / "masks.npz")["masks"].astype(bool)
        H, W = masks.shape[-2:]
        for t in range(min(T, len(masks))):
            px = np.clip(np.round(tracks[t]).astype(int), 0, [W - 1, H - 1])
            off[t] &= ~(masks[t, 0] | masks[t, 1])[px[:, 1], px[:, 0]]
    src = P3[ref_frame]
    cam_R, cam_t = comp["camera_rotation"][:T], comp["camera_translation"][:T]
    rng = np.random.default_rng(0)
    inl = np.zeros(T, int)
    for t in range(T):
        use = off[t] & np.isfinite(P3[t]).all(1) & np.isfinite(src).all(1)
        if use.sum() < min_visible:
            continue
        R, tr, mask = kabsch_ransac(src[use], P3[t][use], 200, 0.012, rng)
        if R is None:
            continue
        Rc, tc = cam_R[t].T @ R, cam_R[t].T @ (tr - cam_t[t])               # into the reference camera
        rot_w[t] = C @ Rc @ C.T
        pos_w[t] = (Rc @ centre + tc) @ C.T + shift
        inl[t] = int(mask.sum())
    n_vis = vis.sum(1)
    return pos_w, rot_w, inl, off.sum(1), np.where(n_vis > 0, 1 - off.sum(1) / np.maximum(n_vis, 1), np.nan)


def consistent_keyframes(pos: np.ndarray, rot: np.ndarray, reliable: np.ndarray, window: int = 3, min_neighbours: int = 2,
                         max_cm: float = 2.0, max_deg: float = 10.0) -> np.ndarray:
    """The reliable frames that agree with their reliable neighbours within +-`window` frames: within `max_cm` of
    their median position and `max_deg` of their mean rotation (a held object moves smoothly; a frame off by more
    is a RANSAC fit on a few points, or on points that slid onto the glove)."""
    T, kf = len(pos), np.zeros(len(pos), bool)
    for t in np.flatnonzero(reliable):
        nb = [s for s in range(max(0, t - window), min(T, t + window + 1)) if s != t and reliable[s]]
        if len(nb) < min_neighbours:
            continue
        dp = np.linalg.norm(pos[t] - np.median(pos[nb], 0)) * 100
        m = Rotation.from_matrix(rot[nb]).mean()
        dr = np.degrees((m.inv() * Rotation.from_matrix(rot[t])).magnitude())
        kf[t] = dp <= max_cm and dr <= max_deg
    return kf


def keyframe_object_trajectory(reference: Path, recon: Path, obj: str, episode: Path, min_inliers: int = 30,
                               min_visible: int = 40, blend: int = 6, anchor: str = "vision",
                               max_spread_deg: float = 15.0, max_spread_cm: float = 3.0, rest_tol: float = 0.04,
                               target: str = "dense", held_weight: float = 0.5, held_margin: float = 3.0) -> None:
    """The object's trajectory from its reliable frames only, and from the hand while it is held.

    CoTracker's object trajectory fails exactly when it matters: the grasping hand hides the object, its points slide
    onto the glove and stay there, and the Kalman filter bridges half the clip at constant velocity
    (pick_up_wrench/094952: poses kept on 18-46% of the held frames, a third of the points on the glove). So the clip
    is cut by the glove's holds (tv2s_data.reference.contact_holds) and the object is
      at rest (before the first hold, between and after them)  one pose: the robust mean of its reliable poses there;
      held                                                      the palm's pose times one offset palm -> object: the
                                                                robust mean of the offsets on the hold's reliable frames
                                                                (`anchor` vision) when they agree (spread under
                                                                `max_spread_deg` and `max_spread_cm`: a held object does
                                                                not move by more in the palm), else, or with `anchor`
                                                                tactile, the in-palm pose that explains the glove and
                                                                that the hand holds under physics (add_grasp_pose,
                                                                select_grasp_pose);
    blended over `blend` frames at each grasp and release. The glove reports a hold under a firm grip only: a stretch
    "at rest" whose reliable poses are still away from its rest pose (> `rest_tol`) next to a hold (the object carried
    more lightly after the release, or already lifted before the onset) is held too, until the object is seen at rest. The poses are solved again on every frame (Kabsch + RANSAC on
    CoTracker's lifted points, no smoothing) from the points off both gloves' masks (EgoTouch masks.npz): the points
    slide onto the glove that hides the object and stay there. A reliable frame: >= `min_visible` such points and
    >= `min_inliers` RANSAC inliers. The tracked trajectory is kept as obj_pos_tracked / obj_rot_tracked;
    obj_traj_source, obj_keyframes and hold_offset_spread_{deg,cm} report what was done; the measured poses themselves
    are kept as obj_pos_measured / obj_rot_measured with their RANSAC inliers (obj_inliers) and obj_kf, the reliable
    frames consistent with their neighbours (consistent_keyframes): the poses eval_rl_recon.py scores the object on."""
    import trimesh

    from tv2s_data.reference import contact_holds

    ref = dict(np.load(reference, allow_pickle=True))
    if "obj_pos_tracked" in ref:                 # run again: from the tracked trajectory, not the previous result
        ref["obj_pos"], ref["obj_rot"] = ref["obj_pos_tracked"], ref["obj_rot_tracked"]
    recon, episode = Path(recon), Path(episode)
    T = len(ref["qpos"])
    pos_w, rot_w, inl, _, on_glove = measure_object_poses(ref, recon, obj, episode, min_visible)
    reliable = inl >= min_inliers
    kf = consistent_keyframes(pos_w, rot_w, reliable)
    def robust_mean(R, p):
        """Rotation mean and median translation of the poses within 2 x the median angle of the first mean."""
        rots = Rotation.from_matrix(R)
        m = rots.mean()
        ang = np.degrees((m.inv() * rots).magnitude())
        keep = ang <= max(10.0, 2 * np.median(ang))
        m = rots[keep].mean()
        p0 = np.median(p[keep], 0)
        spread = (np.median(np.degrees((m.inv() * rots[keep]).magnitude())), np.median(np.linalg.norm(p[keep] - p0, axis=1)))
        return m.as_matrix(), p0, spread

    side_holds = contact_holds(ref["real_contact"] > 0)
    out_R, out_p = np.zeros((T, 3, 3)), np.zeros((T, 3))
    source = np.array(["rest"] * T, dtype=object)
    report, spreads, anchors = [], [], []
    # held: the palm times one offset
    Rp, pp = ref["wrist_rot"], ref["wrist_pos"]
    for a, b in side_holds:
        frames = np.arange(a, min(b, T))
        good = frames[reliable[frames]]
        how, spread = anchor, (np.nan, np.nan)
        if anchor == "vision" and len(good) >= 5:
            A_R = np.einsum("tji,tjk->tik", Rp[good], rot_w[good])                         # palm^T object
            A_p = np.einsum("tji,tj->ti", Rp[good], pos_w[good] - pp[good])
            R_off, p_off, spread = robust_mean(A_R, A_p)
            spreads.append(spread)
            if spread[0] > max_spread_deg or spread[1] * 100 > max_spread_cm:
                how = "tactile"                  # the video's offsets disagree: not a reliable view of the held object
        if how != "vision":
            if "grasp_R_palm" not in ref:
                raise SystemExit(f"hold {a}-{b}: {len(good)} reliable frames and no grasp pose (add_grasp_pose)")
            R_off, p_off, how = ref["grasp_R_palm"], ref["grasp_p_palm"], "tactile"
        out_R[frames] = np.einsum("tij,jk->tik", Rp[frames], R_off)
        out_p[frames] = pp[frames] + np.einsum("tij,j->ti", Rp[frames], p_off)
        source[frames] = f"held ({how})"
        anchors.append((a, min(b, T), R_off, p_off))
        seen = (f"the video's offsets on {len(good)} reliable frames spread {spread[0]:.0f} deg / {spread[1] * 100:.1f} cm"
                if np.isfinite(spread[0]) else f"{len(good)} reliable frames")
        report.append(f"held {a}-{b - 1}: offset from {'the video' if how == 'vision' else 'the tactile grasp pose'} ({seen})")
    # at rest: one pose per stretch between holds
    held = source != "rest"
    stretches, t = [], 0
    while t < T:
        if not held[t]:
            s = t
            while t < T and not held[t]:
                t += 1
            stretches.append((s, t))
        else:
            t += 1
    def hold_with(frames, R_off, p_off):
        out_R[frames] = np.einsum("tij,jk->tik", Rp[frames], R_off)
        out_p[frames] = pp[frames] + np.einsum("tij,j->ti", Rp[frames], p_off)
        source[frames] = "held (extended)"

    for s, e in stretches:
        prev = next((h for h in anchors if h[1] == s), None)          # the hold ending where this stretch starts
        nxt = next((h for h in anchors if h[0] == e), None)           # the hold starting where it ends
        frames = np.arange(s, e)
        good = frames[reliable[frames]]
        if len(good) >= 3:
            # the rest pose from where the object surely rests: away from the hand that brings or takes it
            pick = good[-15:] if prev is not None and nxt is None else good[:15] if nxt is not None and prev is None else good
            R0, p0, _ = robust_mean(rot_w[pick], pos_w[pick])
            how = f"{len(pick)} reliable frames"
            # the glove reports a hold only under a firm grip: the object can still be in the hand, carried more
            # lightly, after the glove's release (or before its onset); held until it is seen at rest
            settled = good[np.linalg.norm(pos_w[good] - p0, axis=1) <= rest_tol]
            if len(settled):
                if prev is not None and settled[0] > s:
                    hold_with(np.arange(s, settled[0]), prev[2], prev[3])
                    report.append(f"held {s}-{settled[0] - 1} too (the object still away from its rest pose after the glove's release)")
                    s = settled[0]
                if nxt is not None and settled[-1] < e - 1:
                    hold_with(np.arange(settled[-1] + 1, e), nxt[2], nxt[3])
                    report.append(f"held {settled[-1] + 1}-{e - 1} too (the object already away from its rest pose before the glove's onset)")
                    e = settled[-1] + 1
        else:                                    # none seen: where the hand left it (or will take it)
            k = s - 1 if s > 0 else e
            if k >= T:                           # no hold at all: the tracked trajectory's first pose
                R0, p0, how = rot_w[0], pos_w[0], "the tracked first frame"
            else:
                R0, p0, how = out_R[k], out_p[k], "the hand's pose at the boundary"
        frames = np.arange(s, e)
        out_R[frames], out_p[frames] = R0, p0
        report.append(f"rest {s}-{e - 1}: one pose from {how}")
    # blend at each grasp and release; how far the object jumps there (the rest pose against the held one)
    is_held = np.array([str(x).startswith("held") for x in source])
    edges = np.flatnonzero(np.diff(is_held.astype(int))) + 1
    report.append("jumps blended over %d frames: %s" % (blend, ", ".join(
        f"{np.linalg.norm(out_p[e] - out_p[e - 1]) * 100:.1f} cm at {e}" for e in edges)))
    for edge in edges:
        lo, hi = max(0, edge - blend // 2), min(T - 1, edge + blend // 2)
        if hi - lo < 2:
            continue
        w = np.linspace(0, 1, hi - lo + 1)
        ends = Rotation.from_matrix(np.stack([out_R[lo], out_R[hi]]))
        out_R[lo:hi + 1] = slerp_matrices(ends, w)
        out_p[lo:hi + 1] = out_p[lo] + w[:, None] * (out_p[hi] - out_p[lo])
    # the frames the RL scores the object on
    measured = ~is_held.copy()
    for a, b, _, _ in anchors:
        if any(r.startswith(f"held {a}-") and "from the video" in r for r in report):
            measured[a:b] |= reliable[a:b]
    for edge in edges:
        measured[max(0, edge - blend // 2):min(T, edge + blend // 2 + 1)] = False
    strict = measured if target == "measured" else np.ones(T, bool)
    obj_weight = np.where(strict, 1.0, held_weight)
    obj_margin_scale = np.where(strict, 1.0, held_margin)
    report.append(f"object target: {target} ({strict.sum()}/{T} frames strict"
                  + (f", the others weight {held_weight:g}, margins x {held_margin:g})" if target == "measured" else ")"))
    # the frame-0 lift putting the object on the table, for the new rest pose
    V = np.asarray(trimesh.load(Path(reference).parent / "object_visual.obj", process=False).vertices)
    z0 = (V @ out_R[0].T + out_p[0])[:, 2].min()
    err = np.linalg.norm(out_p - pos_w, axis=1)[reliable]
    print(f"object trajectory from keyframes: {reliable.sum()}/{T} reliable frames (>= {min_inliers} inliers among the "
          f"points off the gloves; {np.nanmedian(on_glove) * 100:.0f}% of the visible points on a glove, "
          f"median); " + "; ".join(report))
    print(f"  against the tracked poses on the reliable frames: median {np.median(err) * 100:.1f} cm; "
          f"lowest point at frame 0: z = {z0 * 100:.1f} cm")
    ref.update(obj_pos_tracked=ref["obj_pos"], obj_rot_tracked=ref["obj_rot"], obj_pos=out_p, obj_rot=out_R,
               obj_valid=np.ones(T, bool), obj_weight=obj_weight, obj_margin_scale=obj_margin_scale,
               obj_measured=measured, obj_target=target,
               obj_pos_measured=pos_w, obj_rot_measured=rot_w, obj_inliers=inl, obj_kf=kf, obj_rest_z=-z0, obj_traj_source=source.astype(str), obj_keyframes=reliable,
               hold_offset_spread_deg=np.array([s[0] for s in spreads]), hold_offset_spread_cm=np.array([s[1] * 100 for s in spreads]))
    np.savez(reference, **ref)


def slerp_matrices(ends: Rotation, w: np.ndarray) -> np.ndarray:
    """Rotation matrices between the two `ends` at weights w in [0, 1]."""
    from scipy.spatial.transform import Slerp

    return Slerp([0.0, 1.0], ends)(w).as_matrix()
