#!/usr/bin/env python3
"""Per-episode RL reference (reference.npz, read by tv2s_data/reference.py) of one reconstruction (launch/reconstruct.sh)
for the rl_recon environment (rl_recon_env.py): Benchmark-Genesis' rl_recon reference chain.

  1  retarget_mocap_wuji.py   the mocap hand retargeted onto the Wuji Hand                          [PY_RETARGET]
  2  build_reference          the simulated world: z up, table at z = 0, the object's mesh and CoTracker trajectory,
                              the hand lifted clear of the table, the glove's channels               (tv2s_data.recon_reference)
  3  add_mask_axes            the object's long axis per frame from its SAM3 mask
  4  add_grasp_pose           the in-palm object pose that best explains the glove (in-hand starts, assist)
  5  select_grasp_pose        among its candidates, the one the retargeted hand holds under physics (here)
  6  keyframe_object_trajectory   the object at rest where it is reliably tracked, and moving with the palm while
                              the glove holds it (--object-trajectory tracked: CoTracker's trajectory as it is)

One hand: the scene's `hands_used` (semantic_parsing.py, --scene), else --side, else the hand whose glove fires; a
scene using both hands is refused (the chain is one-handed). Writes <out>/reference.npz, object_visual.obj,
object_collision.obj, wuji_ref_<side>.npz, reference_info.json, reference_source.json.

    python scripts/rl/prepare_reference.py --recon data/egotouch/recon/pick_up_screwdriver__20260321_103227_213 \\
        --scene data/egotouch/scene/pick_up_screwdriver__20260321_103227_213.json --out data/references/et_screwdriver
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.recon_reference import add_grasp_pose, add_mask_axes, build_reference, keyframe_object_trajectory  # noqa: E402
from tv2s_data.reference import load_reference  # noqa: E402

CONDA = Path(os.environ.get("CONDA_BASE", Path.home() / "miniconda3"))
PY_RETARGET = os.environ.get("PY_RETARGET", str(CONDA / "envs" / "wuji_retargeting" / "bin" / "python"))


def hands_from_scene(scene: Path | None, default: list[str] | None) -> list[str]:
    """The hands the demonstration uses: the scene's hands_used, else `default`."""
    if scene is not None and scene.exists():
        s = json.loads(scene.read_text())
        if "hands_used" in s:
            used = s["hands_used"]
            print(f"scene {scene}: hands used: {used} ({s.get('hands_reason', '')})")
            return ["left", "right"] if used == "both" else [used]
        print(f"scene {scene} has no hands_used")
    if not default:
        raise SystemExit("which hand? --scene with hands_used, or --side")
    return default


def glove_side(episode: Path) -> str:
    """The hand whose glove reports the most contact (TouchAnything grid, > 0.1)."""
    z = np.load(episode / "pressure_grids.npz")
    n = {s: float((np.nan_to_num(z[f"{s}_pressure_grid"]) > 0.1).sum()) for s in ("left", "right")}
    return max(n, key=n.get)


def select_grasp_pose(reference: Path, frames: int = 40, max_slip: float = 0.02, start: int | None = None) -> None:
    """Each grasp-pose candidate (add_grasp_pose) placed in the reference hand at the first firm-grasp frame, one env
    per candidate, simulated with a zero residual (the retargeted hand, PD-tracked), no assist and no termination for
    `frames` frames. Held: the object moves less than `max_slip` in the palm frame and stays above the table. The
    held candidate of lowest optimisation loss (glove + video) replaces grasp_R_palm / grasp_p_palm; none held keeps
    the optimiser's best (the residual can still learn the finger corrections that hold it). The optimiser alone has
    no notion of stability: on the screwdriver its best video-consistent pose fell out of the hand in 11 frames."""
    import torch
    from scipy.spatial.transform import Rotation

    from tv2s_env.sim.envs.config.schema import ReferenceDataArgs, RlReconEnvArgs
    from tv2s_env.sim.envs.manipulation.rl_recon_env import RlReconEnv

    ref = dict(np.load(reference, allow_pickle=True))
    Rc, pc = ref["grasp_candidates_R"], ref["grasp_candidates_p"]
    K = len(Rc)
    env = RlReconEnv(RlReconEnvArgs(data=ReferenceDataArgs(reference=str(reference))), K, eval_mode=True)
    env.disable_termination = True
    if start is None and not len(env.rsi_frames):
        print("no firm-grasp frame where the held object clears the table: keeping the optimiser's best pose")
        return
    t0 = int(env.rsi_frames[0]) if start is None else start
    dev = env.device
    idx = torch.arange(K, device=dev)
    t = torch.full((K,), t0, dtype=torch.long, device=dev)
    env._reset_hands_and_objects(idx, t, torch.ones(K, dtype=torch.bool, device=dev))
    env.t[:], env.start[:], env.horizon[:] = t0, t0, env.T - 1
    Rw, pw = ref["wrist_rot"][t0], ref["wrist_pos"][t0]
    q = Rotation.from_matrix(np.einsum("ij,kjl->kil", Rw, Rc)).as_quat()
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    obj = env.scene_entities[0]
    obj.set_pos(T(pw + pc @ Rw.T), envs_idx=idx, zero_velocity=True)
    obj.set_quat(T(np.c_[q[:, 3], q[:, :3]]), envs_idx=idx, zero_velocity=True)
    env.update_buffers()

    def in_palm():
        return ((env.all_object_pos[:, 0] - env.base_pos[:, 0])[:, None] @ env.base_rot[:, 0])[:, 0], env.all_object_pos[:, 0, 2]

    rel0, _ = in_palm()
    zero = torch.zeros(K, env.action_dim, device=dev)
    f1_sum = torch.zeros(K, device=dev)
    for _ in range(min(frames, env.T - 1 - t0)):
        info = env.step(zero)[-1]["metrics"]
        p = info["tac_tp"] / info["tac_sim"].clamp_min(1)
        r = info["tac_tp"] / info["tac_real"].clamp_min(1)
        f1_sum += 2 * p * r / (p + r).clamp_min(1e-9)
    rel1, z = in_palm()
    slip = (rel1 - rel0).norm(dim=1).cpu().numpy()
    held = (slip < max_slip) & (z.cpu().numpy() > 0.02)
    f1_sim = (f1_sum / frames).cpu().numpy()
    print(f"{K} candidates tested at frame {t0} for {frames} frames: {held.sum()} held")
    print(" #   loss    F1(opt)  F1(sim)  axis(deg)  lateral(cm)  slip(cm)  held")
    for k in range(K):
        print(f"{k:2d} {ref['grasp_candidates_loss'][k]:.5f}  {ref['grasp_candidates_f1'][k]:5.2f}    {f1_sim[k]:5.2f}"
              f"    {ref['grasp_candidates_axis_deg'][k]:6.0f}     {ref['grasp_candidates_across'][k] * 100:6.1f}"
              f"     {slip[k] * 100:6.1f}   {'yes' if held[k] else ''}")
    if not held.any():
        print("no candidate is held by the uncorrected hand: keeping the optimiser's best pose")
        return
    best = int(np.flatnonzero(held)[0])                       # candidates are sorted by loss
    print(f"-> candidate {best}")
    ref.update(grasp_R_palm=Rc[best], grasp_p_palm=pc[best], grasp_f1=float(ref["grasp_candidates_f1"][best]),
               grasp_selected=best)
    np.savez(reference, **ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="launch/reconstruct.sh output folder")
    ap.add_argument("--out", required=True, help="folder of reference.npz")
    ap.add_argument("--episode", default=None, help="the episode folder (pressure_grids.npz: EgoTouch's, or extract_ace.py's); "
                    "default <recon>/source")
    ap.add_argument("--object", default=None, help="object id (default: the one with a trajectory and a mesh)")
    ap.add_argument("--scene", default=None, help="scene JSON with hands_used (semantic_parsing.py)")
    ap.add_argument("--side", choices=("left", "right"), default=None, help="without hands_used: the hand")
    ap.add_argument("--hand-table-clearance", type=float, default=0.008,
                    help="lift the hand so its lowest GT joint clears the table by this much (m); < 0: off")
    ap.add_argument("--clearance-mode", default="global", choices=["global", "frame"],
                    help="global: one lift for the whole clip; frame: each frame lifted by what it needs, smoothed "
                    "(a dip after the release must not lift the grasp)")
    ap.add_argument("--object-trajectory", choices=("keyframes", "tracked"), default="keyframes",
                    help="keyframes: the object at rest on its reliable frames and moving with the palm while the glove "
                    "holds it (keyframe_object_trajectory); tracked: the reconstruction's CoTracker + Kalman trajectory")
    ap.add_argument("--anchor", choices=("vision", "tactile"), default="vision",
                    help="keyframes: the held object's offset in the palm from its reliable frames (vision, else the "
                    "tactile grasp pose) or from the tactile grasp pose")
    ap.add_argument("--object-target", choices=("dense", "measured"), default="dense",
                    help="keyframes: how the RL scores the object: alike on every frame (dense), or strictly where its "
                    "pose is measured (at rest, the reliable frames of a hold seen in the video) and loosely while it is "
                    "held elsewhere (measured: --held-weight, --held-margin)")
    ap.add_argument("--held-weight", type=float, default=0.5, help="measured: weight of the object terms on the unmeasured held frames")
    ap.add_argument("--held-margin", type=float, default=3.0, help="measured: their position margins x this")
    ap.add_argument("--overwrite", action="store_true", help="redo the retargeting")
    args = ap.parse_args()
    recon, out = Path(args.recon).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    episode = Path(args.episode).expanduser().resolve() if args.episode else recon / "source"
    if not (episode / "pressure_grids.npz").exists():
        raise SystemExit(f"no pressure_grids.npz in {episode} (--episode: the EgoTouch episode folder)")
    scene = Path(args.scene) if args.scene else None
    default = [args.side] if args.side else None
    if default is None and not (scene and scene.exists()):
        default = [glove_side(episode)]
        print(f"no scene: the hand whose glove fires: {default[0]}")
    sides = hands_from_scene(scene, default)
    if len(sides) != 1:
        raise SystemExit(f"the reference chain simulates one hand; this scene uses {sides} (--side to pick one)")
    side = sides[0]
    objs = [p.name[: -len("_traj_cotracker.npz")] for p in sorted(recon.glob("*_traj_cotracker.npz"))]
    objs = [o for o in objs if (recon / "recgen_out_moge" / o / "posed_mesh.obj").exists()]
    if not args.object and len(objs) != 1:
        raise SystemExit(f"--object: one of {objs}")
    obj = args.object or objs[0]
    for f in ("mocap_joints_cotracker.npz", f"{obj}_traj_cotracker.npz", f"recgen_out_moge/{obj}/posed_mesh.obj",
              "camera_traj_cotracker.npz", "ground_plane_moge.json", "intrinsics.yaml"):
        if not (recon / f).exists():
            raise SystemExit(f"missing {recon / f}: run launch/reconstruct.sh first")

    print(f"=== reference: {recon.name}, object {obj}, {side} hand -> {out}")
    wuji_ref = out / f"wuji_ref_{side}.npz"
    if args.overwrite or not wuji_ref.exists():
        print("1/6 the mocap hand retargeted onto the Wuji Hand")
        video = recon / "video.json"                     # reconstruct.sh: the video's frame rate (EgoTouch 30, ACE 20)
        fps = json.loads(video.read_text())["fps"] if video.exists() else 30.0
        cmd = [PY_RETARGET, str(REPO / "scripts/rl/retarget_mocap_wuji.py"), "--hand-meshes", str(recon / "mocap_joints_cotracker.npz"),
               "--side", side, "--out", str(wuji_ref), "--fps", str(fps)]
        print("  $", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print("2/6 reference (world frame)")
    path = build_reference(recon, episode, obj, side, wuji_ref, out, hand_table_clearance=args.hand_table_clearance,
                           clearance_mode=args.clearance_mode)
    print("3/6 object axis per frame from the SAM3 masks")
    add_mask_axes(path, recon, obj)
    print("4/6 in-hand object pose that best explains the glove")
    add_grasp_pose(path)
    print("5/6 which candidate grasp poses the hand holds")
    select_grasp_pose(path)
    if args.object_trajectory == "keyframes":
        print("6/6 the object's trajectory: at rest from its reliable frames, held with the palm")
        keyframe_object_trajectory(path, recon, obj, episode, anchor=args.anchor, target=args.object_target,
                                   held_weight=args.held_weight, held_margin=args.held_margin)

    r = load_reference(path)
    print(f"-> {path}: {len(r['qpos'][0])} frames at {r['fps']:g} Hz, hands {r['sides']}, {len(r['holds'])} holds "
          f"{[(r['sides'][h], int(a), int(b)) for h, _, a, b in r['holds']]}, {len(r['rsi_frames'])} in-hand start "
          f"candidates, glove channels {r['real_contact'].shape[-1]}")
    (out / "reference_source.json").write_text(json.dumps(dict(
        recon=str(recon), episode=str(episode), scene=str(scene.resolve()) if scene else None, sides=r["sides"]), indent=1))


if __name__ == "__main__":
    main()
