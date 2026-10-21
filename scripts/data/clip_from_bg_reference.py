#!/usr/bin/env python3
"""An episode clip (build_episode_clip.py's format) from Benchmark-Genesis' reconstruction of an EgoTouch video.

Benchmark-Genesis' rl_recon reference (<recon>/rl/reference.npz: one hand, one object) already has the reconstructed
object (its RecGen mesh, object_collision.obj, and its trajectory, locked to the hand while held) and the retargeted
mocap hand in its simulation frame (table at z = 0). This writes them as a whole-episode clip, so that every
BimanualTactileEnv method (DexTaG, rl_recon on clips) and render_rollouts.py run on the real object:
  - the hand of the reference (its palm pose and Wuji joints; fingertips and keypoints by forward kinematics);
  - the glove: this pipeline's EgoTouch cut (<data>/cuts/<prefix>_c00.npz: the 151 normalised taxels, mapping
    mano_egotouch), frame for frame (both are the episode's 30 Hz frames);
  - the object: its mesh, and its pose at every frame. --objects tracked (default): the reference's trajectory
    (CoTracker + MoGe depth), lifted onto the table at frame 0 (the reference has it a few mm into it), the guide the
    rl_recon object terms follow with their margins; it is ~7 cm off the hand during the hold (the hand hides the
    object). Per hold (the glove's contact runs) the object also gets `hold_offsets`, its pose in the palm while held:
    the reference's tactile grasp pose (grasp_R_palm / grasp_p_palm, Benchmark-Genesis' tactile_grasp_pose.py), where
    rl_recon's assist carries it (recon_env.py), as Benchmark-Genesis' hold_R_palm / hold_p_palm.
    --objects anchored: the trajectory itself in the palm at that pose while held, standing on the table before and
    after (build_episode_clip.anchor); replayed on the reference hand, the glove F1 is 0.004 tracked, 0.39 anchored.
The hands used come from the Gemini scene (<data>/scene/<key>.json: hands_used, semantic_parsing.py) when it exists,
else from the reference (Benchmark-Genesis reconstructs the one hand it uses). Only the used hand gets the object on its
schedule, so --env.active_hands=auto simulates it alone (the other hand is not in the scene).

    python scripts/data/clip_from_bg_reference.py \
        --recon $SCRATCH/egotouch/recon/Workbench/pick_up_screwdriver/20260321_103227_213 \
        --data data/egotouch --out data/episodes/et_screwdriver_103227
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
from build_episode_clip import _close, _runs, anchor  # noqa: E402
from tv2s_data.clips import wuji_reference_geometry  # noqa: E402
from tv2s_data.cuts import EGOTOUCH_GRASP_MIN_CHANNELS  # noqa: E402
from tv2s_data.egotouch import LAYOUT_JSON  # noqa: E402
from tv2s_data.humantouch import SIDES  # noqa: E402
from tv2s_env.robots.wuji_model import WujiHand  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="Benchmark-Genesis reconstruction folder (rl/reference.npz)")
    ap.add_argument("--data", default="data/egotouch", help="prepared EgoTouch folder (cuts/)")
    ap.add_argument("--out", required=True, help="clip folder to write")
    ap.add_argument("--mass", type=float, default=0.08, help="object mass (kg; rl_recon's default 80 g)")
    ap.add_argument("--clearance", type=float, default=0.001, help="--objects tracked: object above the table at frame 0 (m)")
    ap.add_argument("--objects", choices=("tracked", "anchored"), default="tracked",
                    help="tracked: the reference's trajectory; anchored: in the palm at the tactile grasp pose while held")
    ap.add_argument("--name", default=None, help="clip name (default: <recording time>_episode)")
    args = ap.parse_args()
    recon = Path(args.recon)
    ref = np.load(recon / "rl" / "reference.npz")
    info = json.loads((recon / "rl" / "reference_info.json").read_text())
    side, hz, T = str(ref["side"]), float(ref["fps"]), len(ref["qpos"])
    episode = Path(info["episode"])
    key = f"{episode.parent.name}__{episode.name}"                       # EgoTouch key: <task>__<time>
    prefix = episode.name.replace("_", "")
    cut = np.load(Path(args.data) / "cuts" / f"{prefix}_c00.npz")
    scene = Path(args.data) / "scene" / f"{key}.json"
    hands = json.loads(scene.read_text()).get("hands_used") if scene.exists() else None
    if hands is not None and hands != side:
        raise SystemExit(f"{scene}: hands_used={hands}, the reconstruction has the {side} hand only")
    print(f"hands used: {side} (" + (f"Gemini, {scene}" if hands else "the reference: no Gemini scene") + ")")
    if len(cut[f"{side}_pressure"]) != T:
        raise SystemExit(f"glove {len(cut[f'{side}_pressure'])} frames, reference {T}")

    # the object: mesh, poses (gaps: the last valid pose)
    import trimesh

    mesh = str(ref["object_collision"])
    verts = np.asarray(trimesh.load(mesh, force="mesh").vertices)
    poses = np.tile(np.eye(4), (T, 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = ref["obj_rot"], ref["obj_pos"]
    bad = ~ref["obj_valid"]
    if bad.any():                                                       # hold the last valid pose
        ok = np.flatnonzero(~bad)
        poses[bad] = poses[ok[np.searchsorted(ok, np.flatnonzero(bad)).clip(max=len(ok) - 1)]]

    # glove and holds (build_episode_clip.py's: contact runs, gaps < 0.2 s closed, >= 0.3 s)
    pressure, contact = cut[f"{side}_pressure"].astype(np.float32), cut[f"{side}_contact"]
    on = _close(contact.sum(1) >= EGOTOUCH_GRASP_MIN_CHANNELS, int(round(0.2 * hz)))
    holds = [(side, int(a), int(b)) for a, b in _runs(on) if b - a >= int(round(0.3 * hz))]
    palm = np.tile(np.eye(4), (T, 1, 1))
    palm[:, :3, :3], palm[:, :3, 3] = ref["wrist_rot"], ref["wrist_pos"]
    grasp = np.eye(4)
    grasp[:3, :3], grasp[:3, 3] = ref["grasp_R_palm"], ref["grasp_p_palm"]
    if args.objects == "anchored":
        tracked = poses.copy()
        poses, dev = anchor(tracked, holds, {side: palm}, verts, blend=4, around=int(round(0.3 * hz)),
                            offsets=[grasp] * len(holds), drops=[(0.0, 0.0)] * len(holds))
        lift = 0.0
        print(f"object anchored in the palm over {len(holds)} holds: median {dev * 100:.1f} cm from the video track")
    else:
        lift = args.clearance - float((verts @ poses[0, :3, :3].T + poses[0, :3, 3])[:, 2].min())
        poses[:, 2, 3] += lift
        tracked = poses
    obj_id = str(info.get("object", "object"))
    objects = [dict(id=obj_id, mesh=mesh, fixed=False, mass=args.mass, material=None, poses=poses.astype(np.float32),
                    tracked_poses=tracked.astype(np.float32), holds=holds, hold_offsets=[grasp.astype(np.float32)] * len(holds),
                    hold_clips=[None] * len(holds), grasp_fits=[])]

    name = args.name or f"{prefix}_episode"
    clip = dict(metadata=dict(name=name, episode=key, dataset="egotouch", clip=0, start=0, end=T, hz=hz,
                              source_frames=np.arange(T, dtype=np.int64), types=["episode"],
                              captions=[episode.parent.name.replace("_", " ")], scene=None, episode_mode=True,
                              hands_source="benchmark_genesis", tactile_available=True, tactile_mapping=str(LAYOUT_JSON),
                              glove_scale=1.0, taxels="mano_egotouch", finger_tip_row=22, clips=[],
                              bg_recon=str(recon), objects=args.objects, object_lift=lift),
                head=dict(pos=ref["cam_pos"].astype(np.float32)), hands={}, scene_objects=objects)
    R = ref["wrist_rot"]
    p = ref["wrist_pos"].astype(np.float64)
    quat = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    for s in SIDES:
        g = wuji_reference_geometry(WujiHand(s), ref["qpos"], p, quat)    # the other hand: a copy, never simulated
        clip["hands"][s] = dict(
            wrist_positions=p.astype(np.float32), wrist_quat=quat.astype(np.float32),
            dof_positions=ref["qpos"].astype(np.float32), human_keypoints=ref["human_kp"].astype(np.float32),
            fingertip_pos=g.tip_pos.astype(np.float32), fingertip_quat=g.tip_quat.astype(np.float32),
            ref_keypoints=g.keypoints.astype(np.float32),
            tactile_pressure=pressure if s == side else np.zeros_like(pressure),
            tactile_contact=contact if s == side else np.zeros_like(contact),
            object=None, object_schedule=np.zeros(T, int) if s == side else np.full(T, -1), semantic=None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(clip, f)
    (out / "index.json").write_text(json.dumps(dict(clips=[dict(name=name, episode=key, frames=T)]), indent=1))
    kp_err = np.linalg.norm(clip["hands"][side]["ref_keypoints"] - ref["ref_keypoints"], axis=-1).max()
    print(f"{name}: {T} frames @ {hz:g} Hz ({T / hz:.1f} s), {side} hand, object {obj_id} ({args.objects}), "
          f"holds {[(a, b) for _, a, b in holds]}, FK vs reference keypoints max {kp_err * 1000:.1f} mm -> {out / name}.pkl")


if __name__ == "__main__":
    main()
