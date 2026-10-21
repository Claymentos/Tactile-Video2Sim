#!/usr/bin/env python3
"""Raw replay in the Genesis viewer: the retargeted Wuji hand(s) and the objects on their raw trajectories.

Kinematic (no physics, no policy, no contact): every frame, each Wuji hand is put on its retargeted pose (palm
pose + 20 finger joints, the episode clip's hands) and each object on its trajectory:
  --objects tracked    the raw tracked trajectory (CoTracker + MoGe depth, interpolated to the clip rate), as
                       reconstructed, before any anchoring (default)
  --objects anchored   the trajectory the RL tracks (build_episode_clip.py: held objects follow the palm)
Objects are drawn translucent green, so a hand passing through one shows.

    python scripts/reconstruction/replay_raw_genesis.py --clip data/episodes/et_bottle_557/20260314000557626_episode.pkl
    python scripts/reconstruction/replay_raw_genesis.py --clip <pkl> --sides left --objects anchored --speed 0.5
    python scripts/reconstruction/replay_raw_genesis.py --clip <pkl> --record outputs/raw_replay.mp4 --no-viewer
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="episode clip .pkl (scripts/data/build_episode_clip.py)")
    ap.add_argument("--sides", default="auto", help="left, right, both, or auto (the hands that manipulate an object)")
    ap.add_argument("--objects", choices=("tracked", "anchored"), default="tracked")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--loops", type=int, default=0, help="number of passes (0: until the viewer is closed)")
    ap.add_argument("--record", default=None, help="also write an mp4 from a fixed camera")
    ap.add_argument("--no-viewer", action="store_true", help="no window (with --record)")
    args = ap.parse_args()

    import genesis as gs

    from tv2s_env.robots.wuji_model import BASE_JOINTS, WujiHand, base_coordinates, default_urdf, mean_rotation

    clip = pickle.load(open(args.clip, "rb"))
    meta, T = clip["metadata"], int(clip["metadata"]["end"] - clip["metadata"]["start"])
    hz = float(meta["hz"])
    if args.sides == "auto":
        sides = [s for s in ("left", "right") if (np.asarray(clip["hands"][s]["object_schedule"]) >= 0).any()] or ["left", "right"]
    else:
        sides = ["left", "right"] if args.sides == "both" else [args.sides]
    rot = lambda q: Rotation.from_quat(np.asarray(q, float)[:, [1, 2, 3, 0]]).as_matrix()        # wxyz -> matrix

    centre = np.mean([np.asarray(clip["hands"][s]["wrist_positions"], float)[:T].mean(0) for s in sides], 0)
    view = dict(pos=tuple(centre + [0.45, -0.45, 0.3]), lookat=tuple(centre))
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1.0 / hz, gravity=(0.0, 0.0, 0.0)),
                     rigid_options=gs.options.RigidOptions(enable_collision=False),
                     viewer_options=gs.options.ViewerOptions(camera_pos=view["pos"], camera_lookat=view["lookat"],
                                                             max_FPS=int(hz * args.speed)),
                     show_viewer=not args.no_viewer)
    scene.add_entity(gs.morphs.Plane())
    hands = []
    work = REPO / "outputs" / "urdf"
    work.mkdir(parents=True, exist_ok=True)
    for s in sides:
        h = clip["hands"][s]
        R, p = rot(h["wrist_quat"])[:T], np.asarray(h["wrist_positions"], float)[:T]
        R0 = mean_rotation(R)
        model = WujiHand(s, urdf=default_urdf(s))
        urdf = model.write_floating_urdf(str(work / f"wuji_{s}_raw_replay.urdf"), R0, base_mass=0.01)
        e = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=True, merge_fixed_links=True),
                             surface=gs.surfaces.Default(color=(1.0, 0.55, 0.1)))
        q = np.concatenate([base_coordinates(R0, p, R), np.asarray(h["dof_positions"], float)[:T]], 1)   # (T, 26)
        hands.append((e, BASE_JOINTS + model.dof_names, q))
    objects = []
    for o in clip["scene_objects"]:
        P = np.asarray(o["tracked_poses"] if args.objects == "tracked" else o["poses"], float)[:T]
        e = scene.add_entity(gs.morphs.Mesh(file=o["mesh"], fixed=False, collision=False),
                             surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2, 0.55)))
        quat = Rotation.from_matrix(P[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        objects.append((e, P[:, :3, 3], quat))
    cam = scene.add_camera(res=(960, 720), pos=view["pos"], lookat=view["lookat"], fov=45, GUI=False) \
        if args.record else None
    scene.build()
    dofs = [[e.get_joint(n).dofs_idx_local[0] for n in names] for e, names, _ in hands]
    print(f"{meta['name']}: {T} frames at {hz:g} Hz, hands {sides}, objects ({args.objects}) "
          f"{[o['id'] for o in clip['scene_objects']]}")

    writer = None
    if cam is not None:
        import imageio_ffmpeg

        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(args.record, (960, 720), fps=hz, codec="libx264", quality=8)
        writer.send(None)
    passes = 0
    while args.loops == 0 or passes < args.loops:
        for t in range(T):
            tic = time.time()
            for (e, _, q), d in zip(hands, dofs):
                e.set_dofs_position(q[t], d, zero_velocity=True)
            for e, pos, quat in objects:
                if np.isfinite(pos[t]).all():
                    e.set_pos(pos[t], zero_velocity=True)
                    e.set_quat(quat[t], zero_velocity=True)
            scene.visualizer.update(force=True)
            if writer is not None:
                writer.send(np.ascontiguousarray(cam.render(rgb=True, force_render=True)[0]))
            if not args.no_viewer:
                time.sleep(max(0.0, 1.0 / (hz * args.speed) - (time.time() - tic)))
        passes += 1
        if writer is not None:
            writer.close()
            print(f"-> {args.record}")
            writer = None
            if args.no_viewer:
                break


if __name__ == "__main__":
    main()
