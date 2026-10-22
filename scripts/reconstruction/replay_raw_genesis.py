#!/usr/bin/env python3
"""Raw replay of an RL reference (scripts/rl/prepare_reference.py) in the Genesis viewer: the retargeted Wuji hand
and the object's mesh on its trajectory, in the world the RL simulates (table at z = 0).

Kinematic (no physics, no policy, no contact): every frame, the Wuji hand is put on its retargeted pose (palm pose
+ 20 finger joints) and the object (its RecGen mesh) on its CoTracker + depth trajectory. The object is drawn
translucent green, so a hand passing through it shows. --human also draws the mocap (GT) joints the hand was
retargeted from, as red dots (the retargeting's input); --no-lift shows the hand where the reconstruction put it,
before the lift that clears the table (hand_dz); --frame N holds one frame.

    python scripts/reconstruction/replay_raw_genesis.py --reference data/references/et_hammer/reference.npz --human
    python scripts/reconstruction/replay_raw_genesis.py --reference <npz> --speed 0.5 --no-lift
    python scripts/reconstruction/replay_raw_genesis.py --reference <npz> --record outputs/raw_replay.mp4 --no-viewer
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def from_reference(args) -> dict:
    """hands [(side, R0, (T, 26) base + finger joints, (T, 21, 3) GT joints)], objects [(mesh, (T, 4, 4))]."""
    from tv2s_data.reference import load_reference
    from tv2s_env.robots.wuji_model import base_coordinates

    path = Path(args.reference)
    r, raw = load_reference(path), np.load(path)
    T = len(r["qpos"][0])
    hands = []
    for h, s in enumerate(r["sides"]):
        q = np.concatenate([r["base_q"][h], r["qpos"][h]], 1)
        human = r["human_kp"][h].copy()
        if args.no_lift and "hand_dz" in raw.files:
            dz = np.asarray(raw["hand_dz"], float).reshape(T, 1)
            pos = r["wrist_pos"][h] - np.concatenate([np.zeros((T, 2)), dz], 1)
            q[:, :6] = base_coordinates(r["base_R0"][h], pos, r["wrist_rot"][h])
            human[..., 2] -= dz
        hands.append((s, r["base_R0"][h], q, human))
    objects = []
    for k, coll in enumerate(r["object_collision"]):
        visual = Path(str(coll)).with_name("object_visual.obj")                  # Benchmark-Genesis' full-resolution mesh
        P = np.tile(np.eye(4), (T, 1, 1))
        P[:, :3, :3], P[:, :3, 3] = r["obj_rot"][k], r["obj_pos"][k]
        P[~r["obj_valid"][k].astype(bool)] = np.nan
        objects.append((str(visual if visual.exists() else coll), P))
    return dict(name=path.parent.name, hz=float(r["fps"]), T=T, hands=hands, objects=objects)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="RL reference .npz (scripts/rl/prepare_reference.py)")
    ap.add_argument("--no-lift", action="store_true", help="the hand before its table-clearance lift")
    ap.add_argument("--human", action="store_true", help="also the mocap joints the hand was retargeted from (red)")
    ap.add_argument("--frame", type=int, default=None, help="hold this frame")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--loops", type=int, default=0, help="number of passes (0: until the viewer is closed)")
    ap.add_argument("--record", default=None, help="also write an mp4 from a fixed camera")
    ap.add_argument("--no-viewer", action="store_true", help="no window (with --record)")
    args = ap.parse_args()

    import genesis as gs

    from tv2s_env.robots.wuji_model import BASE_JOINTS, WujiHand, default_urdf

    S = from_reference(args)
    T, hz = S["T"], S["hz"]
    centre = np.mean([np.nanmean(q[:, :3], 0) for _, _, q, _ in S["hands"]], 0)        # the palms' mean position
    view = dict(pos=tuple(centre + [0.45, -0.45, 0.3]), lookat=tuple(centre))
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1.0 / hz, gravity=(0.0, 0.0, 0.0)),
                     rigid_options=gs.options.RigidOptions(enable_collision=False),
                     viewer_options=gs.options.ViewerOptions(camera_pos=view["pos"], camera_lookat=view["lookat"],
                                                             max_FPS=int(hz * args.speed)),
                     show_viewer=not args.no_viewer)
    scene.add_entity(gs.morphs.Plane())
    hands, dots = [], []
    work = REPO / "outputs" / "urdf"
    work.mkdir(parents=True, exist_ok=True)
    for s, R0, q, human in S["hands"]:
        model = WujiHand(s, urdf=default_urdf(s))
        urdf = model.write_floating_urdf(str(work / f"wuji_{s}_raw_replay.urdf"), R0, base_mass=0.01)
        e = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=True, merge_fixed_links=True),
                             surface=gs.surfaces.Default(color=(1.0, 0.55, 0.1)))
        hands.append((e, BASE_JOINTS + model.dof_names, q))
        if args.human and human is not None:
            dots.append(([scene.add_entity(gs.morphs.Sphere(radius=0.004, fixed=False, collision=False),
                                           surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1)))
                          for _ in range(human.shape[1])], human))
    objects = []
    for mesh, P in S["objects"]:
        e = scene.add_entity(gs.morphs.Mesh(file=mesh, fixed=False, collision=False),
                             surface=gs.surfaces.Default(color=(0.1, 0.8, 0.2, 0.55)))
        ok = np.isfinite(P[:, :3, :3]).all((1, 2))
        quat = np.full((T, 4), np.nan)
        quat[ok] = Rotation.from_matrix(P[ok, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        objects.append((e, P[:, :3, 3], quat))
    cam = scene.add_camera(res=(960, 720), pos=view["pos"], lookat=view["lookat"], fov=45, GUI=False) \
        if args.record else None
    scene.build()
    dofs = [[e.get_joint(n).dofs_idx_local[0] for n in names] for e, names, _ in hands]
    print(f"{S['name']}: {T} frames at {hz:g} Hz, hands {[h[0] for h in S['hands']]}, {len(objects)} objects")

    writer = None
    if cam is not None:
        import imageio_ffmpeg

        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(args.record, (960, 720), fps=hz, codec="libx264", quality=8)
        writer.send(None)
    frames = [args.frame] * (T if args.record else 1) if args.frame is not None else range(T)
    passes = 0
    while args.loops == 0 or passes < args.loops:
        for t in frames:
            tic = time.time()
            for (e, _, q), d in zip(hands, dofs):
                e.set_dofs_position(q[t], d, zero_velocity=True)
            for spheres, human in dots:
                for sph, p in zip(spheres, human[t]):
                    if np.isfinite(p).all():
                        sph.set_pos(p, zero_velocity=True)
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
