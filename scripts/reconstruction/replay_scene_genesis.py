#!/usr/bin/env python3
"""Kinematic replay of an assembled scene in Genesis, rendered from the real camera next to the video.

Loads sim_scene.json / sim_trajectories.npz (assemble_scene.py): the table as the ground plane, every
object's RecGen mesh, placed on its trajectory at each sampled frame (no physics: a check of the
reconstruction, not a simulation). Renders each frame from the recorded camera (T_sim_cam(t) =
T_sim_world T_world_cam(t), OpenCV -> OpenGL axes, the real vertical field of view) and from a fixed view
of the table, and writes real | simulated (camera) | simulated (table view) side by side.

Writes <perception>/replay.mp4.
    python scripts/reconstruction/replay_scene_genesis.py --perception data/touchscale/perception/<episode_id>
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])               # camera axes: OpenCV (y down, z forward) -> OpenGL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception", required=True, help="<data>/perception/<key>")
    ap.add_argument("--height", type=int, default=360, help="height of each panel (px)")
    args = ap.parse_args()
    root = Path(args.perception)
    meta = json.loads((root / "frames.json").read_text())
    sc = json.loads((root / "sim_scene.json").read_text())
    traj = np.load(root / "sim_trajectories.npz")
    Twc = np.load(root / "camera.npz")["T_world_cam"]
    T_sim_cam = np.asarray(sc["T_sim_world"]) @ Twc
    W0, H0 = meta["size"]
    h, w = args.height, int(round(args.height * W0 / H0)) // 2 * 2
    fov = math.degrees(2 * math.atan(H0 / 2 / meta["intrinsics"]["fy"]))

    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.5, 0.5, 0.5)))
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.85, 0.85, 0.85)))
    ents = {o["id"]: scene.add_entity(gs.morphs.Mesh(file=o["mesh"], fixed=False, collision=False), material=gs.materials.Rigid(gravity_compensation=1.0))
            for o in sc["objects"]}
    cam = scene.add_camera(res=(w, h), fov=fov, GUI=False)
    side = scene.add_camera(res=(w, h), fov=50, GUI=False)
    scene.build()
    centre = np.mean([traj[o][0][:3, 3] for o in ents], 0)
    side.set_pose(pos=centre + np.array([0.45, -0.45, 0.45]), lookat=centre, up=(0, 0, 1))

    writer = cv2.VideoWriter(str(root / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["sampled_fps"], (3 * w, h))
    for t in range(len(meta["frame_index"])):
        for oid, e in ents.items():
            P = traj[oid][t]
            if np.isfinite(P).all():
                e.set_pos(P[:3, 3])
                e.set_quat(Rotation.from_matrix(P[:3, :3]).as_quat()[[3, 0, 1, 2]])
        if np.isfinite(T_sim_cam[t]).all():
            cam.set_pose(transform=T_sim_cam[t] @ CV_TO_GL)
        sim = cam.render(rgb=True, force_render=True)[0]
        top = side.render(rgb=True, force_render=True)[0]
        real = cv2.resize(cv2.imread(str(root / "frames" / f"{t:06d}.jpg")), (w, h))
        row = np.hstack([real, cv2.cvtColor(np.asarray(sim), cv2.COLOR_RGB2BGR), cv2.cvtColor(np.asarray(top), cv2.COLOR_RGB2BGR)])
        cv2.putText(row, f"{meta['times_s'][t]:.1f} s   real | simulation from the camera | table view", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(row)
    writer.release()
    print(f"-> {root / 'replay.mp4'}")


if __name__ == "__main__":
    main()
