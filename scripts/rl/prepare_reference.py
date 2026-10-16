#!/usr/bin/env python3
"""Per-episode RL reference (reference.npz, tv2s_data/reference.py) for the rl_recon environment (rl_recon_env.py).

Which hands: the scene's `hands_used` (scripts/reconstruction/semantic_parsing.py, --scene), else --side / --sides.
One hand used, a one-hand reference, and the environment simulates that hand only.

Two sources:

  --bg-recon DIR   a Benchmark-Genesis reconstruction of the video (its run_touchanything_pipeline.sh output: the
                   mocap hands in the reference camera, the RecGen mesh, the CoTracker + MoGe object trajectory, the
                   table and the camera). Benchmark-Genesis' own reference chain is run unchanged
                   (run_rl_reconstruction.sh steps 1-3b), so the reference is the one its RL trained on:
                     1  hawor_to_wuji_qpos.py   the GT hand retargeted onto the Wuji Hand       [PY_RETARGET]
                     2  map_tactile_to_wuji.py  the glove's taxels on the Wuji (once per side)    [PY_SIM]
                     3  prepare_reference.py    the simulated world: table at z = 0, hand lifted clear of it
                     4  mask_axes.py            the object's long axis per frame from its SAM3 mask
                     5  tactile_grasp_pose.py   the in-palm object pose explaining the glove (in-hand starts, assist)
                     6  select_grasp_pose.py    the candidate pose the hand holds under physics
                   Benchmark-Genesis' chain is one-handed: a scene using both hands is refused (use --clip).
  --clip PKL       an episode clip of this pipeline (scripts/data/build_episode_clip.py): its retargeted hands, its
                   glove, every reconstructed object and the holds the glove gives (tv2s_data.reference.clip_reference).

Writes <out>/reference.npz (and, with --bg-recon, Benchmark-Genesis' by-products next to it).

    python scripts/rl/prepare_reference.py --bg-recon ~/Benchmark-Genesis/reconstruction/human_videos/touchanything/pick_up_screwdriver_103227 \\
        --episode ~/EgoTouch/Workbench/pick_up_screwdriver/20260321_103227_213 --scene data/references/screwdriver_103227/scene.json \\
        --out data/references/screwdriver_103227
    python scripts/rl/prepare_reference.py --clip data/episodes/et_screwdriver/20260321103227213_episode.pkl \\
        --scene data/egotouch/scene/pick_up_screwdriver__20260321_103227_213.json --out data/references/et_screwdriver
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tv2s_data.reference import clip_reference, load_reference  # noqa: E402

CONDA = Path(os.environ.get("CONDA_BASE", Path.home() / "miniconda3"))
PY_SIM = os.environ.get("PY_SIM", str(CONDA / "envs" / "genesis" / "bin" / "python"))
PY_RETARGET = os.environ.get("PY_RETARGET", str(CONDA / "envs" / "wuji_retargeting" / "bin" / "python"))
BG_ROOT = Path(os.environ.get("BG_RECON_ROOT", Path.home() / "Benchmark-Genesis" / "reconstruction"))


def hands_from_scene(scene: Path | None, default: list[str] | None) -> list[str]:
    """The hands the demonstration uses: the scene's hands_used, else `default`."""
    if scene is not None and scene.exists():
        s = json.loads(scene.read_text())
        if "hands_used" in s:
            used = s["hands_used"]
            print(f"scene {scene}: hands used: {used} ({s.get('hands_reason', '')})")
            return ["left", "right"] if used == "both" else [used]
        print(f"scene {scene} has no hands_used (semantic_parsing.py --hands-only adds it)")
    if not default:
        raise SystemExit("which hands? --scene with hands_used, or --side / --sides")
    return default


def glove_side(episode: Path) -> str:
    """The hand whose glove reports the most contact (TouchAnything grid, > 0.1)."""
    z = np.load(episode / "pressure_grids.npz")
    n = {s: float((np.nan_to_num(z[f"{s}_pressure_grid"]) > 0.1).sum()) for s in ("left", "right")}
    return max(n, key=n.get)


def run(cmd: list[str], cwd: Path) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ, RECON_ROOT=str(BG_ROOT), SCRIPTS_DIR=str(BG_ROOT / "scripts"),
               WUJI_RETARGETING_DIR=os.environ.get("WUJI_RETARGETING_DIR",
                                                   str(Path.home() / "wuji-ego-mint/eval/simulate/wuji-retargeting")))
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def bg_reference(args, sides: list[str]) -> Path:
    if len(sides) != 1:
        raise SystemExit(f"Benchmark-Genesis' reference chain simulates one hand; this scene uses {sides} "
                         "(build the episode with this pipeline and use --clip)")
    side, recon, out = sides[0], Path(args.bg_recon).resolve(), Path(args.out).resolve()
    objs = [p.name[: -len("_traj_cotracker.npz")] for p in sorted(recon.glob("*_traj_cotracker.npz"))]
    objs = [o for o in objs if (recon / "recgen_out_moge" / o / "posed_mesh.obj").exists()]
    if not args.object and len(objs) != 1:
        raise SystemExit(f"--object: one of {objs}")
    obj = args.object or objs[0]
    episode = Path(args.episode).expanduser().resolve() if args.episode else recon / "source"
    for f in ("mocap_joints_cotracker.npz", f"{obj}_traj_cotracker.npz", f"recgen_out_moge/{obj}/posed_mesh.obj",
              "camera_traj_cotracker.npz", "ground_plane_moge.json", "intrinsics.yaml"):
        if not (recon / f).exists():
            raise SystemExit(f"missing {recon / f}: run Benchmark-Genesis' run_touchanything_pipeline.sh on the video first")
    if not (episode / "pressure_grids.npz").exists():
        raise SystemExit(f"no pressure_grids.npz in {episode} (--episode: the TouchAnything / EgoTouch episode folder)")
    out.mkdir(parents=True, exist_ok=True)
    rl = BG_ROOT / "rl_recon"
    wuji_ref, taxels, reference = out / f"wuji_ref_{side}.npz", rl / "assets" / f"wuji_taxels_{side}", out / "reference.npz"
    print(f"=== Benchmark-Genesis reference: {recon.name}, object {obj}, {side} hand -> {out}")
    if args.overwrite or not wuji_ref.exists():
        print("1/6 retargeting the GT hand onto the Wuji Hand")
        run([PY_RETARGET, "hawor_to_wuji_qpos.py", "--hand-meshes", recon / "mocap_joints_cotracker.npz", "--side", side,
             "--out", wuji_ref, "--fps", "30"], BG_ROOT / "scripts")
    if not Path(f"{taxels}.npz").exists():
        print("2/6 glove taxels MANO -> Wuji")
        run([PY_SIM, "map_tactile_to_wuji.py", "--side", side, "--out", taxels], rl)
    print("3/6 reference (world frame)")
    run([PY_SIM, "prepare_reference.py", "--recon-dir", recon, "--episode", episode, "--object", obj, "--side", side,
         "--wuji-ref", wuji_ref, "--taxels", f"{taxels}.npz", "--out-dir", out, *args.prep_args.split()], rl)
    print("4/6 object axis per frame from the SAM3 masks")
    run([PY_SIM, "mask_axes.py", "--reference", reference], rl)
    print("5/6 in-hand object pose that best explains the glove")
    run([PY_SIM, "tactile_grasp_pose.py", "--reference", reference], rl)
    print("6/6 which candidate grasp poses the hand holds")
    run([PY_SIM, "select_grasp_pose.py", "--reference", reference], rl)
    return reference


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bg-recon", help="Benchmark-Genesis reconstruction folder of the video")
    src.add_argument("--clip", help="episode clip (.pkl) of this pipeline")
    ap.add_argument("--out", required=True, help="folder of reference.npz")
    ap.add_argument("--scene", default=None, help="scene JSON with hands_used (semantic_parsing.py)")
    ap.add_argument("--side", choices=("left", "right"), default=None, help="without hands_used: the one hand")
    ap.add_argument("--sides", default=None, help="without hands_used: left,right")
    ap.add_argument("--episode", default=None, help="--bg-recon: the TouchAnything / EgoTouch episode (pressure_grids.npz); "
                                                    "default <recon>/source")
    ap.add_argument("--object", default=None, help="--bg-recon: object id (default: the one with a trajectory and a mesh)")
    ap.add_argument("--prep-args", default="", help="--bg-recon: extra Benchmark-Genesis prepare_reference.py options")
    ap.add_argument("--start-clearance", type=float, default=0.005, help="--clip: starts need the reference taxels this "
                                                                         "far from resting objects (m)")
    ap.add_argument("--hand-table-clearance", type=float, default=0.008, help="--clip: lift each hand so its lowest GT "
                    "joint clears the table by this much (m; Benchmark-Genesis' prepare_reference.py); < 0: off")
    ap.add_argument("--overwrite", action="store_true", help="--bg-recon: redo the retargeting")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    default = [args.side] if args.side else args.sides.split(",") if args.sides else None
    scene = Path(args.scene) if args.scene else None

    if args.bg_recon:
        if default is None and not (scene and scene.exists()):
            ep = Path(args.episode).expanduser() if args.episode else Path(args.bg_recon) / "source"
            default = [glove_side(ep)]
            print(f"no scene: the hand whose glove fires: {default[0]}")
        path = bg_reference(args, hands_from_scene(scene, default))
    else:
        clip = pickle.load(open(args.clip, "rb"))
        if "scene_objects" not in clip:
            raise SystemExit(f"{args.clip} is not an episode clip (scripts/data/build_episode_clip.py)")
        if default is None and not (scene and scene.exists()):
            sched = {s: (np.asarray(clip["hands"][s]["object_schedule"]) >= 0).any() for s in ("left", "right")}
            default = [s for s in ("left", "right") if sched[s]]
            print(f"no scene: the hands with an object on their schedule: {default}")
        sides = hands_from_scene(scene, default)
        ref = clip_reference(clip, sides, start_clearance=args.start_clearance, hand_table_clearance=args.hand_table_clearance)
        path = out / "reference.npz"
        np.savez(path, **ref)
    r = load_reference(path)
    T = len(r["qpos"][0])
    print(f"-> {path}: {T} frames at {r['fps']:g} Hz, hands {r['sides']}, objects {list(r['object_ids'])}, "
          f"{len(r['holds'])} holds {[(r['sides'][h], int(a), int(b)) for h, _, a, b in r['holds']]}, "
          f"{len(r['rsi_frames'])} in-hand start candidates, glove channels {r['real_contact'].shape[-1]}")
    (out / "reference_source.json").write_text(json.dumps(dict(
        source="benchmark_genesis" if args.bg_recon else "clip", input=str(Path(args.bg_recon or args.clip).resolve()),
        scene=str(scene.resolve()) if scene else None, sides=r["sides"]), indent=1))


if __name__ == "__main__":
    main()
