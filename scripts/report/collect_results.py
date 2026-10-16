#!/usr/bin/env python3
"""Gather every result of the reconstruction pipeline into outputs/report/ for the results page.

Per episode (HumanTouch episode 2, TouchScale 29029fe2): the perception check videos (SAM3 masks, camera
registration, Benchmark-Genesis' CoTracker overlays), a Gemini clip cut, the HaWoR vs mocap comparison, the
reference replay and the policy replays (scripts/policy/visualize_episode.py), re-encoded to H.264 at
--width px (the page plays them inline); plus, in data.json, the scene (objects, sizes, masses), the
training curves (progress.csv) and the evaluations of the episode-mode runs.

    python scripts/report/collect_results.py
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EPISODES = {
    "humantouch": dict(name="HumanTouch, épisode 2", perception="data/humantouch_gemini/perception/episode_000002",
                       cuts="data/humantouch_gemini/cuts/videos", clip="data/episodes/ht_ep2/ep000002_episode.pkl",
                       run="ht_ep2_episode", reference="ep000002_episode_reference"),
    "touchscale": dict(name="TouchScale, 29029fe2", perception="data/touchscale/perception/29029fe2-8f70-47f6-b53a-f8acb9691df7",
                       cuts="data/touchscale/cuts/videos", clip="data/episodes/ts_29029/29029fe2_episode.pkl",
                       run="ts_29029_episode", reference="29029fe2_episode_reference"),
}


def encode(src: Path, dst: Path, width: int) -> bool:
    if not src.exists():
        return False
    import imageio_ffmpeg
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return True
    done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf",
                           f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", "-an", str(dst)]).returncode == 0
    if not done:  # still being written (visualize_episode.py running): left for the next collection
        print(f"skipped {src.name} (unreadable)")
        dst.unlink(missing_ok=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "outputs" / "report"))
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    data = {}
    for key, e in EPISODES.items():
        P = REPO / e["perception"]
        media = []

        def add(src: Path, title: str, note: str, kind: str = "video", width: int = args.width):
            name = f"{key}_{src.parent.name}_{src.stem}".replace(" ", "_") + (".mp4" if kind == "video" else src.suffix)
            ok = encode(src, out / "media" / name, width) if kind == "video" else (src.exists() and shutil.copy(src, out / "media" / name))
            if ok:
                media.append(dict(file=f"media/{name}", title=title, note=note, kind=kind))

        add(P / "sam3" / "overlay.mp4", "Masques SAM3", "Chaque objet de l'inventaire Gemini, prompté par sa description (ou des clics), propagé dans la vidéo.")
        add(P / "register_overlay.mp4", "Recalage de la caméra", "Articulations mocap avec la calibration du dataset (gris) et après recalage (couleur), sur les masques SAM3 des mains.")
        for f in sorted((P / "cotracker").glob("*_overlay.mp4")):
            add(f, f"CoTracker: {f.stem.replace('_overlay', '').replace('_', ' ')}", "track_points_cotracker.py de Benchmark-Genesis : points suivis puis remontés en 3D avec la profondeur (couleur = profondeur).")
        cuts = sorted((REPO / e["cuts"]).glob("*.mp4"))
        if cuts:
            add(cuts[0], "Un clip Gemini avec le gant", "Caméra tête, légende Gemini, objet par main et chronologie des saisies du gant.")
        add(P / "hawor" / "compare_mocap.png", "HaWoR vs mocap", "Erreur du poignet et PA-MPJPE au cours du temps.", kind="image")
        add(REPO / "outputs" / "episodes" / f"{e['reference']}.mp4", "Replay de référence (sans politique)",
            "La cible du RL : mains et objets posés sur la reconstruction, image par image. Réel | simulation vue de la caméra enregistrée | vue de la table.")
        for f in sorted((REPO / "outputs" / "episodes").glob(f"{e['run']}_*.mp4")):
            add(f, f"Replay de la politique ({f.stem.split('_')[-1]})", "La politique entraînée joue toute la démonstration depuis l'image 0 ; les fantômes orange / vert sont la référence.")

        scene = json.loads((P / "sim_scene.json").read_text()) if (P / "sim_scene.json").exists() else {}
        objects = json.loads((P / "objects.json").read_text()) if (P / "objects.json").exists() else {}
        hawor = json.loads((P / "hawor" / "compare_mocap.json").read_text()) if (P / "hawor" / "compare_mocap.json").exists() else None
        runs = sorted((REPO / "logs" / e["run"]).glob("*/progress.csv"))
        curve = {}  # a resumed run (--resume_checkpoint) continues the curve: later runs win on overlapping iterations
        for run in runs:
            for r in csv.DictReader(open(run)):
                g = lambda k: round(float(r[k]), 4) if r.get(k) not in (None, "") else None
                it = int(float(r["iteration"]))
                curve[it] = dict(it=it, reward=g("rollout/mean_reward"), length=g("rollout/mean_length"),
                                 tac_f1=g("metrics/tac_f1"), e_obj=g("metrics/e_obj"), e_wrist=g("metrics/e_wrist"), e_tip=g("metrics/e_tip"))
        curve = [curve[k] for k in sorted(curve)]
        evals = []
        if runs:
            for f in sorted((runs[-1].parent / "eval").glob("checkpoint_*.json")):
                s = json.loads(f.read_text()).get("summary", {})
                evals.append(dict(checkpoint=f.stem, **{k: s.get(k) for k in ("completion", "tac_f1", "e_wrist", "e_obj", "success")}))
        replays = [json.loads(f.read_text()) for f in sorted((REPO / "outputs" / "episodes").glob(f"{e['run']}_*.json"))]
        data[key] = dict(name=e["name"], media=media, hawor=hawor, curve=curve, evals=evals, replays=replays,
                         objects=[dict(id=o["id"], fixed=o["fixed"], mass_g=round(o["mass"] * 1000), material=o["material"],
                                       extent_cm=objects.get(o["id"], {}).get("extent_cm"), gemini_cm=objects.get(o["id"], {}).get("gemini_size_cm"))
                                  for o in scene.get("objects", [])])
        print(f"{key}: {len(media)} media, {len(curve)} training iterations, {len(evals)} evaluations")
    (out / "data.json").write_text(json.dumps(data, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
