"""Stage 1 of the clip files: cut an episode into clips with what is known without reconstructing anything.

A *cut* is a clip window of an episode's head-camera video, in video frames, with:
  - where it comes from: the Gemini scene parsing of the video (<data>/scene/<key>.json,
    scripts/reconstruction/semantic_parsing.py), else the dataset's own segmentation (HumanTouch's
    annotations, TouchScale's task segment);
  - what it shows: caption, and per hand the object manipulated, its source and destination, the objects
    present (ids into the scene's objects, with their SAM3 prompt, material, shape and size);
  - the glove at every video frame: per hand its tactile signal, its contacts, and the sustained grasp
    inside the cut;
  - a video: the head camera with the caption, the objects per hand and the glove grasp timeline.
No hand pose and no retargeting are needed, so both datasets go through it. The perception of the
episode (SAM3 masks, depth, RecGen meshes, object trajectories) and the simulation clips
(scripts/data/build_sim_clips.py: retargeted hands, objects) are built on top of the cuts, by name.

A dataset plugs in as an `EpisodeSource` (humantouch_sources, touchscale_sources).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from tv2s_data.clips import ClipConfig, grasp_interval, scene_windows, segment_episode
from tv2s_data.glove import clean_tactile, contact_from_pressure, load_glove_layout
from tv2s_data.humantouch import SIDES, dataset_paths


@dataclass
class EpisodeSource:
    dataset: str
    key: str                     # its scene file: <data>/scene/<key>.json
    prefix: str                  # cut names <prefix>_cKK (the part before '_' keys the train/eval split)
    video: Path                  # head-camera video
    fps: float                   # video frame rate: cut frames are video frames
    n_frames: int
    tactile: dict[str, dict[str, np.ndarray]]   # per hand, named arrays (n_frames, ...) at the video frames
    contact: dict[str, np.ndarray]              # per hand, (n_frames, n_channels) bool
    grasp: Callable[[str, int, int], tuple[int, int] | None]   # sustained grasp of a hand in [s, e), cut frames
    windows: list[dict]          # the dataset's segmentation: dict(start, end, types, captions), video frames
    info: dict = field(default_factory=dict)    # copied into every cut's metadata


def cut_episode(src: EpisodeSource, scene: dict | None, cfg: ClipConfig) -> list[tuple[dict, dict]]:
    """(cut, arrays) of every clip of one episode: the scene's clips if given, else the dataset's windows."""
    if scene is not None:
        windows = scene_windows(scene, src.n_frames, np.arange(src.n_frames), src.fps, src.fps, cfg)
        objects = {o["id"]: o for o in scene["objects"]}
        # containers other objects are taken from / put into (the basket): what the hands carry in bulk
        places = {c[f"{side}_hand"][k] for c in scene["clips"] for side in SIDES for k in ("from", "to")}
        bins = {i for i, o in objects.items() if o["is_container"] and i in places}
    else:
        windows, objects, bins = src.windows, {}, set()
    cuts = []
    for k, w in enumerate(windows):
        s, e = int(w["start"]), int(w["end"])
        hands = {}
        for side in SIDES:
            sem = None
            if scene is not None:
                sem = dict(w["scene"][f"{side}_hand"], object_is_bin=w["scene"][f"{side}_hand"]["object"] in bins)
                if sem["object"] in objects:
                    sem["attributes"] = objects[sem["object"]]
            grasp = src.grasp(side, s, e)
            hands[side] = dict(grasp_interval=None if grasp is None else [int(grasp[0]), int(grasp[1])], semantic=sem)
        meta = dict(name=f"{src.prefix}_c{k:02d}", dataset=src.dataset, clip=k, **src.info,
                    start=s, end=e, fps=src.fps, start_s=round(s / src.fps, 3), end_s=round(e / src.fps, 3),
                    video=str(src.video), segments="scene" if scene is not None else "dataset",
                    types=w["types"], captions=w["captions"],
                    scene=None if scene is None else dict(w["scene"], objects={
                        i: objects[i] for i in w["scene"]["objects_present"] if i in objects}))
        arrays = {f"{side}_{name}": a[s:e] for side in SIDES for name, a in src.tactile[side].items()}
        arrays.update({f"{side}_contact": src.contact[side][s:e] for side in SIDES})
        cuts.append((dict(metadata=meta, hands=hands), arrays))
    return cuts


def video_view(cut: dict) -> dict:
    """The cut as tv2s_data.clip_video draws it (frames of the video, at its rate)."""
    m = cut["metadata"]
    return dict(metadata=dict(m, source_frames=np.arange(m["start"], m["end"]), hz=m["fps"]), hands=cut["hands"])


# ================================================================================ datasets
def humantouch_sources(data: Path, dataset_root: Path, episodes: set[int] | None, cfg: ClipConfig,
                       finger_tip_row: int = 22) -> Iterator[EpisodeSource]:
    """HumanTouch episodes extracted by extract_humantouch.py (<data>/episodes): 60 Hz glove and video.
    Tactile: per hand the cleaned pressure of the 360 assigned glove cells (`pressure`, GloveLayout
    channel order) and the 460 raw cells (`raw`); grasp: tv2s_data.clips.grasp_interval."""
    from tv2s_data.clip_video import head_video_path

    mapping = dataset_paths(dataset_root)["tactile_mapping"].resolve()
    layouts = {s: load_glove_layout(mapping, s, finger_tip_row=finger_tip_row) for s in SIDES}
    for path in sorted((data / "episodes").glob("episode_*.npz")):
        e = int(path.stem.split("_")[1])
        if episodes is not None and e not in episodes:
            continue
        ep = dict(np.load(path))
        fps, n = float(ep["fps"]), len(ep["head_pose"])
        tactile, contact = {}, {}
        for s in SIDES:
            raw = ep[f"{s}_tactile_raw"].astype(np.float32)
            pressure = clean_tactile(raw, fps, window_s=cfg.baseline_window_s)[:, layouts[s].channels]
            tactile[s] = dict(pressure=pressure.astype(np.float32), raw=raw)
            contact[s] = contact_from_pressure(pressure, cfg.contact_threshold)

        def grasp(side, a, b, contact=contact, fps=fps):
            return grasp_interval(contact[side][a:b], layouts[side], fps, cfg)

        windows = segment_episode(json.loads(str(ep["semantic_annotations"])), n, np.arange(n), fps, cfg)
        yield EpisodeSource(dataset="humantouch", key=f"episode_{e:06d}", prefix=f"ep{e:06d}",
                            video=head_video_path(dataset_root, e), fps=fps, n_frames=n, tactile=tactile,
                            contact=contact, grasp=grasp, windows=windows,
                            info=dict(episode=e, tactile_mapping=str(mapping), finger_tip_row=finger_tip_row))


EGOTOUCH_GRASP_MIN_CHANNELS = 8       # of its 151 taxels (an idle hand reads <= 7 above the threshold)


def egotouch_sources(data: Path, root: Path, spec: str | None, cfg: ClipConfig,
                     max_episodes: int | None = None) -> Iterator[EpisodeSource]:
    """EgoTouch episodes extracted by extract_egotouch.py (<data>/episodes/<key>.npz): 30 Hz glove and video.
    Tactile: per hand the 151 taxels of the normalised grid (`pressure`, GloveLayout channel order, in [0, 1])
    and its 15 bend sensors (`bend`); contact above tv2s_data.egotouch.CONTACT_THRESHOLD; grasp:
    tv2s_data.clips.grasp_interval with EGOTOUCH_GRASP_MIN_CHANNELS. One window: the whole task."""
    import dataclasses

    from tv2s_data import egotouch as et

    layouts = {s: load_glove_layout(et.LAYOUT_JSON, s) for s in SIDES}
    bend = {s: [c for _, c in et.glove_cells(s)[1]] for s in SIDES}
    gcfg = dataclasses.replace(cfg, grasp_min_channels=EGOTOUCH_GRASP_MIN_CHANNELS)
    for ep in et.select_episodes(root, spec, max_episodes):
        path = data / "episodes" / f"{ep.key}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        n = len(d["time_s"])
        tactile, contact = {}, {}
        for s in SIDES:
            grid = np.nan_to_num(d[f"{s}_tactile"])
            pressure = grid[:, layouts[s].channels]
            tactile[s] = dict(pressure=pressure.astype(np.float32), bend=grid[:, bend[s]].astype(np.float32))
            contact[s] = pressure > et.CONTACT_THRESHOLD

        def grasp(side, a, b, contact=contact):
            return grasp_interval(contact[side][a:b], layouts[side], et.FPS, gcfg)

        task = ep.task.replace("_", " ")
        yield EpisodeSource(dataset="egotouch", key=ep.key, prefix=ep.prefix, video=ep.video, fps=et.FPS,
                            n_frames=n, tactile=tactile, contact=contact, grasp=grasp,
                            windows=[dict(start=0, end=n, types=[ep.task], captions=[task])],
                            info=dict(episode_key=ep.key, task_name=ep.task, scene_category=ep.scene,
                                      tactile_mapping=str(et.LAYOUT_JSON)))


def touchscale_sources(root: Path, ids: str | None, max_episodes: int | None) -> Iterator[EpisodeSource]:
    """TouchScale episodes (tv2s_data/touchscale.py): 30 Hz video, glove resampled at its frames.
    Tactile: per hand the normal force and the shear of every taxel (`force`, `shear_x`, `shear_y`, N);
    grasp: the sustained contact overlapping the cut most."""
    from tv2s_data import touchscale as ts

    for ep in ts.select_episodes(root, ids, max_episodes):
        times = ts.frame_times(ep)
        tactile = {s: ts.glove_at_frames(ep, s, times) for s in SIDES}
        contact = {s: tactile[s]["force"] > ts.CONTACT_FORCE_N for s in SIDES}
        runs = {s: ts.contact_intervals(contact[s].sum(1)) for s in SIDES}

        def grasp(side, a, b, runs=runs):
            overlap = [(min(y, b) - max(x, a), x, y) for x, y in runs[side]]
            best = max(overlap, default=(0, 0, 0))
            return (max(best[1], a) - a, min(best[2], b) - a) if best[0] > 0 else None

        s, e = ts.task_window(ep)
        task = ep.meta["task_name"]
        yield EpisodeSource(dataset="touchscale", key=ep.id, prefix=ep.short_id, video=ep.video, fps=ts.FPS,
                            n_frames=len(times), tactile=tactile, contact=contact, grasp=grasp,
                            windows=[dict(start=s, end=min(e, len(times)), types=[task], captions=[task])],
                            info=dict(episode_id=ep.id, task_name=task, primary_hand=ep.meta.get("primary_hand"),
                                      items=ep.meta.get("items", []), taxel_pads=ts.pad_shapes(ep)))
