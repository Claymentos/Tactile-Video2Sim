"""TouchScale (huggingface: 2077AIDataFoundation/TouchScale) episodes, as the clip pipeline reads them.

One episode is one short task (10-20 s) recorded with a head RGB-D camera (rgb_head.mp4, 1280x720 at
30 Hz, one timestamp per frame in rgb_head.csv), two wrist cameras, an IMU and two tactile gloves (15
pads per hand, normal force in N plus shear, ~100 Hz, timestamped on the same clock, in <id>.episode.h5).
Files are `data/train-XXXXX-of-00430/<episode_id>.<kind>` once the shard .tar files are extracted; <id>.json
gives the task name, the instruction, the items and the primary hand.

There is no hand pose in TouchScale (no skeleton, no wrist tracker): its episodes can be parsed, cut into
clips and watched with the glove, but not retargeted onto the Wuji hands without a video hand-pose
estimator first.

Contact: a hand touches something when at least CONTACT_MIN_TAXELS taxels read more than
CONTACT_FORCE_N. This reproduces the dataset's own `tactile_contact_frac_<side>` within 3.6 points on
average (the 64 hands of shard 0).
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
FPS = 30.0
CONTACT_FORCE_N = 0.2
CONTACT_MIN_TAXELS = 5
_EPISODE_JSON = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.json$")


@dataclass
class Episode:
    id: str
    dir: Path
    meta: dict

    @property
    def short_id(self) -> str:
        return self.id[:8]

    def path(self, kind: str) -> Path:
        return self.dir / f"{self.id}.{kind}"

    @property
    def video(self) -> Path:
        return self.path("rgb_head.mp4")


def list_episodes(root: str | Path) -> list[Episode]:
    """Every episode of the extracted shards under <root>/data, by shard then id."""
    eps = []
    for shard in sorted(Path(root).expanduser().glob("data/train-*-of-*")):
        if shard.is_dir():
            eps += [Episode(p.stem, shard, json.loads(p.read_text())) for p in sorted(shard.iterdir())
                    if _EPISODE_JSON.match(p.name)]
    return eps


def select_episodes(root: str | Path, ids: str | None = None, max_episodes: int | None = None) -> list[Episode]:
    """Episodes whose id starts with one of the comma-separated `ids`, else the first `max_episodes`."""
    eps = list_episodes(root)
    if ids:
        prefixes = [i for i in ids.split(",") if i]
        eps = [e for e in eps if any(e.id.startswith(p) for p in prefixes)]
    return eps[:max_episodes] if max_episodes else eps


def context(ep: Episode) -> str:
    """What the dataset says about the episode (helps Gemini name the objects)."""
    m = ep.meta
    return (f"Task: {m['task_name']} (scene: {m.get('scene', '?')}; primary hand: {m.get('primary_hand', '?')}). "
            f"Items: {', '.join(m.get('items', []))}.")


def pad_shapes(ep: Episode) -> dict[str, list[int]]:
    """Taxel grid of every glove pad, in the order the taxels are flattened (glove_at_frames)."""
    import h5py

    with h5py.File(ep.path("episode.h5")) as f:
        g = f["obs/tactile/raw/left"]
        return {k[4:]: list(g[k].shape[1:]) for k in _pads(g)}


def _pads(g) -> list[str]:
    return sorted((k for k in g if k.startswith("pad_")), key=lambda k: int(k[4:]))


def glove_at_frames(ep: Episode, side: str, times: np.ndarray) -> dict[str, np.ndarray]:
    """One glove at `times` (nearest glove sample): `force` (T, n_taxels) normal force and `shear_x`,
    `shear_y` (T, n_taxels), in N; taxels flattened pad by pad (pad_shapes)."""
    import h5py

    with h5py.File(ep.path("episode.h5")) as f:
        g = f[f"obs/tactile/raw/{side}"]
        t = g["t_epoch"][:]
        pads = _pads(g)
        out = {name: np.concatenate([g[f"{prefix}{k[4:]}"][:].reshape(len(t), -1) for k in pads], 1)
               for name, prefix in (("force", "pad_"), ("shear_x", "shear_x_"), ("shear_y", "shear_y_"))}
    i = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    i -= (times - t[i - 1]) < (t[i] - times)                          # nearest of the two neighbours
    return {k: v[i].astype(np.float32) for k, v in out.items()}
