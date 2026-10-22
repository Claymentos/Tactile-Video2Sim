"""ACE-Data-0 (huggingface: ACERobotics/ACE-Data-0, gated; arXiv 2607.28625), Table-scale Tactile shards, as the
pipeline reads them: one egocentric video and the two tactile gloves, nothing else.

A take is `<root>/Table-scale/Tactile/ACE-Data-Tactile-<shard>/` + its index (LeRobot v3, one file per take):
  videos/egocap/stereo0/take-<i>.mp4   the headset's forward camera, 720x720 at 20 fps, no calibration given (the
                                       other egocap views, cam0-3, are 180 deg fisheyes; stereo1 is its pair)
  raw/tactile/<side>/take-<i>.parquet  the glove, ~80 Hz: sync_time_s (the take's clock, the video's), pressure_0..255
                                       (8-bit), the glove IMU's quaternion_{w,x,y,z}, a contact flag
  raw/captions/take-<i>.txt            the task and its sub-tasks with their [start-end] times in seconds
The Tactile shards carry no hand pose, no object pose or mesh and no headset pose (the HOI subset has those, without
gloves): the hands come from the video (launch/reconstruct.sh, HANDS=hawor).

Glove. The JQ gloves of EgoTouch (256 raw cells per hand): assets/tactile/jq_glove_map.json (fit_jq_glove_map.py)
lays the raw cells on EgoTouch's 21x21 grid, so `extract_ace.py` writes an EgoTouch-shaped episode folder
(chest.mp4, pressure_grids.npz) the reconstruction and the reference chain read unchanged. Some cells sit at a
constant offset for the whole take: each cell's baseline (a low percentile over the take) is removed.

Key of a take: `t<shard>_<index>`, e.g. t4_000078.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
FPS = 20.0
VIEW = "stereo0"
REPO_ID = "ACERobotics/ACE-Data-0"
TACTILE_MAX = 255.0                      # pressure_grids.npz's normalisation: the grid times this is in raw units
BASELINE_PERCENTILE = 5.0
GLOVE_MAP = Path(__file__).resolve().parents[2] / "assets" / "tactile" / "jq_glove_map.json"


@dataclass
class Take:
    root: Path               # the dataset (ACE-Data-0)
    shard: int
    index: int

    @property
    def key(self) -> str:
        return f"t{self.shard}_{self.index:06d}"

    @property
    def shard_dir(self) -> Path:
        return self.root / "Table-scale" / "Tactile" / f"ACE-Data-Tactile-{self.shard}"

    def file(self, kind: str, ext: str) -> str:
        """Path of one of the take's files inside the shard (as on the hub)."""
        return f"{kind}/take-{self.index:06d}.{ext}"

    @property
    def video(self) -> Path:
        return self.shard_dir / self.file(f"videos/egocap/{VIEW}", "mp4")

    def tactile(self, side: str) -> Path:
        return self.shard_dir / self.file(f"raw/tactile/{side}", "parquet")

    @property
    def captions(self) -> Path:
        return self.shard_dir / self.file("raw/captions", "txt")

    def hub_files(self) -> list[str]:
        """What the pipeline downloads of a take (paths in the hub repo)."""
        rel = [self.file(f"videos/egocap/{VIEW}", "mp4"), self.file("raw/captions", "txt")]
        rel += [self.file(f"raw/tactile/{s}", "parquet") for s in SIDES]
        return [f"Table-scale/Tactile/ACE-Data-Tactile-{self.shard}/{r}" for r in rel]


def take_from_key(root: str | Path, key: str) -> Take:
    m = re.fullmatch(r"t(\d+)_(\d+)", key)
    if not m:
        raise SystemExit(f"ACE take key t<shard>_<index> expected, got {key!r}")
    return Take(Path(root).expanduser(), int(m[1]), int(m[2]))


def read_captions(take: Take) -> dict:
    """{'task': str, 'subtasks': [(start_s, end_s, text)]} (the caption can be a bare file name: no task)."""
    text = take.captions.read_text() if take.captions.exists() else ""
    task = re.search(r"Task:\s*\n(.+)", text)
    subtasks = [(float(a), float(b), t.strip()) for a, b, t in re.findall(r"\[([\d.]+)-([\d.]+)\]\s*(.+)", text)]
    return dict(task=task[1].strip() if task else "", subtasks=subtasks)


def context(take: Take) -> str:
    """What the dataset says about the take (helps the scene parser name the objects)."""
    c = read_captions(take)
    lines = [f"Task: {c['task']}"] if c["task"] else []
    lines += [f"[{a:.1f}-{b:.1f} s] {t}" for a, b, t in c["subtasks"]]
    return "\n".join(lines + ["Head-mounted camera looking down at the hands, black tactile gloves on both hands."])


def load_glove(take: Take, side: str) -> dict[str, np.ndarray]:
    """One glove's samples as recorded: time_s (N,), raw (N, 256), imu (N, 4) quaternion xyzw, contact (N,).
    Needs pandas (the `lerobot` env)."""
    import pandas as pd

    df = pd.read_parquet(take.tactile(side))
    num = lambda cols: df[cols].apply(pd.to_numeric).to_numpy(np.float64)
    order = np.argsort(num(["sync_time_s"])[:, 0], kind="stable")
    return dict(time_s=num(["sync_time_s"])[order, 0],
                raw=num([f"pressure_{i}" for i in range(256)])[order].astype(np.float32),
                imu=num([f"quaternion_{a}" for a in "xyzw"])[order].astype(np.float32),
                contact=num(["contact"])[order, 0] > 0)


def resample_max(time_s: np.ndarray, values: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    """Per video frame, the max of the samples within half a frame of it (a short press between frames is kept);
    the nearest sample where none falls inside."""
    dt = np.median(np.diff(frame_times))
    lo = np.searchsorted(time_s, frame_times - dt / 2)
    hi = np.searchsorted(time_s, frame_times + dt / 2)
    nearest = np.clip(np.searchsorted(time_s, frame_times), 1, len(time_s) - 1)
    nearest -= (frame_times - time_s[nearest - 1]) < (time_s[nearest] - frame_times)
    return np.stack([values[a:b].max(0) if b > a else values[n] for a, b, n in zip(lo, hi, nearest)])


def load_glove_map(path: Path = GLOVE_MAP) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per side, (grid channels, raw cells) of jq_glove_map.json."""
    spec = json.loads(Path(path).read_text())
    return {s: (np.array([int(c) for c in m], int), np.array(list(m.values()), int)) for s, m in spec["cells"].items()}


def glove_grid(raw: np.ndarray, side: str, glove_map: dict | None = None) -> np.ndarray:
    """(T, 21, 21) EgoTouch grid in [0, 1] (NaN off the glove; a grid cell the map lacks reads 0) from baseline-free
    raw cells (T, 256)."""
    channels, cells = (glove_map or load_glove_map())[side]
    from tv2s_data.egotouch import GRID, glove_cells

    taxels, bend = glove_cells(side)
    grid = np.full((len(raw), GRID[0] * GRID[1]), np.nan, np.float32)
    grid[:, [c[0] for c in taxels] + [b[1] for b in bend]] = 0.0
    grid[:, channels] = raw[:, cells] / TACTILE_MAX
    return grid.reshape(len(raw), *GRID)


def remove_baseline(raw: np.ndarray, percentile: float = BASELINE_PERCENTILE) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's resting level subtracted (clipped at 0); returns (corrected, baseline). A cell reading non-zero most
    of the take rests at an offset, its low percentile over the non-zero samples: such a cell drops to 0 for single
    samples (t4_000078: a cell at 10-12 for 28 s, at 0 on 6 % of the samples, scattered), and a plain percentile
    lands on those drops. A cell at 0 most of the take rests at 0."""
    base = np.zeros(raw.shape[1], np.float32)
    for k in np.flatnonzero((raw > 0).mean(0) > 0.5):
        base[k] = np.percentile(raw[raw[:, k] > 0, k], percentile)
    return np.clip(raw - base, 0, None), base
