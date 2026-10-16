"""EgoTouch (huggingface: zhouzhoujy/EgoTouch; TouchAnything, arXiv 2605.13083) episodes, as the pipeline reads them.

One episode is one short bimanual task (~3-30 s, 213 tasks under Home/Office/Outdoor/Retail/Workbench) in
`<root>/<scene>/<task>/<YYYYMMDD_HHMMSS_mmm>/`, every stream on one 30 Hz timeline (`frame_index`):
  chest.mp4, left.mp4, right.mp4   head/chest camera and two wrist cameras, 320x240 or 640x480, no intrinsics
  rokoko_hands.json                Rokoko gloves: 21 joints per hand (MediaPipe order), in Rokoko's world
  vive_poses.json                  Vive trackers: chest, left_wrist, right_wrist (trans + quaternion xyzw)
  pressure_grids.npz               both tactile gloves on a 21x21 hand-shaped grid, normalised to [0, 1],
                                   baseline-corrected (NaN off the glove)
  jq_pressure.json                 the raw gloves (256 8-bit cells per hand) and their IMU quaternions
  masks.npz                        both gloves' masks in chest.mp4 (track 0: left, 1: right; not in every episode)
  wilor_hands.json, hamer_hands.json   video hand estimates (camera frame, not aligned), unused
(*.json are JSON lines, one per frame.)

Frames. Both Rokoko and Vive record in left-handed worlds (a skeleton read as right-handed has the other
hand's chirality; projecting the hands into the chest camera only fits once the trackers are mirrored too).
Both are mirrored like HumanTouch's (tv2s_data.humantouch.unity_to_zup: swap y and z), which makes them
right-handed but not z-up: Rokoko's world is not gravity-aligned (the table's normal, measured from the
video, lies ~75 deg off its "up" in pick_up_bottle), so the up direction comes from the table
(assemble_scene.py puts the table plane at z = 0). The two worlds are NOT the same frame either (a rotation
of up to ~110 deg and a few cm between them, per episode), and Rokoko's wrist is consistent with its own
fingers and with the video: the world of an episode is Rokoko's, the mocap hands are the Rokoko skeletons,
and the chest camera is placed in it by scripts/reconstruction/register_camera.py: T_world_cam(t) = W @
T_vive_chest(t) @ X, W (Vive -> Rokoko) and X (camera in the chest tracker) fitted on the glove masks, with
the focal length (the dataset gives none).

Stream quality varies per episode: a Vive tracker can be absent or frozen (repeats its pose), a Rokoko hand
can stall. `load_episode` flags those frames (`<side>_skeleton_valid`, `<name>_valid`); `quality` sums
them up. Without a chest track the camera is taken as static.

Glove. Of the 217 grid cells of a hand, 151 are taxels and 15 are bend sensors drawn as strips (one
sensor per column, repeated over the strip's rows: the 4 x 3 x 5 finger strips and the thumb's 3 x 2;
`glove_layout`). The layout follows the left glove seen from the palm (the right one is its mirror image):
    rows  2-5    fingertip pads, 3 columns per finger: pinky 1-3, ring 5-7, middle 9-11, index 13-15
    rows  7-11   bend strips of the same fingers
    rows  6-9    thumb pad (columns 17-19), rows 11-12 its bend strip
    rows 13-17   palm (row 17 at the wrist, columns 1 -> 19 from the ulnar to the radial side)
Contact: a taxel above CONTACT_THRESHOLD (TouchAnything's contact-region threshold).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
FPS = 30.0
REPO_ID = "zhouzhoujy/EgoTouch"
GRID = (21, 21)
CONTACT_THRESHOLD = 0.1
NOMINAL_HFOV_DEG = 50.0      # no intrinsics in the dataset: a start for register_camera.py's focal fit
TRACKERS = ("chest", "left_wrist", "right_wrist")
FILES = ("chest.mp4", "left.mp4", "right.mp4", "rokoko_hands.json", "vive_poses.json", "pressure_grids.npz",
         "jq_pressure.json", "masks.npz", "manual_contact_annotation.json")
LAYOUT_JSON = Path(__file__).resolve().parents[2] / "assets" / "tactile" / "egotouch_glove.json"

# the left glove's grid (module docstring); the right glove's column c is the left's 20 - c
FINGER_COLUMNS = {"pinky": (1, 3), "ring": (5, 7), "middle": (9, 11), "index": (13, 15), "thumb": (17, 19)}
PAD_ROWS = {"finger": (2, 5), "thumb": (6, 9)}
BEND_ROWS = {"finger": (7, 11), "thumb": (11, 12)}
PALM_ROWS = (13, 17)
PAD_U = (0.72, 0.98)          # the pads cover the distal phalanx: u (base -> tip of the finger) in this range


@dataclass
class Episode:
    scene: str
    task: str
    stamp: str                # recording time, YYYYMMDD_HHMMSS_mmm
    dir: Path

    @property
    def key(self) -> str:     # perception / scene files
        return f"{self.task}__{self.stamp}"

    @property
    def prefix(self) -> str:  # clip names: no '_' (the part before the first '_' keys the train / eval split)
        return self.stamp.replace("_", "")

    @property
    def video(self) -> Path:
        return self.dir / "chest.mp4"

    @property
    def relative(self) -> str:
        return f"{self.scene}/{self.task}/{self.stamp}"


def list_episodes(root: str | Path) -> list[Episode]:
    """Every downloaded episode under <root>, by scene, task and time."""
    root = Path(root).expanduser()
    return [Episode(p.parent.parent.parent.name, p.parent.parent.name, p.parent.name, p.parent)
            for p in sorted(root.glob("*/*/*/rokoko_hands.json"))]


def select_episodes(root: str | Path, spec: str | None = None, max_episodes: int | None = None) -> list[Episode]:
    """Episodes matching one of the comma-separated `spec` items (a key, a recording time or its prefix, a
    task name, a scene/task path), else all; the first `max_episodes`."""
    eps = list_episodes(root)
    if spec:
        items = [s for s in spec.split(",") if s]
        eps = [e for e in eps if any(i in (e.key, e.task, f"{e.scene}/{e.task}") or e.stamp.startswith(i)
                                     or e.key.startswith(i) or e.relative.startswith(i) for i in items)]
    return eps[:max_episodes] if max_episodes else eps


def episode_from_key(root: str | Path, key: str) -> Episode:
    eps = [e for e in list_episodes(root) if e.key == key]
    if len(eps) != 1:
        raise SystemExit(f"{len(eps)} EgoTouch episodes with key {key!r} under {root}")
    return eps[0]


def context(ep: Episode) -> str:
    """What the dataset says about the episode (helps Gemini name the objects)."""
    return f"Task: {ep.task.replace('_', ' ')} (scene category: {ep.scene}). Head-mounted camera looking down at the hands."


def read_jsonl(path: Path) -> list[dict]:
    text = Path(path).read_text()
    try:
        out = json.loads(text)
        return out if isinstance(out, list) else [out]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# ================================================================================ frames
def unity_points(p: np.ndarray) -> np.ndarray:
    """Unity's left-handed y-up positions -> right-handed z-up (swap y and z)."""
    return np.asarray(p)[..., [0, 2, 1]]


def unity_quats(q: np.ndarray) -> np.ndarray:
    """Quaternions xyzw, conjugated by the same mirror: (x, y, z, w) -> (-x, -z, -y, w)."""
    q = np.asarray(q)
    return np.stack([-q[..., 0], -q[..., 2], -q[..., 1], q[..., 3]], -1)


def _stalled(a: np.ndarray) -> np.ndarray:
    """(T,) True where a stream repeats its previous sample exactly (it stopped updating)."""
    flat = a.reshape(len(a), -1)
    out = np.zeros(len(a), bool)
    out[1:] = (flat[1:] == flat[:-1]).all(1)
    return out


def load_episode(ep: Episode) -> dict[str, np.ndarray]:
    """Every stream of an episode on its 30 Hz timeline, in the z-up world (module docstring):
    <side>_skeleton (T, 21, 3) MediaPipe joints, <side>_skeleton_valid (T,); <tracker>_pose (T, 7) xyz + quat
    xyzw (NaN when absent), <tracker>_valid (T,); <side>_tactile (T, 441) the 21x21 grid flattened (NaN off
    the glove), <side>_raw (T, 256), <side>_imu (T, 4); time_s (T,)."""
    rok = read_jsonl(ep.dir / "rokoko_hands.json")
    vive = read_jsonl(ep.dir / "vive_poses.json")
    T = len(rok)
    out: dict[str, np.ndarray] = {"time_s": np.array([r["ts"] for r in rok], np.float64) - rok[0]["ts"]}
    for s in SIDES:
        kp = np.full((T, 21, 3), np.nan, np.float32)
        for t, r in enumerate(rok):
            if len(r[f"{s}_pos"]) == 21:
                kp[t] = r[f"{s}_pos"]
        valid = np.isfinite(kp).all((1, 2)) & ~_stalled(np.nan_to_num(kp))
        out[f"{s}_skeleton"] = unity_points(kp).astype(np.float32)
        out[f"{s}_skeleton_valid"] = valid
    for name in TRACKERS:
        pose = np.full((T, 7), np.nan)
        for t, v in enumerate(vive[:T]):
            p = v.get("poses", {}).get(name)
            if p:
                pose[t] = list(p["trans"]) + list(p["rot"])
        valid = np.isfinite(pose).all(1) & ~_stalled(np.nan_to_num(pose))
        pose = np.concatenate([unity_points(pose[:, :3]), unity_quats(pose[:, 3:])], 1)
        out[f"{name}_pose"], out[f"{name}_valid"] = pose.astype(np.float32), valid
    grids = np.load(ep.dir / "pressure_grids.npz")
    jq = read_jsonl(ep.dir / "jq_pressure.json")
    for s in SIDES:
        g = grids[f"{s}_pressure_grid"][:T]
        out[f"{s}_tactile"] = np.concatenate([g, np.repeat(g[-1:], T - len(g), 0)]).reshape(T, -1).astype(np.float32)
        out[f"{s}_raw"] = np.array([j[f"sensor_{s}"] for j in jq[:T]], np.float32)
        out[f"{s}_imu"] = np.array([j[f"quat_{s}"] for j in jq[:T]], np.float32)
    return out


def quality(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of valid frames of every stream (an episode is usable with both hands' skeletons valid; the
    chest track places a moving camera)."""
    return {k[:-6]: round(float(v.mean()), 3) for k, v in data.items() if k.endswith("_valid")}


def video_size(ep: Episode) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(ep.video))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size


def glove_masks(ep: Episode) -> np.ndarray | None:
    """(2, T, H, W) bool masks of the left and right glove in chest.mp4, None if the episode has none."""
    if not (ep.dir / "masks.npz").exists():
        return None
    z = np.load(ep.dir / "masks.npz")
    m = z["masks"].astype(bool)
    if "obj_ids" not in z.files:
        return np.moveaxis(m[:, :2], 1, 0)
    ids = z["obj_ids"]                                     # per frame and slot; a slot keeps its glove
    slot = [int(np.bincount(np.clip(ids[:, j], 0, None)).argmax()) if (ids[:, j] >= 0).any() else -1 for j in range(ids.shape[1])]
    return np.stack([m[:, slot.index(k)] if k in slot else np.zeros_like(m[:, 0]) for k in range(2)])


# ================================================================================ glove
def glove_cells(side: str) -> tuple[list[list], list[tuple[str, int]]]:
    """The taxels ([channel, row, col, region, u, v], channel = row * 21 + col on the grid) and the bend sensors
    ((finger, channel)) of one glove. u: proximal -> distal, v: ulnar -> radial (tv2s_data/glove.py)."""
    to_grid = (lambda c: c) if side == "left" else (lambda c: 20 - c)      # canonical (left) column -> grid
    cells, bend = [], []
    for finger, (c0, c1) in FINGER_COLUMNS.items():
        kind = "thumb" if finger == "thumb" else "finger"
        r0, r1 = PAD_ROWS[kind]
        for r in range(r0, r1 + 1):
            u = PAD_U[0] + (r1 - r + 0.5) / (r1 - r0 + 1) * (PAD_U[1] - PAD_U[0])   # the top row is the tip
            for c in range(c0, c1 + 1):
                col = to_grid(c)
                cells.append([r * GRID[1] + col, r, col, finger, round(u, 4), round((c - c0 + 0.5) / (c1 - c0 + 1), 4)])
        b0 = BEND_ROWS[kind][0]
        bend += [(finger, b0 * GRID[1] + to_grid(c)) for c in range(c0, c1 + 1)]
    r0, r1 = PALM_ROWS
    for r in range(r0, r1 + 1):
        for c in range(1, 20):
            if r == r0 and c > 15:                                           # the palm's first row stops at 15
                continue
            col = to_grid(c)
            cells.append([r * GRID[1] + col, r, col, "palm", round((r1 - r + 0.5) / (r1 - r0 + 1), 4), round((c - 0.5) / 19, 4)])
    return cells, bend


def write_layout(path: Path = LAYOUT_JSON) -> Path:
    """The glove layout in tv2s_data.glove's cell-list format (load_glove_layout reads it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = dict(dataset="egotouch", grid_shape=list(GRID), n_raw=GRID[0] * GRID[1],
                cells={s: glove_cells(s)[0] for s in SIDES}, bend={s: glove_cells(s)[1] for s in SIDES})
    path.write_text(json.dumps(spec))
    return path


def check_layout(tactile: np.ndarray, side: str) -> None:
    """The layout's cells are exactly the glove's (non-NaN) grid cells, the bend strips repeat one value."""
    cells, bend = glove_cells(side)
    on = np.isfinite(tactile[0])
    ch = {c[0] for c in cells}
    strips = {r * GRID[1] + c for f, (c0, c1) in FINGER_COLUMNS.items()
              for r in range(BEND_ROWS["thumb" if f == "thumb" else "finger"][0], BEND_ROWS["thumb" if f == "thumb" else "finger"][1] + 1)
              for c in ([20 - x for x in range(c0, c1 + 1)] if side == "right" else range(c0, c1 + 1))}
    if set(np.flatnonzero(on)) != ch | strips:
        raise ValueError(f"EgoTouch {side} glove: the grid's cells differ from the layout's")
