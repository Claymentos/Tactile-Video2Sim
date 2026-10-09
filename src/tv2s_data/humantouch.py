"""HumanTouch (LeRobot v2.1, `humantouch` embodiment) constants and episode I/O.

One HumanTouch episode holds, at 60 Hz and in one world frame (z up, metres):
  - the head pose and both wrist tracker poses (xyz + quaternion xyzw),
  - both hand skeletons: 25 OpenXR-style joints (wrist, then 4-5 joints per finger),
  - both tactile gloves: 460 raw cells each on a 23x20 layout (meta/humantouch/tactile_mapping.json),
  - the semantic sub-task segmentation (meta/humantouch/episodes/episode_XXXXXX.json).
There is no object pose or mesh in the dataset.

The world frame is left-handed (Unity convention, z up): read as right-handed, every left skeleton has
the chirality of a right hand (its fingers curl towards cross(index MCP, pinky MCP)) and vice versa,
while the gloves and skeletons agree on which hand is which. `extract_episode` mirrors y (positions,
and quaternions as (x, y, z, w) -> (-x, y, -z, w)), which makes the world right-handed and every hand
geometrically correct for its label.

`extract_episode` needs pandas/pyarrow (scripts/data/extract_humantouch.py runs it in an env that has
them); everything else here only needs numpy.
"""

import json
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
FPS = 60.0

# 25-joint skeleton, OpenXR order. "metacarpal" is the base of the metacarpal bone (for the thumb, the
# CMC joint), "proximal" the MCP knuckle, "intermediate" the PIP, "distal" the DIP (thumb: IP).
SKELETON_JOINTS = [
    "hand",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
]

# The 21 MediaPipe / OpenPose keypoints the Wuji retargeter expects (wrist, then CMC/MCP/IP/tip for the
# thumb and MCP/PIP/DIP/tip for the other fingers), as indices into SKELETON_JOINTS.
MEDIAPIPE_FROM_SKELETON = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# MediaPipe indices per finger: base joint (thumb CMC / finger MCP), then the next three joints.
MEDIAPIPE_FINGER_CHAINS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# Parquet columns read by extract_episode -> key in the extracted .npz
_COLUMNS = {
    "observation.human.pose.head": "head_pose",
    "observation.human.pose.wrist_left": "left_wrist_tracker",
    "observation.human.pose.wrist_right": "right_wrist_tracker",
    "observation.human.hand_skeleton.left.position": "left_skeleton",
    "observation.human.hand_skeleton.right.position": "right_skeleton",
    "observation.human.hand_skeleton.left.valid": "left_skeleton_valid",
    "observation.human.hand_skeleton.right.valid": "right_skeleton_valid",
    "observation.tactile.raw_left": "left_tactile_raw",
    "observation.tactile.raw_right": "right_tactile_raw",
    "observation.tactile.valid": "tactile_valid",
    "observation.semantic.subtask_index": "subtask_index",
}


def dataset_paths(root: str | Path) -> dict[str, Path]:
    """Standard locations inside one HumanTouch task folder (e.g. HumanTouch/X001)."""
    root = Path(root)
    return {
        "root": root,
        "data": root / "data",
        "info": root / "meta" / "info.json",
        "episodes": root / "meta" / "episodes.jsonl",
        "episode_meta": root / "meta" / "humantouch" / "episodes",
        "tactile_mapping": root / "meta" / "humantouch" / "tactile_mapping.json",
    }


def episode_parquet(root: str | Path, episode_index: int, chunks_size: int = 1000) -> Path:
    chunk = episode_index // chunks_size
    return Path(root) / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def list_episodes(root: str | Path) -> list[int]:
    with open(dataset_paths(root)["episodes"]) as f:
        return [json.loads(line)["episode_index"] for line in f if line.strip()]


def load_episode_meta(root: str | Path, episode_index: int) -> dict:
    path = dataset_paths(root)["episode_meta"] / f"episode_{episode_index:06d}.json"
    with open(path) as f:
        return json.load(f)


_POSE_KEYS = ("head_pose", "left_wrist_tracker", "right_wrist_tracker")
_POSITION_KEYS = ("left_skeleton", "right_skeleton")


def mirror_y(data: dict[str, np.ndarray]) -> None:
    """Left-handed -> right-handed world, in place: y -> -y on positions, (x,y,z,w) -> (-x,y,-z,w)."""
    for key in _POSE_KEYS:
        data[key][..., 1] *= -1
        data[key][..., 3] *= -1
        data[key][..., 5] *= -1
    for key in _POSITION_KEYS:
        data[key][..., 1] *= -1


def extract_episode(root: str | Path, episode_index: int, right_handed: bool = True) -> dict[str, np.ndarray]:
    """Read one episode's parquet into dense float32 / bool arrays (needs pandas + pyarrow).
    `right_handed` mirrors the left-handed source world (see the module docstring)."""
    import pandas as pd

    df = pd.read_parquet(episode_parquet(root, episode_index), columns=list(_COLUMNS))
    out: dict[str, np.ndarray] = {}
    for column, key in _COLUMNS.items():
        values = df[column].values
        first = values[0]
        if isinstance(first, np.ndarray) and first.dtype == object:  # nested lists (T, 25, 3)
            arr = np.stack([np.stack(v) for v in values])
        elif isinstance(first, np.ndarray):
            arr = np.stack(values)
        else:
            arr = np.asarray(values)
        out[key] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
    if right_handed:
        mirror_y(out)
    meta = load_episode_meta(root, episode_index)
    out["right_handed"] = np.array(right_handed)
    out["semantic_annotations"] = np.array(json.dumps(meta["semantic_annotations"], ensure_ascii=False))
    out["episode_index"] = np.array(episode_index)
    out["fps"] = np.array(FPS)
    return out


def mediapipe_keypoints(skeleton: np.ndarray) -> np.ndarray:
    """(T, 25, 3) HumanTouch skeleton -> (T, 21, 3) MediaPipe keypoints."""
    return skeleton[:, MEDIAPIPE_FROM_SKELETON]


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]
