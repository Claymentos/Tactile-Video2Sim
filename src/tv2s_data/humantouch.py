"""HumanTouch (LeRobot v2.1, `humantouch` embodiment) constants and episode I/O.

One HumanTouch episode holds, at 60 Hz and in one world frame (metres):
  - the head pose and both wrist tracker poses (xyz + quaternion xyzw),
  - both hand skeletons: 25 OpenXR-style joints (wrist, then 4-5 joints per finger),
  - both tactile gloves: 460 raw cells each on a 23x20 layout (meta/humantouch/tactile_mapping.json),
  - the semantic sub-task segmentation (meta/humantouch/episodes/episode_XXXXXX.json).
There is no object pose or mesh in the dataset.

The world frame is Unity's: left-handed, y up. Read as a right-handed frame, every skeleton has the
chirality of the other hand (its fingers curl the wrong way about cross(index MCP, pinky MCP)) and gravity
is not along -z: the table plane, measured from the head-camera depth and carried into the world with the
recorded camera calibration, has its normal along +y (0.97). `extract_episode` converts the world to a
right-handed, z-up frame by swapping y and z (a mirror, which also fixes the handedness): positions
(x, y, z) -> (x, z, y), quaternions (x, y, z, w) -> (-x, -z, -y, w). Then every hand has its label's
chirality, the table is horizontal, and the mocap hands, projected with the camera calibration
(`camera_from_head`), fall inside the gloves' SAM3 masks once its offset is refined
(scripts/reconstruction/register_camera.py).
(An earlier fix mirrored each skeleton in its own wrist frame: it gave the right chirality but left the
world mirrored and tilted.)

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


UNITY_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])          # (x, y, z) -> (x, z, y), det -1


def unity_to_zup(data: dict[str, np.ndarray]) -> None:
    """Unity's left-handed y-up world -> right-handed z-up, in place: positions swap y and z, rotations
    are conjugated by that mirror (quaternion (x, y, z, w) -> (-x, -z, -y, w))."""
    for key in ("head_pose", "left_wrist_tracker", "right_wrist_tracker"):
        p = data[key]
        data[key] = np.concatenate([p[:, [0, 2, 1]], np.stack([-p[:, 3], -p[:, 5], -p[:, 4], p[:, 6]], -1)], -1)
    for side in SIDES:
        data[f"{side}_skeleton"] = data[f"{side}_skeleton"][..., [0, 2, 1]]


def camera_from_head(extrinsics: dict) -> np.ndarray:
    """(4, 4) OpenCV camera pose in the head tracker's frame, in the z-up world (unity_to_zup), from the
    dataset's calibration_extrinsics.json (Euler xyz in degrees, translation in metres).

    Found by projecting the mocap hands into the frames (register_camera.py): read with Unity's world taken
    as right-handed, the calibration (Euler xyz, no flip) is right once the world is mirrored in x; the z-up
    world is that mirrored world rotated by Q = S diag(-1, 1, 1) (proper), hence [Q rot | Q t]. It still
    leaves the hands ~10 cm off the gloves, refined per episode by register_camera.py."""
    from scipy.spatial.transform import Rotation

    Q = UNITY_TO_ZUP @ np.diag([-1.0, 1.0, 1.0])
    m = np.eye(4)
    m[:3, :3] = Q @ Rotation.from_euler("xyz", extrinsics["rotation_rpy_deg"], degrees=True).as_matrix()
    m[:3, 3] = Q @ np.asarray(extrinsics["translation_m"])
    return m


def extract_episode(root: str | Path, episode_index: int, zup: bool = True) -> dict[str, np.ndarray]:
    """Read one episode's parquet into dense float32 / bool arrays (needs pandas + pyarrow).
    `zup` converts Unity's world to a right-handed z-up frame (see the module docstring)."""
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
    if zup:
        unity_to_zup(out)
    meta = load_episode_meta(root, episode_index)
    out["world"] = np.array("zup" if zup else "unity")
    out["semantic_annotations"] = np.array(json.dumps(meta["semantic_annotations"], ensure_ascii=False))
    out["episode_index"] = np.array(episode_index)
    out["fps"] = np.array(FPS)
    return out
