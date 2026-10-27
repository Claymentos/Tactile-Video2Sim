Copied from Benchmark-Genesis (reconstruction/scripts, 2026-10-21) so that the perception chain runs from this repo
alone: CoTracker point tracking (`track_points_cotracker.py`, run in the `sam3` env), the rigid pose from the points
(`pose_from_points.py`, with `kalman_poses.py` and the pose utilities of `track_object_v2s2r.py`). Called by
`scripts/reconstruction/track_objects_cotracker.py`. The CoTracker checkout and checkpoint come from the torch hub
cache (~/.cache/torch/hub) or COTRACKER_DIR.
