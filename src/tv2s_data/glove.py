"""The HumanTouch tactile glove: its 23x20 cell layout and the cleaning of its raw signal.

Layout (meta/humantouch/tactile_mapping.json, `right_region` shown, the left glove is mirrored):

    rows  0-14   palm, 15 x 16 cells (right glove: columns 0-15, left glove: columns 4-19)
    rows 15-16   unassigned
    rows 17-22   one 6 x 4 patch per finger (right glove, columns: pinky 0-3, ring 4-7, middle 8-11,
                 index 12-15, thumb 16-19)

Every assigned cell gets normalised coordinates in its region, shared by both gloves after mirroring:
    u in [0, 1]  proximal -> distal  (palm: wrist -> knuckles, finger: base -> tip)
    v in [0, 1]  ulnar -> radial     (pinky side -> thumb side)
The layout gives region labels only, not which end of a finger patch is the tip. The default
(`finger_tip_row=22`: the patch row farthest from the palm is the fingertip) follows the drawing; pass
`finger_tip_row=17` to flip it.

Raw cells carry a slowly drifting per-cell offset (glove pre-load: ~145 of 460 cells read > 0 with the
hand open). `clean_tactile` removes a running low-percentile baseline, then thresholds.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGIONS = ("palm", "thumb", "index", "middle", "ring", "pinky")
GRID_SHAPE = (23, 20)


@dataclass(frozen=True)
class GloveLayout:
    side: str
    channels: np.ndarray  # (C,) raw sensor ids of the assigned cells (C = 360)
    rows: np.ndarray  # (C,)
    cols: np.ndarray  # (C,)
    region: np.ndarray  # (C,) region name
    u: np.ndarray  # (C,) proximal -> distal in [0, 1]
    v: np.ndarray  # (C,) ulnar -> radial in [0, 1]
    n_raw: int = 460

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def to_image(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """(..., C) channel values -> (..., 23, 20) glove image."""
        img = np.full(values.shape[:-1] + GRID_SHAPE, fill, dtype=values.dtype)
        img[..., self.rows, self.cols] = values
        return img

    def region_mask(self, name: str) -> np.ndarray:
        return self.region == name


def load_glove_layout(mapping_json: str | Path, side: str, finger_tip_row: int = 22) -> GloveLayout:
    with open(mapping_json) as f:
        sensors = json.load(f)["sensors"]
    ids, rows, cols, region = [], [], [], []
    for s in sensors:
        r = s[f"{side}_region"]
        if r == "unassigned":
            continue
        ids.append(s["sensor_id"]), rows.append(s["row"]), cols.append(s["column"]), region.append(r)
    ids, rows, cols, region = map(np.asarray, (ids, rows, cols, region))
    u, v = np.zeros(len(ids)), np.zeros(len(ids))
    for name in REGIONS:
        m = region == name
        r_lo, r_hi = rows[m].min(), rows[m].max()
        c_lo, c_hi = cols[m].min(), cols[m].max()
        n_r, n_c = r_hi - r_lo + 1, c_hi - c_lo + 1
        if name == "palm":
            # palm rows run from the wrist (row 0) to the knuckles (row 14)
            u[m] = (rows[m] - r_lo + 0.5) / n_r
        else:
            from_tip = np.abs(rows[m] - finger_tip_row)
            u[m] = 1.0 - (from_tip + 0.5) / n_r
        # the right glove's thumb is at the high columns, the left glove's at the low columns
        radial_is_high = side == "right"
        c_rel = (cols[m] - c_lo + 0.5) / n_c
        v[m] = c_rel if radial_is_high else 1.0 - c_rel
    return GloveLayout(side=side, channels=ids, rows=rows, cols=cols, region=region, u=u, v=v)


def clean_tactile(
    raw: np.ndarray,
    fps: float,
    window_s: float = 10.0,
    percentile: float = 5.0,
    noise_margin: float = 2.0,
) -> np.ndarray:
    """Remove each cell's drifting offset: raw - baseline - noise margin, clipped at 0.

    The offset drifts slowly but also steps (the glove settles after a grasp). The baseline is the larger
    of two running `percentile`s, one over the past `window_s` and one over the next `window_s`: after a
    step, one of the two windows only sees the new level, and a contact shorter than the window leaves
    enough released frames in both. Computed on a ~4 Hz grid and interpolated back.
    """
    from scipy.ndimage import percentile_filter

    step = max(1, int(round(fps / 4)))
    coarse = raw[::step]
    size = max(3, int(round(window_s * fps / step)) | 1)
    centred = percentile_filter(coarse, percentile, size=(size, 1), mode="nearest")
    half = size // 2
    idx = np.arange(len(coarse))
    past = centred[np.clip(idx - half, 0, len(coarse) - 1)]      # window [t - 2 half, t]
    future = centred[np.clip(idx + half, 0, len(coarse) - 1)]    # window [t, t + 2 half]
    base_coarse = np.maximum(past, future)
    t_coarse = idx * step
    t = np.arange(len(raw))
    base = np.stack([np.interp(t, t_coarse, base_coarse[:, c]) for c in range(raw.shape[1])], 1)
    return np.clip(raw - base - noise_margin, 0.0, None).astype(np.float32)


def contact_from_pressure(pressure: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Binary contact per cell (Benchmark-Genesis' glove contact: a channel above a raw threshold)."""
    return pressure > threshold
