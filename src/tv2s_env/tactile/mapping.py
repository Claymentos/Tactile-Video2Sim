"""The glove's taxels on the Wuji Hand: Benchmark-Genesis' map_tactile_to_wuji.py output
(assets/tactile/wuji_taxels_<side>.npz). Each glove cell is anchored on the MANO surface (Digital-Tactile-Twin),
then carried onto the same Wuji phalanx in coordinates both hands share (the fraction along the phalanx, the
angle around it from the palmar side; palm cells by their affine coordinates on wrist, index and little
knuckles), and ray-cast onto the convex hulls Genesis collides."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TaxelSet:
    side: str
    method: str
    link: np.ndarray        # (P,) Wuji link names
    pos: np.ndarray         # (P, 3) link frame
    normal: np.ndarray      # (P, 3) link frame, outward
    channel: np.ndarray     # (P,) index into the glove layout's assigned channels
    n_channels: int
