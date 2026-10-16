"""Glove cells -> MANO surface -> Wuji taxels: the dataset-agnostic tactile mapping ("mano").

MANO is the pivot every glove is anchored on once: a glove adapter puts each of its cells on the flat
MANO hand (an anchor: 3 vertex ids + barycentric weights on MANO's 778-vertex surface, and the MANO bone it
rides on). The anchors are then carried onto the flat Wuji hand bone by bone (Benchmark-Genesis'
rl_recon/map_tactile_to_wuji.py), in coordinates both hands share:
  - a finger anchor keeps its phalanx, its fraction s along the phalanx axis and its angle around that
    axis, measured from the phalanx's palmar direction towards the radial side; the taxel is cast onto
    the Wuji link's collision hull along that direction (tactile/mapping.py: WujiSurface.finger_point);
  - a palm anchor keeps its affine coordinates on (wrist, index MCP, little MCP) and is cast along the
    palm normal (WujiSurface.palm_point).
The correspondence is computed once, both hands flat. Placing taxels per frame through the retargeting
does not work: the retargeter matches joint positions only, so a phalanx's roll about its own axis is
free and the taxels come out 40-110 deg off the pads (Benchmark-Genesis). The taxels ride on the Wuji
links, so they follow whatever pose the retargeting or the policy gives the hand.

Glove adapters: `humantouch_anchors` (any GloveLayout's regions and normalised cell coordinates: HumanTouch's
23x20 layout, EgoTouch's 21x21 one).
MANO model files (mano_{right,left}.npz, MANO's licence forbids redistributing them) are read from
MANO_DIR (default: Benchmark-Genesis' Digital-Tactile-Twin/sim/assets).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from tv2s_data.glove import GloveLayout
from tv2s_env.robots.wuji_model import WujiHand
from tv2s_env.tactile.mapping import (
    GLOVE_TO_WUJI_FINGER,
    PHALANGES,
    TaxelSet,
    WujiSurface,
    _cell_angle,
    _palm_affine,
    _perp,
    _unit,
)

MANO_DIR = Path(os.environ.get("MANO_DIR", Path.home() / "Benchmark-Genesis/reconstruction/Digital-Tactile-Twin/sim/assets"))
TIP_VIDS = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "little": 673}
# MANO joints: 0 wrist, 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb (CMC, MCP, IP)
MANO_CHAIN = {"index": [1, 2, 3], "middle": [4, 5, 6], "little": [7, 8, 9], "ring": [10, 11, 12], "thumb": [13, 14, 15]}
BONE_TO_FINGER = {b: (f, k) for f, chain in MANO_CHAIN.items() for k, b in enumerate(chain)}


@dataclass
class ManoAnchors:
    """Glove cells on the flat MANO surface."""
    side: str
    vids: np.ndarray        # (N, 3) MANO vertex ids
    bary: np.ndarray        # (N, 3) barycentric weights
    bone: np.ndarray        # (N,) MANO joint the cell rides on (0: palm)
    channel: np.ndarray     # (N,) glove channel
    n_channels: int

    def save(self, path: str) -> None:
        np.savez(path, side=self.side, vids=self.vids, bary=self.bary, bone=self.bone, channel=self.channel,
                 n_channels=self.n_channels)

    @staticmethod
    def load(path: str) -> "ManoAnchors":
        d = np.load(path)
        return ManoAnchors(str(d["side"]), d["vids"], d["bary"], d["bone"], d["channel"], int(d["n_channels"]))


class FlatMano:
    """The flat MANO hand (mean shape, zero pose), with the same reference directions as WujiSurface."""

    def __init__(self, side: str):
        m = np.load(MANO_DIR / f"mano_{side}.npz")
        self.side = side
        self.V = m["v_template"].astype(float)
        self.F = m["faces"]
        self.J = m["J_regressor"] @ self.V
        self.mesh = trimesh.Trimesh(self.V, self.F, process=False)
        # palmar normal: cross(index MCP, little MCP) from the wrist points palmar on a right hand
        n = _unit(np.cross(self.J[1] - self.J[0], self.J[7] - self.J[0]))
        self.normal = n if side == "right" else -n
        self.radial = _unit(self.J[1] - self.J[7])                       # little -> index: towards the thumb

    def segment(self, finger: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Start and end of phalanx k (0 proximal) of a finger."""
        chain = MANO_CHAIN[finger]
        end = self.J[chain[k + 1]] if k < 2 else self.V[TIP_VIDS[finger]]
        return self.J[chain[k]], end

    def palmar(self, finger: str, k: int) -> np.ndarray:
        return self.normal                                               # MANO is flat: the palm's for all

    def frame(self, finger: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phalanx axis u, palmar p and radial r directions (orthonormal)."""
        a, b = self.segment(finger, k)
        u = _unit(b - a)
        p = _perp(self.palmar(finger, k), u)
        r = _perp(self.radial, u)
        return u, p, _unit(r - (r @ p) * p)

    def cast_out(self, origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, int]:
        """First crossing of the skin from inside the hand along `direction` (point, face)."""
        locs, _, tri = self.mesh.ray.intersects_location([origin], [direction])
        if len(locs) == 0:
            p, _, t = trimesh.proximity.closest_point(self.mesh, [origin + 0.01 * direction])
            return p[0], int(t[0])
        k = int(np.argmin(np.linalg.norm(locs - origin, axis=1)))
        return locs[k], int(tri[k])

    def anchor(self, point: np.ndarray, face: int) -> tuple[np.ndarray, np.ndarray]:
        vids = self.F[face]
        bary = trimesh.triangles.points_to_barycentric(self.V[vids][None], point[None])[0]
        return vids, bary


def humantouch_anchors(layout: GloveLayout, angle_span: float = np.radians(140),
                       palm_wrist_ratio: float = 0.7) -> ManoAnchors:
    """Glove cells on MANO (HumanTouch, EgoTouch: any GloveLayout): a finger cell at the fraction u of the finger's length (over its
    three phalanges) and the angle (v - 0.5) * angle_span around the phalanx from the palmar side; a palm
    cell at its place on the palm trapezoid (wrist -> knuckles, little -> index), cast from inside the
    hand onto the palmar skin."""
    mano = FlatMano(layout.side)
    vids, bary, bone = [], [], []
    for c in range(layout.n_channels):
        region, u, v = layout.region[c], float(layout.u[c]), float(layout.v[c])
        if region == "palm":
            a, b = _palm_affine(u, v, mano.J[0], mano.J[1], mano.J[7], mano.normal, palm_wrist_ratio)
            inside = mano.J[0] + a * (mano.J[1] - mano.J[0]) + b * (mano.J[7] - mano.J[0])
            point, face = mano.cast_out(inside, mano.normal)
            b_id = 0
        else:
            finger = GLOVE_TO_WUJI_FINGER[region]
            lengths = np.array([np.linalg.norm(np.subtract(*mano.segment(finger, k))) for k in range(3)])
            arc = u * lengths.sum()
            k = int(min(np.searchsorted(np.cumsum(lengths), arc, side="right"), 2))
            s = float(np.clip((arc - np.r_[0, np.cumsum(lengths)][k]) / lengths[k], 0, 1))
            a_pt, b_pt = mano.segment(finger, k)
            ua, p, r = mano.frame(finger, k)
            angle = _cell_angle(v, angle_span)
            point, face = mano.cast_out(a_pt + s * (b_pt - a_pt), np.cos(angle) * p + np.sin(angle) * r)
            b_id = MANO_CHAIN[finger][k]
        vi, ba = mano.anchor(point, face)
        vids.append(vi), bary.append(ba), bone.append(b_id)
    return ManoAnchors(layout.side, np.array(vids), np.array(bary), np.array(bone),
                       np.arange(layout.n_channels), layout.n_channels)


def mano_to_wuji(hand: WujiHand, anchors: ManoAnchors) -> TaxelSet:
    """Carry MANO anchors onto the flat Wuji hand, bone by bone (module docstring)."""
    mano = FlatMano(anchors.side)
    surf = WujiSurface(hand)
    X = (anchors.bary[:, :, None] * mano.V[anchors.vids]).sum(1)
    A = np.stack([mano.J[1] - mano.J[0], mano.J[7] - mano.J[0], mano.normal], 1)
    links, pos, nrm = [], [], []
    for x, b in zip(X, anchors.bone):
        if b == 0:
            a_, b_, _ = np.linalg.solve(A, x - mano.J[0])
            link, p, n = surf.palm_point(a_, b_)
        else:
            finger, k = BONE_TO_FINGER[int(b)]
            a_pt, b_pt = mano.segment(finger, k)
            u, pdir, rdir = mano.frame(finger, k)
            rel = x - a_pt
            s = float(np.clip(rel @ u / np.linalg.norm(b_pt - a_pt), 0, 1))
            radial = rel - (rel @ u) * u
            angle = float(np.arctan2(radial @ rdir, radial @ pdir))
            link, p, n = surf.finger_point(finger, PHALANGES[k], s, angle)
        links.append(link), pos.append(p), nrm.append(n)
    return TaxelSet(side=hand.side, method="mano", link=np.array(links), pos=np.array(pos), normal=np.array(nrm),
                    channel=anchors.channel.copy(), n_channels=anchors.n_channels)
